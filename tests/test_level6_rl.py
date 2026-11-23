"""
tests/test_level6_rl.py
Kiểm thử tác tử Học tăng cường Cấp 6 (KRK, KQK, KPK), đối xứng màu/D4,
Afterstate V(s'), REINFORCE inference và kịch bản train_rl_krk.py.
"""

import json
from pathlib import Path
import chess

from chess_ai.agents.level6_reinforcement import (
    Level6ReinforcementAgent,
    detect_endgame_type,
    encode_afterstate,
    encode_endgame_state_and_move,
    get_endgame_info,
)
from chess_ai.core.chess_env import ChessEnv
from chess_ai.training.train_rl_krk import train_krk_rl


def test_endgame_detection_and_symmetry() -> None:
    assert detect_endgame_type(chess.Board()) is None

    krk = chess.Board("4k3/8/8/8/8/8/R7/4K3 w - - 0 1")
    kqk = chess.Board("4k3/8/8/8/8/8/Q7/4K3 w - - 0 1")
    kpk = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    krk_black = chess.Board("4k3/r7/8/8/8/8/8/4K3 b - - 0 1")
    assert detect_endgame_type(krk) == "KRK"
    assert detect_endgame_type(kqk) == "KQK"
    assert detect_endgame_type(kpk) == "KPK"
    assert get_endgame_info(krk_black) == ("KRK", chess.BLACK)

    # Đối xứng gương dọc (lật ^ 56) của KRK cho ra cùng canonical state_key
    krk_a1 = chess.Board("8/8/8/8/8/8/1R6/K6k w - - 0 1")
    krk_a8 = chess.Board("K6k/1R6/8/8/8/8/8/8 w - - 0 1")
    enc1 = encode_endgame_state_and_move(krk_a1)
    enc2 = encode_endgame_state_and_move(krk_a8)
    assert enc1 is not None and enc2 is not None
    assert enc1[0] == enc2[0]
    assert encode_afterstate(krk_a1) == encode_afterstate(krk_a8)

    # Chuẩn hóa màu: KRK Trắng và bản mirror của Đen cho cùng afterstate key
    assert encode_afterstate(krk) == encode_afterstate(krk.mirror())


def test_unseen_move_not_losing_to_negative_visited_q() -> None:
    """Đảm bảo khi chỉ 1 nước bị ghi nhận Q < 0, tác tử không ép chọn nước xấu đó."""
    agent = Level6ReinforcementAgent(q_table_path=None, policy_mode="q_table")
    env = ChessEnv(fen="4k3/8/8/8/8/8/R7/4K3 w - - 0 1")
    bad_move = chess.Move.from_uci("a2a8")
    enc = encode_endgame_state_and_move(env.board, bad_move)
    assert enc is not None and enc[1] is not None
    agent.q_tables = {"KRK": {(enc[0], enc[1]): -0.9}}

    chosen = agent.select_move(env)
    assert chosen in env.legal_actions()
    assert chosen != bad_move


def test_train_rl_krk_and_agent_inference(tmp_path: Path) -> None:
    q_path = tmp_path / "l6_test.pkl"
    m_path = tmp_path / "l6_curve.json"

    res = train_krk_rl(
        output_qtable=str(q_path),
        output_metrics=str(m_path),
        episodes_override=120,
        reinforce_episodes=60,
        endgames=("KQK", "KRK", "KPK"),
        seed=42,
        eval_games=20,
        verbose=False,
    )
    assert q_path.exists()
    assert m_path.exists()
    loaded_json = json.loads(m_path.read_text(encoding="utf-8"))
    assert set(loaded_json["tabular_q_curves"].keys()) == {"KQK", "KRK", "KPK"}
    assert len(loaded_json["reinforce_krk_curve"]) >= 1
    assert loaded_json["hyperparameters"]["tabular_q_episodes_per_endgame"] == 120
    assert "hardware_environment" in loaded_json
    assert "stage_summary" in loaded_json
    assert "greedy_win_rate" in loaded_json["stage_summary"]["reinforce_KRK"]
    assert res["hyperparameters"]["tabular_q_episodes_per_endgame"] == 120

    # Kiểm tra inference ở chế độ auto (Afterstate V-table) cho cả Trắng và Đen
    agent = Level6ReinforcementAgent(q_table_path=str(q_path), policy_mode="auto")
    assert set(agent.v_tables.keys()) == {"KQK", "KRK", "KPK"}
    assert agent.reinforce_policy is not None

    for fen in (
        "4k3/8/8/8/8/8/R7/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/Q7/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        "4k3/r7/8/8/8/8/8/4K3 b - - 0 1",
    ):
        env = ChessEnv(fen=fen)
        move = agent.select_move(env)
        assert move in env.legal_actions()

    # Kiểm tra inference ở chế độ REINFORCE thuần
    reinf_agent = Level6ReinforcementAgent(q_table_path=str(q_path), policy_mode="reinforce")
    env_krk = ChessEnv(fen="4k3/8/8/8/8/8/R7/4K3 w - - 0 1")
    reinf_move = reinf_agent.select_move(env_krk)
    assert reinf_move in env_krk.legal_actions()
