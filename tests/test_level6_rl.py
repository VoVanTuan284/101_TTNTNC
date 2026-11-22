"""
tests/test_level6_rl.py
Kiểm thử tác tử Học tăng cường Cấp 6 (KRK, KQK, KPK) và kịch bản train_rl_krk.py.
"""

import json
from pathlib import Path
import chess

from chess_ai.agents.level6_reinforcement import (
    Level6ReinforcementAgent,
    detect_endgame_type,
    encode_endgame_state_and_move,
)
from chess_ai.core.chess_env import ChessEnv
from chess_ai.training.train_rl_krk import train_krk_rl


def test_endgame_detection_and_symmetry() -> None:
    assert detect_endgame_type(chess.Board()) is None

    krk = chess.Board("4k3/8/8/8/8/8/R7/4K3 w - - 0 1")
    kqk = chess.Board("4k3/8/8/8/8/8/Q7/4K3 w - - 0 1")
    kpk = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    assert detect_endgame_type(krk) == "KRK"
    assert detect_endgame_type(kqk) == "KQK"
    assert detect_endgame_type(kpk) == "KPK"

    # Đối xứng gương dọc (lật ^ 56) của KRK cho ra cùng canonical state_key
    krk_a1 = chess.Board("8/8/8/8/8/8/1R6/K6k w - - 0 1")
    krk_a8 = chess.Board("K6k/1R6/8/8/8/8/8/8 w - - 0 1")
    enc1 = encode_endgame_state_and_move(krk_a1)
    enc2 = encode_endgame_state_and_move(krk_a8)
    assert enc1 is not None and enc2 is not None
    assert enc1[0] == enc2[0]


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
    )
    assert q_path.exists()
    assert m_path.exists()
    loaded_json = json.loads(m_path.read_text(encoding="utf-8"))
    assert set(loaded_json["tabular_q_curves"].keys()) == {"KQK", "KRK", "KPK"}
    assert len(loaded_json["reinforce_krk_curve"]) >= 1
    assert loaded_json["hyperparameters"]["tabular_q_episodes_per_endgame"] == 120
    assert "hardware_environment" in loaded_json
    assert "stage_summary" in loaded_json
    assert res["hyperparameters"]["tabular_q_episodes_per_endgame"] == 120

    agent = Level6ReinforcementAgent(q_table_path=str(q_path))
    for fen in (
        "4k3/8/8/8/8/8/R7/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/Q7/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    ):
        env = ChessEnv(fen=fen)
        move = agent.select_move(env)
        assert move in env.legal_actions()
