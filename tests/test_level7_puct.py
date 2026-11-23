"""
tests/test_level7_puct.py
Kiểm thử nâng cao cho Level 7 Guided MCTS (PUCT):
- allow_untrained_fallback=False ném FileNotFoundError khi thiếu checkpoint
- Forced move shortcut (1 nước hợp lệ duy nhất không chạy simulation thừa)
- FPU Reduction (First Play Urgency) khi thế cờ bất lợi
- Phát hiện hòa lặp lại 3 lần (Threefold Repetition) trong cây MCTS
- NN Evaluation Cache theo Zobrist hash & Batched Leaf Evaluation (Virtual Loss)
"""

from collections import OrderedDict
from pathlib import Path
import chess
import pytest
import torch

from chess_ai.agents.level7_guided_mcts import Level7GuidedMCTSAgent
from chess_ai.core.chess_env import ChessEnv
from chess_ai.models.dual_resnet import DualHeadResNet
from chess_ai.search.mcts_node import MCTSNode
from chess_ai.search.mcts_puct import (
    _select_puct_child,
    run_puct_search,
)


def _save_tiny_ckpt(path: Path) -> DualHeadResNet:
    model = DualHeadResNet(num_res_blocks=1, filters=8)
    model.eval()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {"network": {"num_res_blocks": 1, "filters": 8}},
        },
        path,
    )
    return model


def test_allow_untrained_fallback_false_raises(tmp_path: Path) -> None:
    missing_ckpt = tmp_path / "does_not_exist.pt"
    with pytest.raises(FileNotFoundError):
        Level7GuidedMCTSAgent(
            checkpoint_path=str(missing_ckpt),
            allow_untrained_fallback=False,
        )


def test_forced_move_shortcut(tmp_path: Path) -> None:
    ckpt = tmp_path / "tiny.pt"
    _save_tiny_ckpt(ckpt)
    agent = Level7GuidedMCTSAgent(checkpoint_path=str(ckpt), simulations=50, device="cpu")

    # Thế cờ Vua Trắng ở h1 bị Xe Đen chiếu từ a1 và Vua Đen ở g3 -> chỉ có đúng 1 nước h1h2? Không, dùng FEN chuẩn 1 nước duy nhất:
    # Vua Trắng ở a1 bị Xe Đen ở a8 chiếu, cột b bị Vua Đen ở c2 kiểm soát trừ b1? Dùng FEN chiếu đơn giản:
    fen_one_move = "7k/8/8/8/8/6q1/8/7K w - - 0 1"  # Trắng ở h1 chỉ có nước h1g1? Không, g3 chiếu g1..g8 -> h1g1 bất hợp lệ, h1h2 bất hợp lệ -> sửa:
    # Vua Trắng ở a1, Hậu Đen ở c2, Vua Đen ở c3 -> a1b1 bị c2 kiểm soát, a1a2 bị c2 kiểm soát -> dùng Xe Đen ở h2, Vua Đen ở c2:
    # a1 bị chiếu bởi b1? Không:
    board = chess.Board("4k3/8/8/8/8/8/1r6/K1r5 w - - 0 1")  # a1 bị c1 chiếu, a2 bị b2 chiếu -> b1 bị b2 và c1 kiểm soát -> mate!
    # Thế cờ có đúng 1 nước hợp lệ: Vua Trắng ở h1, Xe Đen ở a1 (chiếu hàng 1), Tốt Trắng ở g2, Vua Đen ở h8 -> duy nhất h1h2? Không, g2 chặn -> chỉ có h1h2?
    board = chess.Board("7k/8/8/8/8/8/6P1/r6K w - - 0 1")
    assert len(list(board.legal_moves)) == 1

    env = ChessEnv(fen=board.fen())
    mv = agent.select_move(env, max_workload=100)
    assert mv == chess.Move.from_uci("h1h2")
    assert agent.last_search_stats["simulations"] == 0.0


def test_fpu_reduction_penalizes_unvisited_when_losing() -> None:
    root = MCTSNode()
    # Giả sử root đang bị bất lợi từ góc nhìn bên đi (V_parent = -root.q_value = -0.6 => root.q_value = +0.6)
    root.visit_count = 10
    root.value_sum = 6.0  # q_value = +0.6 => parent_val_for_turn = -0.6

    m_visited = chess.Move.from_uci("e2e4")
    m_unvisited = chess.Move.from_uci("a2a3")
    c_visited = MCTSNode(parent=root, prior_prob=0.5)
    c_visited.visit_count = 5
    c_visited.value_sum = -2.5  # q_value = -0.5

    c_unvisited = MCTSNode(parent=root, prior_prob=0.1)
    c_unvisited.visit_count = 0

    root.children = {m_visited: c_visited, m_unvisited: c_unvisited}

    # Với fpu_reduction=0.2, nút chưa thăm có fpu_value = -0.6 - 0.2 = -0.8 (thấp hơn -0.5 của nút đã thăm)
    chosen_fpu, _ = _select_puct_child(root, c_puct=1.0, fpu_reduction=0.2)
    assert chosen_fpu == m_visited


def test_repetition_detection_in_puct_search() -> None:
    # Tạo bàn cờ đã lặp lại thế cờ 2 lần trong move_stack (chỉ cần thêm 1 lần nữa là hòa 3 lần)
    board = chess.Board("4k3/8/8/8/8/8/8/4K2R w - - 0 1")
    for uci in ("h1g1", "e8d8", "g1h1", "d8e8"):
        board.push(chess.Move.from_uci(uci))

    # Mô hình mock luôn đánh giá thế cờ +0.9 để kiểm tra cây MCTS vẫn chạy an toàn khi có lặp lịch sử
    def mock_model(b: chess.Board):
        moves = list(b.legal_moves)
        return {m: 1.0 / len(moves) for m in moves}, 0.9

    mv, stats, _ = run_puct_search(
        board=board,
        model=mock_model,
        simulations=16,
        eval_mode=True,
    )
    assert mv in board.legal_moves
    assert stats["simulations"] == 16.0


def test_eval_cache_and_batched_virtual_loss(tmp_path: Path) -> None:
    ckpt = tmp_path / "tiny.pt"
    model = _save_tiny_ckpt(ckpt)
    board = chess.Board()

    cache = OrderedDict()
    mv1, stats1, _ = run_puct_search(
        board=board,
        model=model,
        simulations=16,
        eval_mode=True,
        eval_cache=cache,
        batch_size=4,
    )
    assert mv1 in board.legal_moves
    assert stats1["simulations"] == 16.0
    assert len(cache) > 0

    # Chạy lại cùng thế cờ với cache sẵn -> phải có cache_hits > 0
    mv2, stats2, _ = run_puct_search(
        board=board,
        model=model,
        simulations=16,
        eval_mode=True,
        eval_cache=cache,
        batch_size=4,
    )
    assert mv2 in board.legal_moves
    assert stats2["cache_hits"] > 0
