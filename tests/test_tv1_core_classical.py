"""
tests/test_tv1_core_classical.py
Bộ kiểm thử đơn vị xác minh phần Core (ChessEnv, Evaluation, Zobrist)
và các cấp độ tác tử cổ điển (Level 1 -> Level 4) do TV1 triển khai.
"""

import chess
import pytest

from chess_ai.agents.level1_random import Level1RandomAgent
from chess_ai.agents.level2_heuristic import Level2HeuristicAgent
from chess_ai.agents.level3_alphabeta import Level3AlphaBetaAgent
from chess_ai.agents.level4_mcts import Level4MCTSAgent
from chess_ai.core.chess_env import ChessEnv
from chess_ai.core.evaluation import MATE_VALUE, evaluate_board, evaluate_board_tanh
from chess_ai.core.zobrist import compute_zobrist_hash
from chess_ai.search.alphabeta import search_alphabeta
from chess_ai.search.transposition import EXACT, TranspositionTable, TTEntry


def test_chess_env_fide_basics() -> None:
    env = ChessEnv()
    assert env.current_player() == 1
    assert not env.is_terminal()
    assert env.result() is None
    assert len(env.legal_actions()) == 20

    move = chess.Move.from_uci("e2e4")
    board, reward, done, info = env.step(move)
    assert env.current_player() == -1
    assert reward == 0.0
    assert not done
    assert info["move_count"] == 1
    assert board.peek() == move

    cloned = env.clone()
    cloned.step(chess.Move.from_uci("e7e5"))
    assert len(env.board.move_stack) == 1
    assert len(cloned.board.move_stack) == 2

    cloned.undo()
    assert len(cloned.board.move_stack) == 1


def test_chess_env_illegal_move_and_checkmate() -> None:
    # Fool's mate sau 1. f3 e5 2. g4 Qh4#
    env = ChessEnv()
    for uci in ("f2f3", "e7e5", "g2g4"):
        _, _, done, _ = env.step(chess.Move.from_uci(uci))
        assert not done

    with pytest.raises(ValueError):
        env.step(chess.Move.from_uci("e1e2"))

    _, reward, done, _ = env.step(chess.Move.from_uci("d8h4"))
    assert done
    assert env.is_terminal()
    assert env.is_terminal_fast()
    assert env.result() == -1.0
    assert reward == 1.0  # Góc nhìn người vừa đi (Đen thắng)


def test_evaluation_symmetry_and_material() -> None:
    start_board = chess.Board()
    assert abs(evaluate_board(start_board)) < 1e-6

    # Trắng hơn 1 Hậu
    no_black_queen = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert evaluate_board(no_black_queen) > 800.0
    assert evaluate_board_tanh(no_black_queen) > 0.7

    # Đảo lượt sang Đen -> điểm đối xứng âm
    no_black_queen_b = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    assert evaluate_board(no_black_queen_b) < -800.0


def test_transposition_table_store_lookup_clear() -> None:
    tt = TranspositionTable(max_size_mb=1)
    board = chess.Board()
    key = compute_zobrist_hash(board)
    move = chess.Move.from_uci("e2e4")

    tt.store(TTEntry(zobrist_key=key, depth=3, value=25.0, flag=EXACT, best_move=move))
    found = tt.lookup(key)
    assert found is not None
    assert found.best_move == move
    assert found.value == 25.0
    assert tt.hits == 1

    tt.clear()
    assert tt.lookup(key) is None


def test_alphabeta_pruning_reduces_nodes_vs_minimax() -> None:
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    move_mm, stats_mm = search_alphabeta(
        board,
        max_depth=2,
        use_tt=False,
        use_move_ordering=False,
        use_quiescence=False,
        use_alpha_beta=False,
    )
    move_ab, stats_ab = search_alphabeta(
        board,
        max_depth=2,
        use_tt=True,
        use_move_ordering=True,
        use_quiescence=False,
        use_alpha_beta=True,
    )
    assert move_mm in board.legal_moves
    assert move_ab in board.legal_moves
    assert stats_ab["eval"] == pytest.approx(stats_mm["eval"], abs=1e-5)
    assert stats_ab["nodes"] < stats_mm["nodes"]


@pytest.mark.parametrize(
    "agent_cls",
    [Level1RandomAgent, Level2HeuristicAgent, Level3AlphaBetaAgent, Level4MCTSAgent],
)
def test_classical_agents_legal_moves(agent_cls) -> None:
    env = ChessEnv()
    agent = agent_cls()
    move = agent.select_move(env, time_limit=0.5, max_workload=100)
    assert move in env.legal_actions()
    assert "nodes" in agent.last_search_stats
    assert "time_sec" in agent.last_search_stats
    agent.reset()


def test_mate_in_one_levels_2_3_4() -> None:
    # Thế cờ chiếu hết trong 1 nước: Qh5xf7# (Scholar's Mate)
    mate_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    expected = chess.Move.from_uci("h5f7")

    for agent in (
        Level2HeuristicAgent(),
        Level3AlphaBetaAgent(default_depth=2),
        Level4MCTSAgent(simulations=120, seed=42),
    ):
        env = ChessEnv(fen=mate_fen)
        move = agent.select_move(env)
        assert move == expected, f"{agent.name} chọn {move} thay vì {expected}"


def test_level4_subtree_reuse() -> None:
    env = ChessEnv()
    agent = Level4MCTSAgent(simulations=50, subtree_reuse=True, seed=42)
    m1 = agent.select_move(env, max_workload=50)
    assert agent._root is not None
    env.step(m1)
    # Đối thủ đi nước phổ biến
    opp_move = env.legal_actions()[0]
    env.step(opp_move)
    m2 = agent.select_move(env, max_workload=50)
    assert m2 in env.legal_actions()
    agent.reset()
    assert agent._root is None
