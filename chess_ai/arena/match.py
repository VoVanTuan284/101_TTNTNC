"""
chess_ai/arena/match.py
Hàm điều phối một ván đấu đơn lẻ giữa hai tác tử (Single-Game Runner),
dùng chung cho:
1. Giải đấu Round-Robin (chess_ai/arena/tournament.py).
2. Thực nghiệm Ablation 2 (chess_ai/experiments/ablation_mcts.py).
3. Đánh giá Checkpoint Arena Gating của Cấp 8 (chess_ai/training/selfplay.py).
4. Chế độ Máy đấu Máy trên giao diện Pygame (chess_ai/ui/desktop_pygame.py).
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

import time
from typing import Any, Dict, List, Optional
import chess

from chess_ai.agents.base_agent import BaseAgent
from chess_ai.arena.metrics import compute_latency_stats
from chess_ai.core.chess_env import ChessEnv
from chess_ai.utils.seeding import set_global_seed


def _termination_reason(env: ChessEnv, hit_max_plies: bool) -> str:
    """Xác định lý do kết thúc ván cờ theo chuẩn FIDE."""
    board = env.board
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient_material"
    if board.is_seventyfive_moves() or board.halfmove_clock >= 100:
        return "fifty_moves"
    if board.is_repetition(3) or board.can_claim_threefold_repetition():
        return "threefold_repetition"
    if hit_max_plies:
        return "max_plies"
    return "draw"


def play_single_match(
    white_agent: BaseAgent,
    black_agent: BaseAgent,
    initial_fen: str = chess.STARTING_FEN,
    time_limit_sec: Optional[float] = None,
    max_workload: Optional[int] = None,
    max_plies: int = 400,
    seed: int = 42,
    white_workload: Optional[int] = None,
    black_workload: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Điều phối một ván đấu hoàn chỉnh từ initial_fen đến khi kết thúc hoặc chạm max_plies.
    Trả về dictionary chứa kết quả, danh sách nước đi, độ trễ từng nước và thống kê tìm kiếm.
    """
    set_global_seed(seed)
    white_agent.reset()
    black_agent.reset()

    env = ChessEnv(fen=initial_fen)
    moves_uci: List[str] = []

    w_latencies_ms: List[float] = []
    b_latencies_ms: List[float] = []
    w_nodes_total = 0.0
    b_nodes_total = 0.0
    w_sims_total = 0.0
    b_sims_total = 0.0

    w_wl = white_workload if white_workload is not None else max_workload
    b_wl = black_workload if black_workload is not None else max_workload

    plies = 0
    while not env.is_terminal() and plies < max_plies:
        is_white_turn = env.board.turn == chess.WHITE
        agent = white_agent if is_white_turn else black_agent
        workload = w_wl if is_white_turn else b_wl

        t0 = time.perf_counter()
        move = agent.select_move(env, time_limit=time_limit_sec, max_workload=workload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        env.step(move)
        moves_uci.append(move.uci())
        plies += 1

        stats = agent.last_search_stats or {}
        nodes = float(stats.get("nodes", 0.0))
        sims = float(stats.get("simulations", 0.0))

        if is_white_turn:
            w_latencies_ms.append(elapsed_ms)
            w_nodes_total += nodes
            w_sims_total += sims
        else:
            b_latencies_ms.append(elapsed_ms)
            b_nodes_total += nodes
            b_sims_total += sims

    hit_max_plies = not env.is_terminal() and plies >= max_plies
    res = env.result() if env.is_terminal() else 0.0
    if res == 1.0:
        result_str = "1-0"
        white_score, black_score = 1.0, 0.0
    elif res == -1.0:
        result_str = "0-1"
        white_score, black_score = 0.0, 1.0
    else:
        result_str = "1/2-1/2"
        white_score, black_score = 0.5, 0.5

    w_lat = compute_latency_stats(w_latencies_ms)
    b_lat = compute_latency_stats(b_latencies_ms)
    w_total_sec = sum(w_latencies_ms) / 1000.0
    b_total_sec = sum(b_latencies_ms) / 1000.0

    return {
        "opening_fen": initial_fen,
        "white": white_agent.name,
        "black": black_agent.name,
        "result": result_str,
        "white_score": white_score,
        "black_score": black_score,
        "termination": _termination_reason(env, hit_max_plies),
        "total_plies": plies,
        "random_seed": seed,
        "white_avg_latency_ms": round(w_lat["avg_ms"], 3),
        "white_p95_latency_ms": round(w_lat["p95_ms"], 3),
        "black_avg_latency_ms": round(b_lat["avg_ms"], 3),
        "black_p95_latency_ms": round(b_lat["p95_ms"], 3),
        "white_avg_nps": round(w_nodes_total / w_total_sec, 2) if w_total_sec > 0 else 0.0,
        "black_avg_nps": round(b_nodes_total / b_total_sec, 2) if b_total_sec > 0 else 0.0,
        "white_avg_sps": round(w_sims_total / w_total_sec, 2) if w_total_sec > 0 else 0.0,
        "black_avg_sps": round(b_sims_total / b_total_sec, 2) if b_total_sec > 0 else 0.0,
        "moves_uci": moves_uci,
    }
