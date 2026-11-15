"""
chess_ai/search/quiescence.py
Thuật toán Tìm kiếm Tĩnh (Quiescence Search) mở rộng các chuỗi bắt quân và nước giải chiếu
kèm kỹ thuật cắt tỉa Delta Pruning nhằm giảm thiểu hiện tượng mù chân trời (Horizon Effect).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from typing import Any, List, Optional
import chess
from chess_ai.config import load_yaml_config
from chess_ai.core.evaluation import MATE_VALUE, PIECE_VALUES, evaluate_board
from chess_ai.search.move_ordering import mvv_lva_score, order_moves


_Q_CFG = load_yaml_config("search_params.yaml")["alpha_beta"]["quiescence"]
_DEFAULT_MAX_Q_PLIES: int = int(_Q_CFG.get("max_plies", 8))
_DEFAULT_DELTA_MARGIN: float = float(_Q_CFG.get("delta_pruning_margin", 950))


def _captured_value(board: chess.Board, move: chess.Move) -> int:
    """Trả về giá trị centipawn của quân bị bắt trong nước đi."""
    if board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]
    victim = board.piece_at(move.to_square)
    return PIECE_VALUES[victim.piece_type] if victim is not None else 0


def quiescence_search(
    board: chess.Board,
    alpha: float,
    beta: float,
    max_q_plies: int = _DEFAULT_MAX_Q_PLIES,
    delta_margin: float = _DEFAULT_DELTA_MARGIN,
    q_ply: int = 0,
    search_ctx: Optional[Any] = None,
) -> float:
    """Đánh giá ổn định tại nút lá bằng chuỗi bắt quân và thoát chiếu."""
    if search_ctx is not None:
        search_ctx.tick()
        if search_ctx.stopped:
            return alpha

    if board.is_checkmate():
        return -MATE_VALUE + q_ply
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_seventyfive_moves()
        or board.halfmove_clock >= 100
    ):
        return 0.0

    in_check = board.is_check()
    stand_pat = 0.0

    if not in_check:
        stand_pat = evaluate_board(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

    if q_ply >= max_q_plies:
        return stand_pat if not in_check else evaluate_board(board)

    if in_check:
        # Khi đang bị chiếu, phải xét toàn bộ nước giải chiếu hợp lệ
        moves: List[chess.Move] = order_moves(board, list(board.legal_moves), ply=q_ply)
    else:
        captures = list(board.generate_legal_captures())
        moves = sorted(captures, key=lambda m: mvv_lva_score(board, m), reverse=True)

    for move in moves:
        # Delta Pruning: bỏ qua nước bắt quân không đủ bù đắp khoảng cách tới alpha
        if not in_check and move.promotion is None:
            gain = _captured_value(board, move)
            if stand_pat + gain + delta_margin < alpha:
                continue

        board.push(move)
        score = -quiescence_search(
            board,
            -beta,
            -alpha,
            max_q_plies=max_q_plies,
            delta_margin=delta_margin,
            q_ply=q_ply + 1,
            search_ctx=search_ctx,
        )
        board.pop()

        if search_ctx is not None and search_ctx.stopped:
            return alpha

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha
