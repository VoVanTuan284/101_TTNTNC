"""
chess_ai/search/quiescence.py
Thuật toán Tìm kiếm Tĩnh (Quiescence Search) mở rộng các chuỗi bắt quân và nước giải chiếu
kèm kỹ thuật cắt tỉa Delta Pruning nhằm giảm thiểu hiện tượng mù chân trời (Horizon Effect).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

import chess


def quiescence_search(
    board: chess.Board,
    alpha: float,
    beta: float,
    max_q_plies: int = 8,
) -> float:
    """Đánh giá ổn định tại nút lá bằng chuỗi bắt quân và thoát chiếu."""
    raise NotImplementedError

