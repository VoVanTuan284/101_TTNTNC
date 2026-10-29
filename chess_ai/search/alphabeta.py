"""
chess_ai/search/alphabeta.py
Thuật toán Negamax kết hợp cắt tỉa Alpha-Beta, Iterative Deepening (ID),
bảng chuyển vị Zobrist (TT) và kiểm soát thời gian hợp tác (Cooperative Time Management).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from typing import Dict, Optional, Tuple
import chess


def search_alphabeta(
    board: chess.Board,
    max_depth: int = 4,
    time_limit: Optional[float] = None,
    max_nodes: Optional[int] = None,
    use_tt: bool = True,
    use_move_ordering: bool = True,
    use_quiescence: bool = True,
) -> Tuple[chess.Move, Dict[str, float]]:
    """
    Thực hiện tìm kiếm Iterative Deepening Alpha-Beta và trả về (best_move, stats).
    """
    raise NotImplementedError

