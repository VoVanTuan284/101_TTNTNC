"""
chess_ai/search/move_ordering.py
Chiến lược sắp xếp thứ tự duyệt nước đi: PV-move từ TT -> MVV-LVA -> Killer Moves -> History Heuristic.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from typing import List, Optional
import chess


def order_moves(
    board: chess.Board,
    moves: List[chess.Move],
    pv_move: Optional[chess.Move] = None,
    ply: int = 0,
) -> List[chess.Move]:
    """Sắp xếp danh sách nước đi hợp lệ theo độ hứa hẹn chiến thuật giảm dần."""
    raise NotImplementedError

