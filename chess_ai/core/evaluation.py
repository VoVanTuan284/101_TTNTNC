"""
chess_ai/core/evaluation.py
Hàm lượng giá tĩnh cổ điển: Material Balance + PST Michniewski + King Safety + Mobility.
Luôn trả về điểm số theo góc nhìn tương đối của bên tới lượt đi (centipawns).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

import chess


def evaluate_board(board: chess.Board) -> float:
    """Tính điểm lượng giá tĩnh tương đối cho bên tới lượt đi tại trạng thái board."""
    raise NotImplementedError


def evaluate_board_tanh(board: chess.Board, scale: float = 600.0) -> float:
    """Chuẩn hóa điểm lượng giá tĩnh về khoảng [-1.0, 1.0] qua hàm tanh (dùng cho MCTS UCT cutoff)."""
    raise NotImplementedError

