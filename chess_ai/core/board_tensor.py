"""
chess_ai/core/board_tensor.py
Biểu diễn trạng thái bàn cờ thành Tensor chuẩn hóa góc nhìn bên đi (18, 8, 8)
và ánh xạ không gian hành động phẳng 4096 chiều (ADR-02 & ADR-03).
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead) | Kiểm thử: TV3 (TC26-TC35)
"""

import chess
import numpy as np


def canonical_square(sq: int, turn: chess.Color) -> int:
    """Lật tọa độ ô cờ theo góc nhìn bên đi (sq ^ 56 nếu Đen đi)."""
    raise NotImplementedError


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Chuyển đổi chess.Board thành mảng numpy shape (18, 8, 8) dtype float32."""
    raise NotImplementedError


def board_to_uint8_planes(board: chess.Board) -> np.ndarray:
    """Chuyển đổi chess.Board thành mảng (18, 8, 8) dtype uint8 để lưu np.memmap."""
    raise NotImplementedError


def move_to_action_index(move: chess.Move, turn: chess.Color) -> int:
    """Ánh xạ nước đi sang chỉ số nguyên phẳng [0, 4095] theo góc nhìn bên đi."""
    raise NotImplementedError


def action_index_to_move(action_idx: int, board: chess.Board) -> chess.Move:
    """Giải mã chỉ số [0, 4095] về chess.Move hợp lệ (mặc định phong Hậu)."""
    raise NotImplementedError


def get_legal_action_mask(board: chess.Board) -> np.ndarray:
    """Tạo mặt nạ nhị phân shape (4096,) đánh dấu các chỉ số hành động hợp lệ."""
    raise NotImplementedError

