"""
chess_ai/search/transposition.py
Cấu trúc Bảng chuyển vị (Transposition Table) lưu trữ cờ EXACT, LOWERBOUND, UPPERBOUND
dựa trên khóa băm Zobrist 64-bit từ chess_ai.core.zobrist.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from dataclasses import dataclass
from typing import Optional
import chess


EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2


@dataclass
class TTEntry:
    """Cấu trúc một mục lưu trong Transposition Table."""
    zobrist_key: int
    depth: int
    value: float
    flag: int
    best_move: Optional[chess.Move] = None


class TranspositionTable:
    """Bảng chuyển vị Zobrist giới hạn kích thước bộ nhớ."""

    def __init__(self, max_size_mb: int = 64) -> None:
        """Khởi tạo bảng chuyển vị."""
        raise NotImplementedError

    def lookup(self, key: int) -> Optional[TTEntry]:
        """Tra cứu trạng thái theo khóa Zobrist 64-bit."""
        raise NotImplementedError

    def store(self, entry: TTEntry) -> None:
        """Lưu mục mới vào bảng chuyển vị."""
        raise NotImplementedError

    def clear(self) -> None:
        """Xóa toàn bộ bảng khi bắt đầu ván mới."""
        raise NotImplementedError

