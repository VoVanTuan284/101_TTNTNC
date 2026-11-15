"""
chess_ai/search/transposition.py
Cấu trúc Bảng chuyển vị (Transposition Table) lưu trữ cờ EXACT, LOWERBOUND, UPPERBOUND
dựa trên khóa băm Zobrist 64-bit từ chess_ai.core.zobrist.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from dataclasses import dataclass
from typing import List, Optional
import chess


EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2


@dataclass(slots=True)
class TTEntry:
    """Cấu trúc một mục lưu trong Transposition Table."""

    zobrist_key: int
    depth: int
    value: float
    flag: int
    best_move: Optional[chess.Move] = None


class TranspositionTable:
    """Bảng chuyển vị Zobrist giới hạn kích thước bộ nhớ bằng mảng băm cố định."""

    _BYTES_PER_SLOT = 128

    def __init__(self, max_size_mb: int = 64) -> None:
        """Khởi tạo bảng chuyển vị với dung lượng tối đa max_size_mb."""
        target_slots = max(1024, (max(1, max_size_mb) * 1024 * 1024) // self._BYTES_PER_SLOT)
        # Làm tròn xuống lũy thừa của 2 để tra cứu bằng phép AND bit cực nhanh
        capacity = 1 << (target_slots.bit_length() - 1)
        self.capacity: int = capacity
        self._mask: int = capacity - 1
        self._table: List[Optional[TTEntry]] = [None] * capacity
        self.hits: int = 0
        self.lookups: int = 0

    def lookup(self, key: int) -> Optional[TTEntry]:
        """Tra cứu trạng thái theo khóa Zobrist 64-bit."""
        self.lookups += 1
        entry = self._table[key & self._mask]
        if entry is not None and entry.zobrist_key == key:
            self.hits += 1
            return entry
        return None

    def store(self, entry: TTEntry) -> None:
        """Lưu mục mới vào bảng chuyển vị theo chiến lược ưu tiên độ sâu (Depth-preferred)."""
        idx = entry.zobrist_key & self._mask
        existing = self._table[idx]
        if (
            existing is None
            or existing.zobrist_key == entry.zobrist_key
            or entry.depth >= existing.depth
        ):
            self._table[idx] = entry

    def clear(self) -> None:
        """Xóa toàn bộ bảng khi bắt đầu ván mới."""
        self._table = [None] * self.capacity
        self.hits = 0
        self.lookups = 0
