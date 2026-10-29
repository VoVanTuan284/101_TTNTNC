"""
chess_ai/utils/epd.py
Bộ đọc tệp thế cờ chuẩn EPD dùng chung cho bộ 50 thế cờ chiến thuật WAC-50
(resources/benchmark_positions.epd) và bộ 50 thế cờ khai cuộc cân bằng (resources/openings_50.epd).
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

from dataclasses import dataclass
from typing import List
import chess


@dataclass
class EPDRecord:
    """Thông tin một dòng thế cờ EPD đã phân tích cú pháp."""
    record_id: str
    fen: str
    best_moves: List[chess.Move]


def load_epd_file(filepath: str) -> List[EPDRecord]:
    """Đọc tệp .epd và trả về danh sách các bản ghi EPDRecord hợp lệ."""
    raise NotImplementedError

