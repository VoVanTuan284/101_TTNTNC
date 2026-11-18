"""
chess_ai/utils/epd.py
Bộ đọc tệp thế cờ chuẩn EPD dùng chung cho bộ 50 thế cờ chiến thuật WAC-50
(resources/benchmark_positions.epd) và bộ 50 thế cờ khai cuộc cân bằng (resources/openings_50.epd).
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List
import chess


_FEN4_SEMICOLON_RE = re.compile(r"^(\S+\s+[wb]\s+\S+\s+\S+);\s*")


@dataclass
class EPDRecord:
    """Thông tin một dòng thế cờ EPD đã phân tích cú pháp."""

    record_id: str
    fen: str
    best_moves: List[chess.Move]


def load_epd_file(filepath: str) -> List[EPDRecord]:
    """Đọc tệp .epd và trả về danh sách các bản ghi EPDRecord hợp lệ."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp EPD: {filepath}")

    records: List[EPDRecord] = []
    with open(path, encoding="utf-8") as f:
        for idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # Chuẩn hóa trường hợp dấu ';' đặt ngay sau 4 trường FEN (ví dụ openings_50.epd)
            normalized = _FEN4_SEMICOLON_RE.sub(r"\1 ", line, count=1)

            board = chess.Board()
            ops = board.set_epd(normalized)

            raw_id = ops.get("id", f"POS.{idx:03d}")
            record_id = str(raw_id)
            bm_val = ops.get("bm", [])
            best_moves = list(bm_val) if isinstance(bm_val, (list, tuple)) else [bm_val]

            records.append(
                EPDRecord(
                    record_id=record_id,
                    fen=board.fen(),
                    best_moves=best_moves,
                )
            )

    return records
