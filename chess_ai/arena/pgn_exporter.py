"""
chess_ai/arena/pgn_exporter.py
Module xuất biên bản ván đấu chuẩn PGN quốc tế và ghi dòng log JSONL.
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

from typing import Any, Dict, Sequence
import chess


def export_game_to_pgn(
    moves: Sequence[chess.Move],
    headers: Dict[str, str],
    initial_fen: str = chess.STARTING_FEN,
) -> str:
    """Chuyển đổi danh sách nước đi và metadata thành chuỗi biên bản PGN chuẩn."""
    raise NotImplementedError


def append_match_jsonl(filepath: str, record: Dict[str, Any]) -> None:
    """Ghi nối tiếp kết quả một ván đấu vào tệp JSONL."""
    raise NotImplementedError

