"""
chess_ai/arena/pgn_exporter.py
Module xuất biên bản ván đấu chuẩn PGN quốc tế và ghi dòng log JSONL.
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

from datetime import date
import json
from pathlib import Path
from typing import Any, Dict, Sequence
import chess
import chess.pgn


def export_game_to_pgn(
    moves: Sequence[chess.Move],
    headers: Dict[str, str],
    initial_fen: str = chess.STARTING_FEN,
) -> str:
    """Chuyển đổi danh sách nước đi và metadata thành chuỗi biên bản PGN chuẩn (đầy đủ tag, không để '?')."""
    game = chess.pgn.Game()
    defaults = {
        "Event": "101_TTNTNC Chess AI 8x8 FIDE Arena",
        "Site": "SGU IT Chess AI Lab (Ho Chi Minh City)",
        "Date": date.today().strftime("%Y.%m.%d"),
        "Round": "1",
    }
    merged = {**defaults, **headers}
    for key, val in merged.items():
        if val is not None and str(val).strip() != "":
            game.headers[key] = str(val)

    if initial_fen and initial_fen != chess.STARTING_FEN:
        board = chess.Board(initial_fen)
        game.setup(board)

    node = game
    for move in moves:
        node = node.add_variation(move)

    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return game.accept(exporter)


def append_pgn_file(filepath: str, pgn_text: str) -> None:
    """Ghi nối tiếp một ván cờ PGN vào tệp .pgn."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(pgn_text.rstrip() + "\n\n")


def append_match_jsonl(filepath: str, record: Dict[str, Any]) -> None:
    """Ghi nối tiếp kết quả một ván đấu vào tệp JSONL."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
