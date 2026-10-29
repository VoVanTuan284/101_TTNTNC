"""
chess_ai/core/zobrist.py
Hàm băm Zobrist 64-bit thống nhất dùng chung cho:
1. Bảng chuyển vị Transposition Table trong tìm kiếm Alpha-Beta (chess_ai/search/transposition.py).
2. Phát hiện lặp lại thế cờ trên đường đi của cây tìm kiếm.
3. Khử trùng lặp thế cờ (Deduplication) trong đường ống ETL (chess_ai/etl/pgn_etl.py).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

import chess
import chess.polyglot


def compute_zobrist_hash(board: chess.Board) -> int:
    """Tính khóa băm Zobrist 64-bit chuẩn cho trạng thái bàn cờ hiện tại."""
    return chess.polyglot.zobrist_hash(board)
