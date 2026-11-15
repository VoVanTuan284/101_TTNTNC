"""
chess_ai/core/evaluation.py
Hàm lượng giá tĩnh cổ điển: Material Balance + PST Michniewski + King Safety + Mobility.
Luôn trả về điểm số theo góc nhìn tương đối của bên tới lượt đi (centipawns).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

import math
from typing import Dict, List
import chess
from chess_ai.config import load_yaml_config


_SEARCH_CFG = load_yaml_config("search_params.yaml")
_EVAL_CFG = _SEARCH_CFG["evaluation"]
_RAW_PIECE_VALUES = _EVAL_CFG["piece_values"]
_RAW_PST = _EVAL_CFG["pst"]

PIECE_TYPE_TO_CHAR: Dict[int, str] = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}

PIECE_VALUES: Dict[int, int] = {
    pt: int(_RAW_PIECE_VALUES[ch]) for pt, ch in PIECE_TYPE_TO_CHAR.items()
}

# Tổng lực lượng không tính Tốt và Vua khi khai cuộc cả 2 bên (~6400 cp)
_MAX_NON_PAWN_MATERIAL = 2 * (
    2 * PIECE_VALUES[chess.KNIGHT]
    + 2 * PIECE_VALUES[chess.BISHOP]
    + 2 * PIECE_VALUES[chess.ROOK]
    + PIECE_VALUES[chess.QUEEN]
)

MATE_VALUE: float = float(PIECE_VALUES[chess.KING])


def _build_pst_tables() -> Dict[str, Dict[bool, List[int]]]:
    """Chuyển ma trận PST 8x8 (hàng 8 -> hàng 1) thành bảng phẳng 64 ô cho Trắng và Đen."""
    tables: Dict[str, Dict[bool, List[int]]] = {}
    for key, grid in _RAW_PST.items():
        white_table = [0] * 64
        black_table = [0] * 64
        for sq in range(64):
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            white_table[sq] = int(grid[7 - rank][file])
            black_table[sq] = int(grid[rank][file])
        tables[key] = {chess.WHITE: white_table, chess.BLACK: black_table}
    return tables


_PST_TABLES = _build_pst_tables()

_NON_KING_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
)

_MOBILITY_WEIGHTS: Dict[int, int] = {
    chess.KNIGHT: 4,
    chess.BISHOP: 4,
    chess.ROOK: 2,
    chess.QUEEN: 1,
}


def _king_safety_score(board: chess.Board, color: bool, pawns_mask: int) -> int:
    """Ước lượng điểm an toàn Vua dựa trên lá chắn Tốt phía trước và độ mở cột cạnh Vua."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0

    k_rank = chess.square_rank(king_sq)
    k_file = chess.square_file(king_sq)
    direction = 1 if color == chess.WHITE else -1
    score = 0

    for df in (-1, 0, 1):
        f = k_file + df
        if not (0 <= f <= 7):
            continue
        file_mask = chess.BB_FILES[f]
        if (pawns_mask & file_mask) == 0:
            # Cột không có Tốt che chắn
            score -= 15 if df == 0 else 10
            continue

        # Thưởng nếu có Tốt đứng ngay phía trước 1 hoặc 2 ô
        r1 = k_rank + direction
        r2 = k_rank + 2 * direction
        if 0 <= r1 <= 7 and (pawns_mask & chess.BB_SQUARES[chess.square(f, r1)]):
            score += 12
        elif 0 <= r2 <= 7 and (pawns_mask & chess.BB_SQUARES[chess.square(f, r2)]):
            score += 6

    return score


def evaluate_board(board: chess.Board) -> float:
    """Tính điểm lượng giá tĩnh tương đối cho bên tới lượt đi tại trạng thái board."""
    if board.is_checkmate():
        return -MATE_VALUE
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_seventyfive_moves()
        or board.halfmove_clock >= 100
    ):
        return 0.0

    white_score = 0.0
    black_score = 0.0
    non_pawn_material = 0

    white_occ = board.occupied_co[chess.WHITE]
    black_occ = board.occupied_co[chess.BLACK]

    for pt in _NON_KING_TYPES:
        val = PIECE_VALUES[pt]
        ch = PIECE_TYPE_TO_CHAR[pt]
        w_pst = _PST_TABLES[ch][chess.WHITE]
        b_pst = _PST_TABLES[ch][chess.BLACK]
        mob_w = _MOBILITY_WEIGHTS.get(pt, 0)

        w_bb = board.pieces_mask(pt, chess.WHITE)
        while w_bb:
            lsb = w_bb & -w_bb
            sq = lsb.bit_length() - 1
            white_score += val + w_pst[sq]
            if pt != chess.PAWN:
                non_pawn_material += val
            if mob_w:
                white_score += mob_w * (board.attacks_mask(sq) & ~white_occ).bit_count()
            w_bb ^= lsb

        b_bb = board.pieces_mask(pt, chess.BLACK)
        while b_bb:
            lsb = b_bb & -b_bb
            sq = lsb.bit_length() - 1
            black_score += val + b_pst[sq]
            if pt != chess.PAWN:
                non_pawn_material += val
            if mob_w:
                black_score += mob_w * (board.attacks_mask(sq) & ~black_occ).bit_count()
            b_bb ^= lsb

    # Nội suy pha tàn cuộc: phase = 1.0 (khai/trung cuộc) -> 0.0 (tàn cuộc)
    phase = min(1.0, non_pawn_material / float(_MAX_NON_PAWN_MATERIAL))

    w_king_sq = board.king(chess.WHITE)
    if w_king_sq is not None:
        w_mg = _PST_TABLES["K_middlegame"][chess.WHITE][w_king_sq]
        w_eg = _PST_TABLES["K_endgame"][chess.WHITE][w_king_sq]
        white_score += phase * w_mg + (1.0 - phase) * w_eg
        white_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
        white_score += phase * _king_safety_score(board, chess.WHITE, white_pawns)

    b_king_sq = board.king(chess.BLACK)
    if b_king_sq is not None:
        b_mg = _PST_TABLES["K_middlegame"][chess.BLACK][b_king_sq]
        b_eg = _PST_TABLES["K_endgame"][chess.BLACK][b_king_sq]
        black_score += phase * b_mg + (1.0 - phase) * b_eg
        black_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)
        black_score += phase * _king_safety_score(board, chess.BLACK, black_pawns)

    diff_white = white_score - black_score
    return diff_white if board.turn == chess.WHITE else -diff_white


def evaluate_board_tanh(board: chess.Board, scale: float = 600.0) -> float:
    """Chuẩn hóa điểm lượng giá tĩnh về khoảng [-1.0, 1.0] qua hàm tanh (dùng cho MCTS UCT cutoff)."""
    return math.tanh(evaluate_board(board) / scale)
