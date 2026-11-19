"""
chess_ai/core/board_tensor.py
Biểu diễn trạng thái bàn cờ thành Tensor chuẩn hóa góc nhìn bên đi (18, 8, 8)
và ánh xạ không gian hành động phẳng 4096 chiều (ADR-02 & ADR-03).
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead) | Kiểm thử: TV3 (TC26-TC35)
"""

import chess
import numpy as np


_PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)


def canonical_square(sq: int, turn: chess.Color) -> int:
    """Lật tọa độ ô cờ theo góc nhìn bên đi (sq ^ 56 nếu Đen đi)."""
    return sq if turn == chess.WHITE else (sq ^ 56)


def board_to_uint8_planes(board: chess.Board) -> np.ndarray:
    """
    Chuyển đổi chess.Board thành mảng (18, 8, 8) dtype uint8 để lưu np.memmap.
    Kênh 0-5: quân bên đi; 6-11: quân đối thủ; 12-15: 4 quyền nhập thành;
    16: En-Passant; 17: min(halfmove_clock, 100) dạng số nguyên [0, 100].
    """
    planes = np.zeros((18, 8, 8), dtype=np.uint8)
    turn = board.turn
    opp = not turn

    for idx, pt in enumerate(_PIECE_TYPES):
        act_bb = board.pieces_mask(pt, turn)
        while act_bb:
            lsb = act_bb & -act_bb
            sq = lsb.bit_length() - 1
            c_sq = canonical_square(sq, turn)
            planes[idx, c_sq >> 3, c_sq & 7] = 1
            act_bb ^= lsb

        opp_bb = board.pieces_mask(pt, opp)
        while opp_bb:
            lsb = opp_bb & -opp_bb
            sq = lsb.bit_length() - 1
            c_sq = canonical_square(sq, turn)
            planes[6 + idx, c_sq >> 3, c_sq & 7] = 1
            opp_bb ^= lsb

    if board.has_kingside_castling_rights(turn):
        planes[12].fill(1)
    if board.has_queenside_castling_rights(turn):
        planes[13].fill(1)
    if board.has_kingside_castling_rights(opp):
        planes[14].fill(1)
    if board.has_queenside_castling_rights(opp):
        planes[15].fill(1)

    if board.ep_square is not None and board.has_legal_en_passant():
        ep_c = canonical_square(board.ep_square, turn)
        planes[16, ep_c >> 3, ep_c & 7] = 1

    planes[17].fill(min(int(board.halfmove_clock), 100))
    return planes


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Chuyển đổi chess.Board thành mảng numpy shape (18, 8, 8) dtype float32."""
    planes = board_to_uint8_planes(board).astype(np.float32)
    planes[17] /= 100.0
    return planes


def move_to_action_index(move: chess.Move, turn: chess.Color) -> int:
    """Ánh xạ nước đi sang chỉ số nguyên phẳng [0, 4095] theo góc nhìn bên đi."""
    u = canonical_square(move.from_square, turn)
    v = canonical_square(move.to_square, turn)
    return u * 64 + v


def action_index_to_move(action_idx: int, board: chess.Board) -> chess.Move:
    """Giải mã chỉ số [0, 4095] về chess.Move hợp lệ (mặc định phong Hậu)."""
    if not (0 <= action_idx < 4096):
        raise ValueError(f"action_idx nằm ngoài [0, 4095]: {action_idx}")

    turn = board.turn
    u_canon, v_canon = divmod(action_idx, 64)
    u = canonical_square(u_canon, turn)
    v = canonical_square(v_canon, turn)

    queen_promo = chess.Move(u, v, promotion=chess.QUEEN)
    if queen_promo in board.legal_moves:
        return queen_promo

    normal_move = chess.Move(u, v)
    if normal_move in board.legal_moves:
        return normal_move

    for move in board.legal_moves:
        if move.from_square == u and move.to_square == v:
            return move

    raise ValueError(f"Chỉ số hành động {action_idx} ({u}->{v}) không hợp lệ tại {board.fen()}")


def get_legal_action_mask(board: chess.Board) -> np.ndarray:
    """Tạo mặt nạ nhị phân shape (4096,) đánh dấu các chỉ số hành động hợp lệ."""
    mask = np.zeros(4096, dtype=np.float32)
    turn = board.turn
    for move in board.legal_moves:
        mask[move_to_action_index(move, turn)] = 1.0
    return mask
