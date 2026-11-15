"""
chess_ai/search/move_ordering.py
Chiến lược sắp xếp thứ tự duyệt nước đi: PV-move từ TT -> MVV-LVA -> Killer Moves -> History Heuristic.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from typing import Dict, List, Optional, Tuple
import chess
from chess_ai.config import load_yaml_config


_MVV_CFG = load_yaml_config("search_params.yaml")["evaluation"]["mvv_lva"]
_CHAR_TO_PIECE_TYPE: Dict[str, int] = {
    "P": chess.PAWN,
    "N": chess.KNIGHT,
    "B": chess.BISHOP,
    "R": chess.ROOK,
    "Q": chess.QUEEN,
    "K": chess.KING,
}
_PIECE_INDEX: Dict[int, int] = {
    _CHAR_TO_PIECE_TYPE[ch]: idx for idx, ch in enumerate(_MVV_CFG["order"])
}
_MVV_LVA_MATRIX: List[List[int]] = _MVV_CFG["matrix"]


def mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    """Tính điểm MVV-LVA cho một nước bắt quân dựa trên ma trận trong search_params.yaml."""
    if board.is_en_passant(move):
        victim_type = chess.PAWN
    else:
        victim = board.piece_at(move.to_square)
        if victim is None:
            return 0
        victim_type = victim.piece_type

    attacker = board.piece_at(move.from_square)
    attacker_type = attacker.piece_type if attacker is not None else chess.PAWN
    return _MVV_LVA_MATRIX[_PIECE_INDEX[victim_type]][_PIECE_INDEX[attacker_type]]


class MoveOrderer:
    """Quản lý Killer Moves (2 slots/ply) và History Heuristic cho một phiên tìm kiếm."""

    def __init__(self, max_ply: int = 64) -> None:
        self.max_ply = max_ply
        self.killers: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(max_ply)]
        self.history: Dict[Tuple[bool, int, int], int] = {}

    def clear(self) -> None:
        """Xóa bảng Killer và History."""
        self.killers = [[None, None] for _ in range(self.max_ply)]
        self.history.clear()

    def record_cutoff(self, board: chess.Board, move: chess.Move, ply: int, depth: int) -> None:
        """Ghi nhận nước đi yên tĩnh gây ra beta-cutoff vào Killer và History."""
        if board.is_capture(move) or move.promotion is not None:
            return
        if 0 <= ply < self.max_ply:
            slot0, slot1 = self.killers[ply]
            if move != slot0:
                self.killers[ply][1] = slot0
                self.killers[ply][0] = move
        key = (board.turn, move.from_square, move.to_square)
        self.history[key] = min(30_000, self.history.get(key, 0) + depth * depth)


def order_moves(
    board: chess.Board,
    moves: List[chess.Move],
    pv_move: Optional[chess.Move] = None,
    ply: int = 0,
    orderer: Optional[MoveOrderer] = None,
) -> List[chess.Move]:
    """Sắp xếp danh sách nước đi hợp lệ theo độ hứa hẹn chiến thuật giảm dần."""
    if len(moves) <= 1:
        return list(moves)

    killers = (
        orderer.killers[ply]
        if orderer is not None and 0 <= ply < orderer.max_ply
        else (None, None)
    )
    history = orderer.history if orderer is not None else None
    turn = board.turn

    def _score(move: chess.Move) -> int:
        if pv_move is not None and move == pv_move:
            return 1_000_000

        score = 0
        if board.is_capture(move):
            score += 100_000 + mvv_lva_score(board, move) * 100
        if move.promotion is not None:
            score += 80_000 + (900 if move.promotion == chess.QUEEN else 300)

        if score > 0:
            return score

        if move == killers[0]:
            return 50_000
        if move == killers[1]:
            return 40_000

        if history is not None:
            return history.get((turn, move.from_square, move.to_square), 0)
        return 0

    return sorted(moves, key=_score, reverse=True)
