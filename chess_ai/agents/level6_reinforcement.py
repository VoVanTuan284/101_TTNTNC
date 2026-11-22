"""
chess_ai/agents/level6_reinforcement.py
Cấp 6 [MỞ RỘNG]: Tác tử Học tăng cường (Tabular Q-Learning & Policy Gradient REINFORCE)
trên không gian trạng thái cờ tàn chuẩn FIDE (KRK, KQK, KPK) quy chiếu đối xứng.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from pathlib import Path
import pickle
import time
from typing import Dict, Optional, Tuple
import chess

from chess_ai.agents.base_agent import BaseAgent
from chess_ai.core.chess_env import ChessEnv


EndgameStateKey = Tuple[str, int, int, int]
CanonicalMoveKey = Tuple[int, int, Optional[int]]


def detect_endgame_type(board: chess.Board) -> Optional[str]:
    """Nhận diện loại cờ tàn 3 quân được hỗ trợ ('KRK', 'KQK', 'KPK')."""
    if board.occupied.bit_count() != 3:
        return None
    w_king = board.king(chess.WHITE)
    b_king = board.king(chess.BLACK)
    if w_king is None or b_king is None:
        return None

    if board.pieces_mask(chess.ROOK, chess.WHITE).bit_count() == 1:
        return "KRK"
    if board.pieces_mask(chess.QUEEN, chess.WHITE).bit_count() == 1:
        return "KQK"
    if board.pieces_mask(chess.PAWN, chess.WHITE).bit_count() == 1:
        return "KPK"
    return None


def _d4_transform_flags(wk_sq: int) -> Tuple[bool, bool, bool]:
    """Xác định bộ 3 phép biến đổi đối xứng D4 đưa Vua Trắng về tam giác chuẩn (0<=f<=r<=3)."""
    r = chess.square_rank(wk_sq)
    f = chess.square_file(wk_sq)
    flip_v = r >= 4
    if flip_v:
        r = 7 - r
    flip_h = f >= 4
    if flip_h:
        f = 7 - f
    swap_diag = f > r
    return flip_v, flip_h, swap_diag


def _apply_d4(sq: int, flip_v: bool, flip_h: bool, swap_diag: bool) -> int:
    if flip_v:
        sq ^= 56
    if flip_h:
        sq ^= 7
    if swap_diag:
        r = chess.square_rank(sq)
        f = chess.square_file(sq)
        sq = chess.square(r, f)
    return sq


def encode_endgame_state_and_move(
    board: chess.Board,
    move: Optional[chess.Move] = None,
) -> Optional[Tuple[EndgameStateKey, Optional[CanonicalMoveKey]]]:
    """
    Mã hóa trạng thái cờ tàn (KRK, KQK, KPK) theo quy chiếu đối xứng chuẩn hóa.
    Nếu truyền move, trả về cả cặp (state_key, canonical_move_key).
    """
    eg_type = detect_endgame_type(board)
    if eg_type is None:
        return None

    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    assert wk is not None and bk is not None

    if eg_type in ("KRK", "KQK"):
        pt = chess.ROOK if eg_type == "KRK" else chess.QUEEN
        piece_sq = chess.lsb(board.pieces_mask(pt, chess.WHITE))
        flip_v, flip_h, swap_diag = _d4_transform_flags(wk)
        c_wk = _apply_d4(wk, flip_v, flip_h, swap_diag)
        c_p = _apply_d4(piece_sq, flip_v, flip_h, swap_diag)
        c_bk = _apply_d4(bk, flip_v, flip_h, swap_diag)
        state_key: EndgameStateKey = (eg_type, c_wk, c_p, c_bk)
        if move is None:
            return state_key, None
        c_from = _apply_d4(move.from_square, flip_v, flip_h, swap_diag)
        c_to = _apply_d4(move.to_square, flip_v, flip_h, swap_diag)
        return state_key, (c_from, c_to, move.promotion)

    # KPK: chỉ cho phép lật ngang (giữ nguyên hướng tiến của Tốt)
    pawn_sq = chess.lsb(board.pieces_mask(chess.PAWN, chess.WHITE))
    flip_h = chess.square_file(pawn_sq) >= 4
    c_wk = wk ^ 7 if flip_h else wk
    c_p = pawn_sq ^ 7 if flip_h else pawn_sq
    c_bk = bk ^ 7 if flip_h else bk
    state_key = (eg_type, c_wk, c_p, c_bk)
    if move is None:
        return state_key, None
    c_from = move.from_square ^ 7 if flip_h else move.from_square
    c_to = move.to_square ^ 7 if flip_h else move.to_square
    return state_key, (c_from, c_to, move.promotion)


class Level6ReinforcementAgent(BaseAgent):
    """Tác tử Cấp 6: Reinforcement Learning trên cờ tàn (KRK, KQK, KPK)."""

    def __init__(self, q_table_path: Optional[str] = "checkpoints/l6_krk_qtable.pkl") -> None:
        super().__init__(name="Level6_RL_KRK")
        self.q_table_path = q_table_path
        self.q_tables: Dict[str, Dict[Tuple[EndgameStateKey, CanonicalMoveKey], float]] = {}
        if q_table_path and Path(q_table_path).exists():
            self.load_q_tables(q_table_path)

    def load_q_tables(self, filepath: str) -> None:
        """Nạp bảng Q-table từ tệp pickle."""
        with open(filepath, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "q_tables" in payload:
            self.q_tables = payload["q_tables"]
        elif isinstance(payload, dict):
            self.q_tables = {"KRK": payload}

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Chọn nước đi từ chính sách RL đã học (ưu tiên Q-table; fallback an toàn nếu chưa thấy)."""
        t0 = time.perf_counter()
        board = env.board
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError(f"Không có nước đi hợp lệ tại thế cờ: {board.fen()}")

        eg_type = detect_endgame_type(board)
        best_move = legal_moves[0]
        best_q = 0.0

        if eg_type is not None and board.turn == chess.WHITE and eg_type in self.q_tables:
            table = self.q_tables[eg_type]
            best_score = -float("inf")
            seen_any = False

            for move in legal_moves:
                encoded = encode_endgame_state_and_move(board, move)
                if encoded is None:
                    continue
                s_key, m_key = encoded
                q_val = table.get((s_key, m_key))
                if q_val is not None:
                    seen_any = True
                    if q_val > best_score:
                        best_score = q_val
                        best_move = move
                        best_q = q_val

            if not seen_any:
                best_move = self._heuristic_endgame_fallback(board, legal_moves)
        else:
            best_move = self._heuristic_endgame_fallback(board, legal_moves)

        self.last_search_stats = {
            "nodes": float(len(legal_moves)),
            "simulations": 0.0,
            "depth": 1.0,
            "eval": float(best_q),
            "time_sec": time.perf_counter() - t0,
        }
        return best_move

    @staticmethod
    def _heuristic_endgame_fallback(board: chess.Board, legal_moves: list[chess.Move]) -> chess.Move:
        """Fallback nhẹ khi gặp thế cờ chưa có trong bảng Q hoặc không phải lượt Trắng."""
        for move in legal_moves:
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return move
        return legal_moves[0]
