"""
chess_ai/agents/level6_reinforcement.py
Cấp 6 [MỞ RỘNG]: Tác tử Học tăng cường (Afterstate Value Learning TD(lambda),
Tabular Q-Learning & Policy Gradient REINFORCE) trên không gian trạng thái cờ tàn
chuẩn FIDE (KRK, KQK, KPK) quy chiếu đối xứng D4 và chuẩn hóa màu.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from pathlib import Path
import pickle
import time
from typing import Dict, List, Optional, Tuple
import chess
import torch
import torch.nn as nn

from chess_ai.agents.base_agent import BaseAgent
from chess_ai.core.chess_env import ChessEnv


EndgameStateKey = Tuple[str, int, int, int]
CanonicalMoveKey = Tuple[int, int, Optional[int]]

REINFORCE_FEATURE_DIM = 18


def get_endgame_info(board: chess.Board) -> Optional[Tuple[str, chess.Color]]:
    """
    Nhận diện loại cờ tàn 3 quân được hỗ trợ ('KRK', 'KQK', 'KPK')
    và màu của bên cầm quân tấn công (Xe/Hậu/Tốt).
    """
    if board.occupied.bit_count() != 3:
        return None
    w_king = board.king(chess.WHITE)
    b_king = board.king(chess.BLACK)
    if w_king is None or b_king is None:
        return None

    for color in (chess.WHITE, chess.BLACK):
        if board.pieces_mask(chess.ROOK, color).bit_count() == 1:
            return "KRK", color
        if board.pieces_mask(chess.QUEEN, color).bit_count() == 1:
            return "KQK", color
        if board.pieces_mask(chess.PAWN, color).bit_count() == 1:
            return "KPK", color
    return None


def detect_endgame_type(board: chess.Board) -> Optional[str]:
    """Nhận diện loại cờ tàn 3 quân được hỗ trợ ('KRK', 'KQK', 'KPK')."""
    info = get_endgame_info(board)
    return info[0] if info is not None else None


def _normalize_board_color(board: chess.Board, attacker_color: chess.Color) -> chess.Board:
    """Chuẩn hóa bàn cờ sao cho bên cầm quân tấn công luôn là Trắng."""
    return board if attacker_color == chess.WHITE else board.mirror()


def _mirror_move(move: chess.Move) -> chess.Move:
    """Lật dọc nước đi (tương ứng phép mirror bàn cờ theo hàng 1 <-> 8)."""
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


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


def encode_afterstate(board: chess.Board) -> Optional[EndgameStateKey]:
    """
    Mã hóa trạng thái cờ tàn (KRK, KQK, KPK) theo quy chiếu đối xứng chuẩn hóa D4
    (hoặc lật ngang cho KPK) và chuẩn hóa màu (bên tấn công luôn là Trắng).
    Dùng cho Afterstate Value Learning V(s_after) với không gian chỉ ~35k trạng thái.
    """
    info = get_endgame_info(board)
    if info is None:
        return None
    eg_type, attacker_color = info
    norm_board = _normalize_board_color(board, attacker_color)

    wk = norm_board.king(chess.WHITE)
    bk = norm_board.king(chess.BLACK)
    if wk is None or bk is None:
        return None

    if eg_type in ("KRK", "KQK"):
        pt = chess.ROOK if eg_type == "KRK" else chess.QUEEN
        piece_sq = chess.lsb(norm_board.pieces_mask(pt, chess.WHITE))
        flip_v, flip_h, swap_diag = _d4_transform_flags(wk)
        c_wk = _apply_d4(wk, flip_v, flip_h, swap_diag)
        c_p = _apply_d4(piece_sq, flip_v, flip_h, swap_diag)
        c_bk = _apply_d4(bk, flip_v, flip_h, swap_diag)
        return (eg_type, c_wk, c_p, c_bk)

    # KPK: chỉ cho phép lật ngang (giữ nguyên hướng tiến của Tốt)
    pawn_sq = chess.lsb(norm_board.pieces_mask(chess.PAWN, chess.WHITE))
    flip_h = chess.square_file(pawn_sq) >= 4
    c_wk = wk ^ 7 if flip_h else wk
    c_p = pawn_sq ^ 7 if flip_h else pawn_sq
    c_bk = bk ^ 7 if flip_h else bk
    return (eg_type, c_wk, c_p, c_bk)


def encode_endgame_state_and_move(
    board: chess.Board,
    move: Optional[chess.Move] = None,
) -> Optional[Tuple[EndgameStateKey, Optional[CanonicalMoveKey]]]:
    """
    Mã hóa trạng thái cờ tàn (KRK, KQK, KPK) theo quy chiếu đối xứng chuẩn hóa và chuẩn hóa màu.
    Nếu truyền move, trả về cả cặp (state_key, canonical_move_key).
    """
    info = get_endgame_info(board)
    if info is None:
        return None
    eg_type, attacker_color = info
    norm_board = _normalize_board_color(board, attacker_color)
    norm_move = move if (move is None or attacker_color == chess.WHITE) else _mirror_move(move)

    wk = norm_board.king(chess.WHITE)
    bk = norm_board.king(chess.BLACK)
    assert wk is not None and bk is not None

    if eg_type in ("KRK", "KQK"):
        pt = chess.ROOK if eg_type == "KRK" else chess.QUEEN
        piece_sq = chess.lsb(norm_board.pieces_mask(pt, chess.WHITE))
        flip_v, flip_h, swap_diag = _d4_transform_flags(wk)
        c_wk = _apply_d4(wk, flip_v, flip_h, swap_diag)
        c_p = _apply_d4(piece_sq, flip_v, flip_h, swap_diag)
        c_bk = _apply_d4(bk, flip_v, flip_h, swap_diag)
        state_key: EndgameStateKey = (eg_type, c_wk, c_p, c_bk)
        if norm_move is None:
            return state_key, None
        c_from = _apply_d4(norm_move.from_square, flip_v, flip_h, swap_diag)
        c_to = _apply_d4(norm_move.to_square, flip_v, flip_h, swap_diag)
        return state_key, (c_from, c_to, norm_move.promotion)

    # KPK: chỉ cho phép lật ngang (giữ nguyên hướng tiến của Tốt)
    pawn_sq = chess.lsb(norm_board.pieces_mask(chess.PAWN, chess.WHITE))
    flip_h = chess.square_file(pawn_sq) >= 4
    c_wk = wk ^ 7 if flip_h else wk
    c_p = pawn_sq ^ 7 if flip_h else pawn_sq
    c_bk = bk ^ 7 if flip_h else bk
    state_key = (eg_type, c_wk, c_p, c_bk)
    if norm_move is None:
        return state_key, None
    c_from = norm_move.from_square ^ 7 if flip_h else norm_move.from_square
    c_to = norm_move.to_square ^ 7 if flip_h else norm_move.to_square
    return state_key, (c_from, c_to, norm_move.promotion)


def is_piece_hanging_after_move(board_after: chess.Board, attacker_color: chess.Color) -> bool:
    """Kiểm tra quân tấn công (Xe/Hậu/Tốt) có bị Vua đối phương ăn ngay không được bảo vệ hay không."""
    defender = not attacker_color
    ak = board_after.king(attacker_color)
    dk = board_after.king(defender)
    if ak is None or dk is None:
        return False
    for pt in (chess.ROOK, chess.QUEEN, chess.PAWN):
        mask = board_after.pieces_mask(pt, attacker_color)
        if mask:
            sq = chess.lsb(mask)
            return chess.square_distance(sq, dk) == 1 and chess.square_distance(sq, ak) > 1
    return False


def _krk_box_area(wr_f: int, wr_r: int, bk_f: int, bk_r: int) -> float:
    """Tính diện tích vùng hình chữ nhật (box) mà Xe đang giam Vua Đen (chuẩn hóa về [0, 1])."""
    if bk_f < wr_f:
        width = wr_f
    elif bk_f > wr_f:
        width = 7 - wr_f
    else:
        width = 8

    if bk_r < wr_r:
        height = wr_r
    elif bk_r > wr_r:
        height = 7 - wr_r
    else:
        height = 8

    return (width * height) / 64.0


def krk_action_features(board: chess.Board, move: chess.Move) -> List[float]:
    """
    Trích xuất 18 đặc trưng hình học & chiến thuật chuẩn hóa cho nước đi trong thế cờ KRK
    (tự động chuẩn hóa màu nếu Đen là bên cầm Xe).
    """
    info = get_endgame_info(board)
    if info is not None and info[1] == chess.BLACK:
        board = board.mirror()
        move = _mirror_move(move)

    board.push(move)
    in_check_bool = board.is_check()
    bk_legal_count = sum(1 for _ in board.generate_legal_moves())
    is_mate = 1.0 if (bk_legal_count == 0 and in_check_bool) else 0.0
    is_stale = 1.0 if (bk_legal_count == 0 and not in_check_bool) else 0.0
    in_check = 1.0 if in_check_bool else 0.0
    rook_hanging = 1.0 if is_piece_hanging_after_move(board, chess.WHITE) else 0.0
    bk_mobility = bk_legal_count / 8.0

    wk = board.king(chess.WHITE) or 0
    bk = board.king(chess.BLACK) or 0
    wr_mask = board.pieces_mask(chess.ROOK, chess.WHITE)
    wr = chess.lsb(wr_mask) if wr_mask else 0
    board.pop()

    wk_f, wk_r = chess.square_file(wk), chess.square_rank(wk)
    bk_f, bk_r = chess.square_file(bk), chess.square_rank(bk)
    wr_f, wr_r = chess.square_file(wr), chess.square_rank(wr)

    king_cheb = max(abs(wk_f - bk_f), abs(wk_r - bk_r)) / 7.0
    king_manh = (abs(wk_f - bk_f) + abs(wk_r - bk_r)) / 14.0
    bk_edge = min(bk_f, 7 - bk_f, bk_r, 7 - bk_r) / 3.5
    rook_dist = (abs(wr_f - bk_f) + abs(wr_r - bk_r)) / 14.0
    rook_same_line = 1.0 if (wr_f == bk_f or wr_r == bk_r) else 0.0
    box_area = _krk_box_area(wr_f, wr_r, bk_f, bk_r)
    opposition = 1.0 if (
        (abs(wk_f - bk_f) == 2 and wk_r == bk_r)
        or (abs(wk_r - bk_r) == 2 and wk_f == bk_f)
    ) else 0.0

    return [
        wk_f / 7.0,
        wk_r / 7.0,
        wr_f / 7.0,
        wr_r / 7.0,
        bk_f / 7.0,
        bk_r / 7.0,
        king_cheb,
        king_manh,
        bk_edge,
        rook_dist,
        in_check,
        rook_same_line,
        is_mate,
        is_stale,
        rook_hanging,
        bk_mobility,
        box_area,
        opposition,
    ]


class ReinforcePolicyNet(nn.Module):
    """Mạng chính sách đánh giá cặp (trạng thái KRK, nước đi ứng viên) cho REINFORCE."""

    def __init__(self, in_dim: int = REINFORCE_FEATURE_DIM, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class Level6ReinforcementAgent(BaseAgent):
    """
    Tác tử Cấp 6: Reinforcement Learning trên cờ tàn (KRK, KQK, KPK).
    Hỗ trợ đồng thời:
      - Afterstate Value Tables V(s_after) (ưu tiên mặc định, không gian nhỏ & hội tụ nhanh)
      - Legacy Tabular Q-Tables Q(s, a) (tương thích ngược với checkpoint cũ)
      - Policy Gradient REINFORCE Network cho KRK
    """

    def __init__(
        self,
        q_table_path: Optional[str] = "checkpoints/l6_krk_qtable.pkl",
        policy_mode: str = "auto",
    ) -> None:
        super().__init__(name="Level6_RL_KRK")
        self.q_table_path = q_table_path
        self.policy_mode = policy_mode
        self.v_tables: Dict[str, Dict[EndgameStateKey, float]] = {}
        self.q_tables: Dict[str, Dict[Tuple[EndgameStateKey, CanonicalMoveKey], float]] = {}
        self.reinforce_policy: Optional[ReinforcePolicyNet] = None
        if q_table_path and Path(q_table_path).exists():
            self.load_q_tables(q_table_path)

    def load_q_tables(self, filepath: str) -> None:
        """Nạp bảng Afterstate V-table, Q-table (nếu có) và mạng REINFORCE từ tệp pickle."""
        with open(filepath, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            return

        if "v_tables" in payload and isinstance(payload["v_tables"], dict):
            self.v_tables = payload["v_tables"]
        if "q_tables" in payload and isinstance(payload["q_tables"], dict):
            self.q_tables = payload["q_tables"]
        elif "v_tables" not in payload:
            # Tương thích ngược với tệp chỉ chứa 1 dict Q-table KRK thuần
            self.q_tables = {"KRK": payload}

        reinf_state = payload.get("reinforce_state_dict")
        if isinstance(reinf_state, dict):
            in_dim = int(payload.get("reinforce_in_dim", REINFORCE_FEATURE_DIM))
            hidden = int(payload.get("reinforce_hidden", 64))
            net = ReinforcePolicyNet(in_dim=in_dim, hidden=hidden)
            net.load_state_dict(reinf_state)
            net.eval()
            self.reinforce_policy = net

    @torch.no_grad()
    def _select_with_reinforce(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
    ) -> Tuple[chess.Move, float]:
        """Chọn nước đi bằng mạng chính sách REINFORCE (KRK)."""
        assert self.reinforce_policy is not None
        feats = torch.tensor(
            [krk_action_features(board, mv) for mv in legal_moves],
            dtype=torch.float32,
        )
        logits = self.reinforce_policy(feats)
        best_idx = int(torch.argmax(logits).item())
        return legal_moves[best_idx], float(logits[best_idx].item())

    def _select_with_afterstate(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        eg_type: str,
        attacker_color: chess.Color,
    ) -> Tuple[Optional[chess.Move], float, bool]:
        """Chọn nước đi từ bảng Afterstate Value V(s_after) với kiểm tra terminal và mất quân."""
        table = self.v_tables.get(eg_type)
        if not table:
            return None, 0.0, False

        kqk_table = self.v_tables.get("KQK")
        best_move = legal_moves[0]
        best_score = -float("inf")
        seen_any = False

        for move in legal_moves:
            board.push(move)
            has_reply = any(board.generate_legal_moves())
            if not has_reply:
                is_mate = board.is_check()
                board.pop()
                if is_mate:
                    return move, 1.0, True
                score = -1.0
                seen_any = True
            elif board.is_insufficient_material():
                score = -1.0
                seen_any = True
                board.pop()
            elif is_piece_hanging_after_move(board, attacker_color):
                score = -1.2
                seen_any = True
                board.pop()
            else:
                after_key = encode_afterstate(board)
                val: Optional[float] = None
                if after_key is not None:
                    if after_key[0] == eg_type:
                        val = table.get(after_key)
                    elif eg_type == "KPK" and after_key[0] == "KQK":
                        # Phong Hậu an toàn: ưu tiên rất cao và nối sang bảng KQK
                        kqk_val = kqk_table.get(after_key, 0.0) if kqk_table else 0.0
                        val = max(0.85, kqk_val)
                    elif eg_type == "KPK" and after_key[0] == "KRK":
                        krk_table = self.v_tables.get("KRK")
                        krk_val = krk_table.get(after_key, 0.0) if krk_table else 0.0
                        val = max(0.70, krk_val)

                if val is not None:
                    seen_any = True
                    score = val
                else:
                    # Coi trạng thái chưa thăm có giá trị 0.0 thay vì bỏ qua để tránh chọn nhầm nước Q < 0
                    score = 0.0
                if board.is_repetition(2):
                    score -= 0.5
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move, float(best_score), seen_any

    def _select_with_legacy_q(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        eg_type: str,
    ) -> Tuple[Optional[chess.Move], float, bool]:
        """Chọn nước đi từ bảng Q(s, a) truyền thống (đã sửa lỗi xử lý nước chưa thăm)."""
        table = self.q_tables.get(eg_type)
        if not table:
            return None, 0.0, False

        best_move = legal_moves[0]
        best_score = -float("inf")
        seen_any = False

        for move in legal_moves:
            encoded = encode_endgame_state_and_move(board, move)
            if encoded is None:
                continue
            s_key, m_key = encoded
            raw_q = table.get((s_key, m_key))
            if raw_q is not None:
                seen_any = True
                q_val = raw_q
            else:
                q_val = 0.0
            if q_val > best_score:
                best_score = q_val
                best_move = move

        return best_move, float(best_score), seen_any

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Chọn nước đi từ chính sách RL đã học (hỗ trợ cả Trắng/Đen, Afterstate, Q-table, REINFORCE)."""
        t0 = time.perf_counter()
        board = env.board
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError(f"Không có nước đi hợp lệ tại thế cờ: {board.fen()}")

        # Ưu tiên tuyệt đối chiếu hết trong 1 nước
        for move in legal_moves:
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                self.last_search_stats = {
                    "nodes": float(len(legal_moves)),
                    "simulations": 0.0,
                    "depth": 1.0,
                    "eval": 1.0,
                    "time_sec": time.perf_counter() - t0,
                }
                return move

        info = get_endgame_info(board)
        best_move: Optional[chess.Move] = None
        best_eval = 0.0
        decided = False

        if info is not None:
            eg_type, attacker_color = info
            if board.turn == attacker_color:
                if self.policy_mode == "reinforce" and eg_type == "KRK" and self.reinforce_policy is not None:
                    best_move, best_eval = self._select_with_reinforce(board, legal_moves)
                    decided = True
                elif self.policy_mode in ("auto", "afterstate") and eg_type in self.v_tables:
                    mv, sc, seen = self._select_with_afterstate(board, legal_moves, eg_type, attacker_color)
                    if seen and mv is not None:
                        best_move, best_eval, decided = mv, sc, True

                if not decided and self.policy_mode in ("auto", "q_table") and eg_type in self.q_tables:
                    mv, sc, seen = self._select_with_legacy_q(board, legal_moves, eg_type)
                    if seen and mv is not None:
                        best_move, best_eval, decided = mv, sc, True

                if not decided and self.policy_mode == "auto" and eg_type == "KRK" and self.reinforce_policy is not None:
                    best_move, best_eval = self._select_with_reinforce(board, legal_moves)
                    decided = True

        if not decided or best_move is None:
            best_move, best_eval = self._heuristic_endgame_fallback(board, legal_moves)

        self.last_search_stats = {
            "nodes": float(len(legal_moves)),
            "simulations": 0.0,
            "depth": 1.0,
            "eval": float(best_eval),
            "time_sec": time.perf_counter() - t0,
        }
        return best_move

    @staticmethod
    def _heuristic_endgame_fallback(
        board: chess.Board,
        legal_moves: List[chess.Move],
    ) -> Tuple[chess.Move, float]:
        """
        Fallback hình học an toàn cho cờ tàn khi trạng thái chưa thấy trong bảng:
        - Nếu là bên tấn công: tránh mất quân, tránh hòa pat/lặp nước, ép Vua đối thủ vào cạnh & áp sát Vua.
        - Nếu là bên phòng thủ: ăn quân không được bảo vệ nếu có, ngược lại kéo Vua về trung tâm.
        """
        info = get_endgame_info(board)
        best_move = legal_moves[0]
        best_score = -float("inf")

        if info is None:
            for move in legal_moves:
                board.push(move)
                score = 10.0 if board.is_checkmate() else (0.5 if board.is_check() else 0.0)
                if board.is_stalemate():
                    score = -5.0
                board.pop()
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_move, best_score

        eg_type, attacker_color = info
        is_attacker_turn = board.turn == attacker_color

        for move in legal_moves:
            is_capture = board.is_capture(move)
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move, 1.0

            if is_attacker_turn:
                if board.is_stalemate() or board.is_insufficient_material():
                    score = -1.0
                elif is_piece_hanging_after_move(board, attacker_color):
                    score = -0.95
                else:
                    norm = _normalize_board_color(board, attacker_color)
                    wk = norm.king(chess.WHITE) or 0
                    bk = norm.king(chess.BLACK) or 0
                    bk_f, bk_r = chess.square_file(bk), chess.square_rank(bk)
                    bk_edge = min(bk_f, 7 - bk_f, bk_r, 7 - bk_r)
                    king_dist = chess.square_distance(wk, bk)

                    if eg_type == "KPK":
                        if norm.pieces_mask(chess.QUEEN, chess.WHITE):
                            score = 0.85
                        else:
                            p_mask = norm.pieces_mask(chess.PAWN, chess.WHITE)
                            wp = chess.lsb(p_mask) if p_mask else 0
                            score = 0.08 * chess.square_rank(wp) - 0.04 * chess.square_distance(wk, wp)
                    else:
                        score = -0.08 * bk_edge - 0.05 * king_dist
                        if norm.is_check():
                            score += 0.02
                    if board.is_repetition(2):
                        score -= 0.4
            else:
                # Bên phòng thủ (chỉ có 1 Vua)
                if is_capture:
                    score = 1.0
                elif board.is_stalemate():
                    score = 0.9
                else:
                    dk = board.king(not attacker_color) or 0
                    dk_f, dk_r = chess.square_file(dk), chess.square_rank(dk)
                    center_dist = abs(dk_f - 3.5) + abs(dk_r - 3.5)
                    score = -0.1 * center_dist

            board.pop()
            if score > best_score:
                best_score = score
                best_move = move

        return best_move, float(best_score)
