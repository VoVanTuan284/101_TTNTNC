"""
chess_ai/search/alphabeta.py
Thuật toán Negamax kết hợp cắt tỉa Alpha-Beta, Iterative Deepening (ID),
bảng chuyển vị Zobrist (TT) và kiểm soát thời gian hợp tác (Cooperative Time Management).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

import time
from typing import Dict, Optional, Set, Tuple
import chess
from chess_ai.core.evaluation import MATE_VALUE, evaluate_board
from chess_ai.core.zobrist import compute_zobrist_hash
from chess_ai.search.move_ordering import MoveOrderer, order_moves
from chess_ai.search.quiescence import quiescence_search
from chess_ai.search.transposition import (
    EXACT,
    LOWERBOUND,
    UPPERBOUND,
    TranspositionTable,
    TTEntry,
)


_INF = float("inf")
_MATE_THRESHOLD = MATE_VALUE - 1000.0


class _SearchContext:
    """Theo dõi ngân sách nút duyệt và thời gian hợp tác (kiểm tra mỗi 256 nút)."""

    def __init__(
        self,
        time_limit: Optional[float],
        max_nodes: Optional[int],
    ) -> None:
        self.start_time = time.perf_counter()
        self.deadline = self.start_time + time_limit if time_limit is not None else None
        self.max_nodes = max_nodes
        self.nodes = 0
        self.stopped = False

    def tick(self) -> None:
        self.nodes += 1
        if self.max_nodes is not None and self.nodes >= self.max_nodes:
            self.stopped = True
            return
        if self.deadline is not None and (self.nodes & 255) == 0:
            if time.perf_counter() >= self.deadline:
                self.stopped = True


def _to_tt_score(value: float, ply: int) -> float:
    if value >= _MATE_THRESHOLD:
        return value + ply
    if value <= -_MATE_THRESHOLD:
        return value - ply
    return value


def _from_tt_score(value: float, ply: int) -> float:
    if value >= _MATE_THRESHOLD:
        return value - ply
    if value <= -_MATE_THRESHOLD:
        return value + ply
    return value


def _build_history_zobrist_counts(board: chess.Board) -> Dict[int, int]:
    """Đếm số lần xuất hiện khóa Zobrist từ lịch sử move_stack (trong phạm vi halfmove_clock)."""
    counts: Dict[int, int] = {}
    root_key = compute_zobrist_hash(board)
    counts[root_key] = 1

    if not board.move_stack or board.halfmove_clock == 0:
        return counts

    tmp = board.copy(stack=True)
    steps_back = min(len(tmp.move_stack), board.halfmove_clock)
    for _ in range(steps_back):
        tmp.pop()
        k = compute_zobrist_hash(tmp)
        counts[k] = counts.get(k, 0) + 1
    return counts


def _negamax(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    ply: int,
    ctx: _SearchContext,
    tt: Optional[TranspositionTable],
    orderer: Optional[MoveOrderer],
    path_hashes: Set[int],
    history_counts: Dict[int, int],
    use_alpha_beta: bool,
    use_move_ordering: bool,
    use_quiescence: bool,
    use_pvs: bool = False,
) -> Tuple[float, Optional[chess.Move]]:
    ctx.tick()
    if ctx.stopped:
        return 0.0, None

    if board.is_checkmate():
        return -MATE_VALUE + ply, None
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_seventyfive_moves()
        or board.halfmove_clock >= 100
    ):
        return 0.0, None

    zobrist_key = compute_zobrist_hash(board)
    if ply > 0 and (zobrist_key in path_hashes or history_counts.get(zobrist_key, 0) >= 2):
        return 0.0, None

    pv_move: Optional[chess.Move] = None
    orig_alpha = alpha

    if tt is not None:
        entry = tt.lookup(zobrist_key)
        if entry is not None:
            pv_move = entry.best_move
            if ply > 0 and entry.depth >= depth:
                tt_val = _from_tt_score(entry.value, ply)
                if entry.flag == EXACT:
                    return tt_val, pv_move
                if use_alpha_beta:
                    if entry.flag == LOWERBOUND:
                        alpha = max(alpha, tt_val)
                    elif entry.flag == UPPERBOUND:
                        beta = min(beta, tt_val)
                    if alpha >= beta:
                        return tt_val, pv_move

    if depth <= 0:
        if use_quiescence:
            q_val = quiescence_search(board, alpha, beta, search_ctx=ctx)
            return q_val, None
        return evaluate_board(board), None

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return 0.0, None

    if use_move_ordering:
        legal_moves = order_moves(
            board,
            legal_moves,
            pv_move=pv_move,
            ply=ply,
            orderer=orderer,
        )

    best_score = -_INF
    best_move: Optional[chess.Move] = legal_moves[0]
    added_hash = zobrist_key not in path_hashes
    if added_hash:
        path_hashes.add(zobrist_key)
    if ply > 0:
        history_counts[zobrist_key] = history_counts.get(zobrist_key, 0) + 1

    try:
        for move_idx, move in enumerate(legal_moves):
            board.push(move)
            if not use_alpha_beta:
                child_score, _ = _negamax(
                    board,
                    depth - 1,
                    -_INF,
                    _INF,
                    ply + 1,
                    ctx,
                    tt,
                    orderer,
                    path_hashes,
                    history_counts,
                    use_alpha_beta,
                    use_move_ordering,
                    use_quiescence,
                    use_pvs,
                )
            elif not use_pvs or move_idx == 0 or alpha == -_INF:
                child_score, _ = _negamax(
                    board,
                    depth - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                    ctx,
                    tt,
                    orderer,
                    path_hashes,
                    history_counts,
                    use_alpha_beta,
                    use_move_ordering,
                    use_quiescence,
                    use_pvs,
                )
            else:
                # Principal Variation Search (null-window probe)
                child_score, _ = _negamax(
                    board,
                    depth - 1,
                    -alpha - 1.0,
                    -alpha,
                    ply + 1,
                    ctx,
                    tt,
                    orderer,
                    path_hashes,
                    history_counts,
                    use_alpha_beta,
                    use_move_ordering,
                    use_quiescence,
                    use_pvs,
                )
                if not ctx.stopped and alpha < -child_score < beta:
                    child_score, _ = _negamax(
                        board,
                        depth - 1,
                        -beta,
                        -alpha,
                        ply + 1,
                        ctx,
                        tt,
                        orderer,
                        path_hashes,
                        history_counts,
                        use_alpha_beta,
                        use_move_ordering,
                        use_quiescence,
                        use_pvs,
                    )
            board.pop()

            if ctx.stopped:
                return best_score, best_move

            score = -child_score
            if score > best_score:
                best_score = score
                best_move = move

            if use_alpha_beta:
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    if use_move_ordering and orderer is not None:
                        orderer.record_cutoff(board, move, ply, depth)
                    break
    finally:
        if added_hash:
            path_hashes.discard(zobrist_key)
        if ply > 0:
            new_cnt = history_counts.get(zobrist_key, 1) - 1
            if new_cnt <= 0:
                history_counts.pop(zobrist_key, None)
            else:
                history_counts[zobrist_key] = new_cnt

    if tt is not None and not ctx.stopped:
        if best_score <= orig_alpha:
            flag = UPPERBOUND
        elif use_alpha_beta and best_score >= beta:
            flag = LOWERBOUND
        else:
            flag = EXACT
        tt.store(
            TTEntry(
                zobrist_key=zobrist_key,
                depth=depth,
                value=_to_tt_score(best_score, ply),
                flag=flag,
                best_move=best_move,
            )
        )

    return best_score, best_move


def search_alphabeta(
    board: chess.Board,
    max_depth: int = 4,
    time_limit: Optional[float] = None,
    max_nodes: Optional[int] = None,
    use_tt: bool = True,
    use_move_ordering: bool = True,
    use_quiescence: bool = True,
    use_alpha_beta: bool = True,
    tt: Optional[TranspositionTable] = None,
    use_pvs: bool = False,
    use_aspiration: bool = False,
    aspiration_window: float = 50.0,
) -> Tuple[chess.Move, Dict[str, float]]:
    """
    Thực hiện tìm kiếm Iterative Deepening Alpha-Beta và trả về (best_move, stats).
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError(f"Không có nước đi hợp lệ tại thế cờ: {board.fen()}")

    active_tt: Optional[TranspositionTable] = None
    if use_tt:
        active_tt = tt if tt is not None else TranspositionTable(max_size_mb=16)
    initial_hits = active_tt.hits if active_tt is not None else 0

    orderer = MoveOrderer() if use_move_ordering else None
    ctx = _SearchContext(time_limit=time_limit, max_nodes=max_nodes)
    path_hashes: Set[int] = set()
    history_counts = _build_history_zobrist_counts(board)

    best_move = legal_moves[0]
    best_value = 0.0
    completed_depth = 0

    work_board = board.copy(stack=False)

    for depth in range(1, max(1, max_depth) + 1):
        if use_alpha_beta and use_aspiration and depth >= 3 and abs(best_value) < _MATE_THRESHOLD:
            delta = max(10.0, float(aspiration_window))
            asp_alpha = best_value - delta
            asp_beta = best_value + delta
            while True:
                val, move = _negamax(
                    work_board,
                    depth=depth,
                    alpha=asp_alpha,
                    beta=asp_beta,
                    ply=0,
                    ctx=ctx,
                    tt=active_tt,
                    orderer=orderer,
                    path_hashes=path_hashes,
                    history_counts=history_counts,
                    use_alpha_beta=use_alpha_beta,
                    use_move_ordering=use_move_ordering,
                    use_quiescence=use_quiescence,
                    use_pvs=use_pvs,
                )
                if ctx.stopped:
                    break
                if val <= asp_alpha:
                    asp_alpha = -_INF if delta >= 400.0 else asp_alpha - delta
                    delta *= 2.0
                elif val >= asp_beta:
                    asp_beta = _INF if delta >= 400.0 else asp_beta + delta
                    delta *= 2.0
                else:
                    break
        else:
            val, move = _negamax(
                work_board,
                depth=depth,
                alpha=-_INF,
                beta=_INF,
                ply=0,
                ctx=ctx,
                tt=active_tt,
                orderer=orderer,
                path_hashes=path_hashes,
                history_counts=history_counts,
                use_alpha_beta=use_alpha_beta,
                use_move_ordering=use_move_ordering,
                use_quiescence=use_quiescence,
                use_pvs=use_pvs,
            )
        if ctx.stopped and completed_depth >= 1:
            break
        if move is not None:
            best_move = move
            best_value = val
            completed_depth = depth
        if ctx.stopped or abs(best_value) >= _MATE_THRESHOLD:
            break

    elapsed = time.perf_counter() - ctx.start_time
    tt_hits = (active_tt.hits - initial_hits) if active_tt is not None else 0
    stats: Dict[str, float] = {
        "nodes": float(ctx.nodes),
        "simulations": 0.0,
        "depth": float(completed_depth),
        "eval": float(best_value),
        "tt_hits": float(tt_hits),
        "time_sec": float(elapsed),
    }
    return best_move, stats
