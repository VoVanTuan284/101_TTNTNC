"""
chess_ai/search/mcts_uct.py
Động cơ Vanilla MCTS sử dụng công thức UCB1 (UCT) cho Tác tử Cấp 4,
tích hợp mô phỏng cắt ngưỡng (max_rollout_depth = 60) và định giá tĩnh tanh.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

import math
import random
import time
from typing import Dict, Optional, Tuple
import chess
from chess_ai.core.evaluation import evaluate_board_tanh, see_capture
from chess_ai.core.zobrist import compute_zobrist_hash
from chess_ai.search.mcts_node import MCTSNode


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


def _is_fast_terminal(board: chess.Board) -> Optional[float]:
    """
    Kiểm tra kết thúc ván cờ nhanh tại nút cây/rollout.
    Trả về giá trị từ góc nhìn bên tới lượt đi (-1.0 nếu bị chiếu hết, 0.0 nếu hòa, None nếu chưa hết).
    """
    if board.is_checkmate():
        return -1.0
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_seventyfive_moves()
        or board.halfmove_clock >= 100
    ):
        return 0.0
    return None


def _select_uct_child(
    node: MCTSNode,
    c_uct: float,
    rng: random.Random,
) -> Tuple[chess.Move, MCTSNode]:
    """Chọn nút con theo công thức UCB1 (UCT), ưu tiên nút chưa thăm."""
    unvisited = [item for item in node.children.items() if item[1].visit_count == 0]
    if unvisited:
        return rng.choice(unvisited)

    log_parent = math.log(max(1, node.visit_count))
    best_score = -float("inf")
    best_item = None

    for move, child in node.children.items():
        uct = child.q_value + c_uct * math.sqrt(log_parent / child.visit_count)
        if uct > best_score:
            best_score = uct
            best_item = (move, child)

    assert best_item is not None
    return best_item


def _rollout(
    board: chess.Board,
    max_rollout_depth: int,
    rng: random.Random,
) -> float:
    """
    Mô phỏng nhanh (Cut-off Rollout) tối đa max_rollout_depth plies,
    ưu tiên nhẹ nước bắt quân và cắt ngưỡng bằng evaluate_board_tanh().
    Trả về giá trị từ góc nhìn bên tới lượt đi tại trạng thái xuất phát của rollout.
    """
    sim_board = board.copy(stack=False)
    plies = 0

    while plies < max_rollout_depth:
        term_val = _is_fast_terminal(sim_board)
        if term_val is not None:
            return term_val if (plies % 2 == 0) else -term_val

        legal_moves = list(sim_board.legal_moves)
        if not legal_moves:
            return 0.0

        # Ưu tiên nhẹ các nước bắt quân không lỗ (SEE >= 0) hoặc phong Hậu (70%) để giảm nhiễu rollout
        if rng.random() < 0.7:
            good_tactical = [
                m
                for m in legal_moves
                if m.promotion == chess.QUEEN or (sim_board.is_capture(m) and see_capture(sim_board, m) >= 0)
            ]
            move = rng.choice(good_tactical) if good_tactical else rng.choice(legal_moves)
        else:
            move = rng.choice(legal_moves)

        sim_board.push(move)
        plies += 1

    term_val = _is_fast_terminal(sim_board)
    leaf_val = term_val if term_val is not None else evaluate_board_tanh(sim_board, scale=600.0)
    return leaf_val if (plies % 2 == 0) else -leaf_val


def _backpropagate(node: MCTSNode, leaf_value_from_node_turn: float) -> None:
    """
    Lan truyền ngược giá trị lên gốc theo quy ước Negamax:
    value_sum tại mỗi nút lưu theo góc nhìn của nút cha (bên đã chọn nước đi tới nút đó).
    """
    curr: Optional[MCTSNode] = node
    value_for_parent = -leaf_value_from_node_turn
    while curr is not None:
        curr.visit_count += 1
        curr.value_sum += value_for_parent
        value_for_parent = -value_for_parent
        curr = curr.parent


def run_uct_search(
    board: chess.Board,
    simulations: int = 400,
    time_limit: Optional[float] = None,
    c_uct: float = 1.414,
    max_rollout_depth: int = 60,
    root: Optional[MCTSNode] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[chess.Move, Dict[str, float], MCTSNode]:
    """Thực thi tìm kiếm Vanilla MCTS (UCT) và trả về (best_move, stats, root)."""
    t0 = time.perf_counter()
    deadline = (t0 + time_limit) if time_limit is not None else None
    local_rng = rng if rng is not None else random.Random()

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError(f"Không có nước đi hợp lệ tại thế cờ: {board.fen()}")

    root_zobrist = compute_zobrist_hash(board)
    if root is None:
        root = MCTSNode(zobrist_key=root_zobrist)
    else:
        root.parent = None
        if root.zobrist_key is None:
            root.zobrist_key = root_zobrist

    if not root.is_expanded():
        root.expand(legal_moves)

    completed_sims = 0
    max_sims = simulations if simulations > 0 else 1
    base_history_counts = _build_history_zobrist_counts(board)

    while completed_sims < max_sims:
        if deadline is not None and completed_sims > 0 and time.perf_counter() >= deadline:
            break

        node = root
        work_board = board.copy(stack=False)
        sim_counts = dict(base_history_counts)
        rep_draw = False

        # 1. Selection
        while node.is_expanded():
            move, child = _select_uct_child(node, c_uct, local_rng)
            work_board.push(move)
            if child.zobrist_key is None:
                child.zobrist_key = compute_zobrist_hash(work_board)
            k = child.zobrist_key
            sim_counts[k] = sim_counts.get(k, 0) + 1
            node = child
            if sim_counts[k] >= 3:
                rep_draw = True
                break
            if node.visit_count == 0:
                break

        # 2. Expansion (nếu nút lá đã được thăm trước đó và chưa kết thúc)
        term_val: Optional[float] = 0.0 if rep_draw else _is_fast_terminal(work_board)
        if term_val is None and node.visit_count > 0 and not node.is_expanded():
            node_moves = list(work_board.legal_moves)
            if node_moves:
                node.expand(node_moves)
                move, child = _select_uct_child(node, c_uct, local_rng)
                work_board.push(move)
                if child.zobrist_key is None:
                    child.zobrist_key = compute_zobrist_hash(work_board)
                k = child.zobrist_key
                sim_counts[k] = sim_counts.get(k, 0) + 1
                node = child
                term_val = 0.0 if sim_counts[k] >= 3 else _is_fast_terminal(work_board)

        # 3. Simulation (Rollout cắt ngưỡng)
        if term_val is not None:
            leaf_value = term_val
        else:
            leaf_value = _rollout(work_board, max_rollout_depth, local_rng)

        # 4. Backpropagation
        _backpropagate(node, leaf_value)
        completed_sims += 1

    best_move, best_child = max(
        root.children.items(),
        key=lambda item: (item[1].visit_count, item[1].q_value),
    )

    elapsed = time.perf_counter() - t0
    stats: Dict[str, float] = {
        "nodes": float(completed_sims),
        "simulations": float(completed_sims),
        "depth": float(max_rollout_depth),
        "eval": float(best_child.q_value),
        "time_sec": float(elapsed),
    }
    return best_move, stats, root


def search_uct(
    board: chess.Board,
    simulations: int = 400,
    time_limit: Optional[float] = None,
    c_uct: float = 1.414,
    max_rollout_depth: int = 60,
) -> Tuple[chess.Move, Dict[str, float]]:
    """Thực thi tìm kiếm Vanilla MCTS (UCT) và trả về (best_move, stats)."""
    best_move, stats, _ = run_uct_search(
        board=board,
        simulations=simulations,
        time_limit=time_limit,
        c_uct=c_uct,
        max_rollout_depth=max_rollout_depth,
    )
    return best_move, stats
