"""
chess_ai/search/mcts_puct.py
Động cơ Guided MCTS sử dụng công thức PUCT tích hợp Policy Prior P(s, a) và Value Head V(s)
dành cho Tác tử Cấp 7 và Cấp 8. Tách biệt rõ eval_mode (tắt nhiễu) và selfplay_mode (bật Dirichlet noise).
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""

import math
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import chess
import numpy as np
import torch

from chess_ai.core.board_tensor import board_to_tensor, move_to_action_index
from chess_ai.core.zobrist import compute_zobrist_hash
from chess_ai.search.mcts_node import MCTSNode


def _is_fast_terminal(board: chess.Board) -> Optional[float]:
    """
    Kiểm tra kết thúc ván cờ tại nút cây.
    Trả về giá trị từ góc nhìn bên tới lượt đi (-1.0 nếu bị chiếu hết, 0.0 nếu hòa, None nếu chưa kết thúc).
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


@torch.no_grad()
def evaluate_board_with_model(
    board: chess.Board,
    model: Any,
    legal_moves: Optional[List[chess.Move]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[chess.Move, float], float]:
    """
    Đánh giá trạng thái bàn cờ bằng mạng DualHeadResNet (hoặc hàm callback tương đương).
    Trả về:
        - priors: Dict[chess.Move, float] có tổng xác suất trên các nước hợp lệ = 1.0
        - value: Giá trị thế cờ V(s) trong [-1.0, 1.0] theo góc nhìn bên tới lượt đi.
    """
    moves = legal_moves if legal_moves is not None else list(board.legal_moves)
    if not moves:
        return {}, 0.0

    logits_np: Optional[np.ndarray] = None
    value: float = 0.0

    if isinstance(model, torch.nn.Module):
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        tensor_np = board_to_tensor(board)
        x = torch.from_numpy(tensor_np).unsqueeze(0).to(device)
        logits_t, value_t = model(x)
        logits_np = logits_t[0].detach().cpu().numpy()
        value = float(value_t[0, 0].detach().cpu().item())
    elif hasattr(model, "predict_policy_and_value"):
        probs_4096, val = model.predict_policy_and_value(board)
        value = float(val)
        turn = board.turn
        raw_probs = np.array(
            [max(0.0, float(probs_4096[move_to_action_index(m, turn)])) for m in moves],
            dtype=np.float64,
        )
        total_p = float(np.sum(raw_probs))
        if total_p > 0.0:
            norm_probs = raw_probs / total_p
        else:
            norm_probs = np.full(len(moves), 1.0 / len(moves), dtype=np.float64)
        return {m: float(p) for m, p in zip(moves, norm_probs)}, float(np.clip(value, -1.0, 1.0))
    elif callable(model):
        out = model(board)
        if isinstance(out, tuple) and len(out) == 2:
            first, second = out
            value = float(second)
            if isinstance(first, dict):
                total_p = sum(max(0.0, float(first.get(m, 0.0))) for m in moves)
                if total_p > 0.0:
                    return (
                        {m: max(0.0, float(first.get(m, 0.0))) / total_p for m in moves},
                        float(np.clip(value, -1.0, 1.0)),
                    )
                uniform = 1.0 / len(moves)
                return {m: uniform for m in moves}, float(np.clip(value, -1.0, 1.0))
            logits_np = np.asarray(first, dtype=np.float32)

    if logits_np is None:
        uniform = 1.0 / len(moves)
        return {m: uniform for m in moves}, 0.0

    turn = board.turn
    move_logits = np.array(
        [float(logits_np[move_to_action_index(m, turn)]) for m in moves],
        dtype=np.float64,
    )
    max_logit = float(np.max(move_logits))
    exp_scores = np.exp(move_logits - max_logit)
    sum_exp = float(np.sum(exp_scores))
    if sum_exp > 0.0 and math.isfinite(sum_exp):
        probs = exp_scores / sum_exp
    else:
        probs = np.full(len(moves), 1.0 / len(moves), dtype=np.float64)

    priors = {m: float(p) for m, p in zip(moves, probs)}
    return priors, float(np.clip(value, -1.0, 1.0))


def _apply_root_dirichlet_noise(
    root: MCTSNode,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    Trộn nhiễu Dirichlet vào xác suất tiên nghiệm P(s_root, a) tại nút gốc:
        P'(s_root, a) = (1 - eps) * P(s_root, a) + eps * Eta_a,  Eta ~ Dir(alpha)
    """
    if not root.children or dirichlet_epsilon <= 0.0:
        return
    moves = list(root.children.keys())
    k = len(moves)
    gen = rng if rng is not None else np.random.default_rng()
    noise = gen.dirichlet(np.full(k, float(dirichlet_alpha), dtype=np.float64))
    eps = float(dirichlet_epsilon)
    for move, eta in zip(moves, noise):
        child = root.children[move]
        child.prior = float((1.0 - eps) * child.prior + eps * float(eta))


def _select_puct_child(node: MCTSNode, c_puct: float) -> Tuple[chess.Move, MCTSNode]:
    """
    Chọn nhánh con theo công thức PUCT:
        score(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a))
    Trong đó Q(s, a) đã được lưu theo góc nhìn của nút cha `node`.
    """
    total_child_visits = sum(c.visit_count for c in node.children.values())
    sqrt_parent = math.sqrt(max(1, total_child_visits))

    best_score = -float("inf")
    best_item: Optional[Tuple[chess.Move, MCTSNode]] = None

    for move, child in node.children.items():
        q_val = child.q_value
        u_val = c_puct * child.prior * (sqrt_parent / (1.0 + child.visit_count))
        puct = q_val + u_val
        if puct > best_score:
            best_score = puct
            best_item = (move, child)

    assert best_item is not None
    return best_item


def _backpropagate(node: MCTSNode, leaf_value_from_node_turn: float) -> None:
    """
    Lan truyền ngược giá trị lên nút gốc theo quy ước dấu Negamax (TC35):
    `value_sum` tại mỗi nút con lưu theo góc nhìn của nút cha (bên vừa đi nước tới nút con đó).
    """
    curr: Optional[MCTSNode] = node
    value_for_parent = -float(leaf_value_from_node_turn)
    while curr is not None:
        curr.visit_count += 1
        curr.value_sum += value_for_parent
        value_for_parent = -value_for_parent
        curr = curr.parent


def _compute_root_policy_target(
    root: MCTSNode,
    turn: chess.Color,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, Dict[chess.Move, float]]:
    """
    Tính phân phối chính sách MCTS pi (4096,) và xác suất chọn nước tại nút gốc từ số lượt thăm N(s_root, a).
    - pi_4096: phân phối chuẩn hóa từ N(s_root, a) / sum_b N(s_root, b) (hoặc theo nhiệt độ).
    - move_probs: xác suất tương ứng với từng nước đi hợp lệ ở nút gốc để chọn/lấy mẫu nước đi.
    """
    pi_4096 = np.zeros(4096, dtype=np.float32)
    if not root.children:
        return pi_4096, {}

    moves = list(root.children.keys())
    visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float64)
    total_visits = float(np.sum(visits))

    if total_visits <= 0.0:
        priors = np.array([max(0.0, root.children[m].prior) for m in moves], dtype=np.float64)
        p_sum = float(np.sum(priors))
        visit_dist = priors / p_sum if p_sum > 0.0 else np.full(len(moves), 1.0 / len(moves))
    else:
        visit_dist = visits / total_visits

    # Vector mục tiêu huấn luyện pi_4096 luôn phản ánh phân phối lượt thăm chuẩn hóa tại gốc
    for m, prob in zip(moves, visit_dist):
        idx = move_to_action_index(m, turn)
        pi_4096[idx] += float(prob)
    pi_sum = float(np.sum(pi_4096))
    if pi_sum > 0.0:
        pi_4096 /= pi_sum

    # Xác suất chọn nước đi thực tế phụ thuộc vào nhiệt độ (temperature)
    if temperature <= 1e-3 or total_visits <= 0.0:
        best_idx = int(
            max(
                range(len(moves)),
                key=lambda i: (
                    root.children[moves[i]].visit_count,
                    root.children[moves[i]].q_value,
                    root.children[moves[i]].prior,
                ),
            )
        )
        move_probs = {m: (1.0 if i == best_idx else 0.0) for i, m in enumerate(moves)}
    else:
        inv_temp = 1.0 / float(temperature)
        max_v = float(np.max(visits))
        scaled = np.power(visits / max(1.0, max_v), inv_temp)
        s_sum = float(np.sum(scaled))
        probs = scaled / s_sum if s_sum > 0.0 else visit_dist
        move_probs = {m: float(p) for m, p in zip(moves, probs)}

    return pi_4096, move_probs


def run_puct_search(
    board: chess.Board,
    model: Any,
    simulations: int = 400,
    time_limit: Optional[float] = None,
    c_puct: float = 1.414,
    eval_mode: bool = True,
    temperature: Optional[float] = None,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    root: Optional[MCTSNode] = None,
    device: Optional[torch.device] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[chess.Move, Dict[str, Any], MCTSNode]:
    """
    Thực thi tìm kiếm Guided MCTS (PUCT) dẫn hướng bởi Policy Prior P(s, a) và Value Head V(s).
    Trả về (selected_move, stats_and_pi, root_node).
    """
    t0 = time.perf_counter()
    deadline = (t0 + time_limit) if time_limit is not None else None
    local_rng = rng if rng is not None else np.random.default_rng()

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

    root_priors, root_value = evaluate_board_with_model(
        board=board,
        model=model,
        legal_moves=legal_moves,
        device=device,
    )
    if not root.is_expanded():
        root.expand(legal_moves, priors=root_priors)
    else:
        for m, p in root_priors.items():
            if m in root.children:
                root.children[m].prior = p

    if not eval_mode:
        _apply_root_dirichlet_noise(
            root=root,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            rng=local_rng,
        )

    completed_sims = 0
    max_sims = simulations if simulations > 0 else 1
    max_tree_depth = 1

    while completed_sims < max_sims:
        if deadline is not None and completed_sims > 0 and time.perf_counter() >= deadline:
            break

        node = root
        work_board = board.copy(stack=False)
        depth = 0

        # 1. Selection: đi xuống theo PUCT cho tới khi gặp nút lá
        while node.is_expanded():
            move, child = _select_puct_child(node, c_puct)
            work_board.push(move)
            depth += 1
            if child.zobrist_key is None:
                child.zobrist_key = compute_zobrist_hash(work_board)
            node = child

        if depth > max_tree_depth:
            max_tree_depth = depth

        # 2. Expansion & Evaluation bằng mạng Dual-Head ResNet (không dùng random rollout)
        term_val = _is_fast_terminal(work_board)
        if term_val is not None:
            leaf_value = term_val
        else:
            node_moves = list(work_board.legal_moves)
            if not node_moves:
                leaf_value = 0.0
            else:
                priors, leaf_value = evaluate_board_with_model(
                    board=work_board,
                    model=model,
                    legal_moves=node_moves,
                    device=device,
                )
                node.expand(node_moves, priors=priors)

        # 3. Backpropagation (chuẩn dấu Negamax)
        _backpropagate(node, leaf_value)
        completed_sims += 1

    eff_temp = (0.0 if eval_mode else 1.0) if temperature is None else float(temperature)
    pi_4096, move_probs = _compute_root_policy_target(
        root=root,
        turn=board.turn,
        temperature=eff_temp,
    )

    if eval_mode or eff_temp <= 1e-3:
        best_move, best_child = max(
            root.children.items(),
            key=lambda item: (item[1].visit_count, item[1].q_value, item[1].prior),
        )
    else:
        moves_list = list(move_probs.keys())
        probs_arr = np.array([move_probs[m] for m in moves_list], dtype=np.float64)
        p_sum = float(np.sum(probs_arr))
        if p_sum > 0.0:
            probs_arr /= p_sum
        else:
            probs_arr = np.full(len(moves_list), 1.0 / len(moves_list), dtype=np.float64)
        chosen_idx = int(local_rng.choice(len(moves_list), p=probs_arr))
        best_move = moves_list[chosen_idx]
        best_child = root.children[best_move]

    elapsed = time.perf_counter() - t0
    stats: Dict[str, Any] = {
        "nodes": float(completed_sims),
        "simulations": float(completed_sims),
        "depth": float(max_tree_depth),
        "eval": float(best_child.q_value),
        "root_value": float(root_value),
        "time_sec": float(elapsed),
        "pi": pi_4096,
        "visit_counts": {m.uci(): int(c.visit_count) for m, c in root.children.items()},
    }
    return best_move, stats, root


def search_puct(
    board: chess.Board,
    model: Any,
    simulations: int = 400,
    time_limit: Optional[float] = None,
    c_puct: float = 1.414,
    eval_mode: bool = True,
    temperature: Optional[float] = None,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
) -> Tuple[chess.Move, Dict[str, Any]]:
    """Thực thi tìm kiếm Guided MCTS (PUCT) và trả về (best_move, stats_and_pi)."""
    best_move, stats, _ = run_puct_search(
        board=board,
        model=model,
        simulations=simulations,
        time_limit=time_limit,
        c_puct=c_puct,
        eval_mode=eval_mode,
        temperature=temperature,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )
    return best_move, stats
