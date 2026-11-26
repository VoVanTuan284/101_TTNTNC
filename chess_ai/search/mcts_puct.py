"""
chess_ai/search/mcts_puct.py
Động cơ Guided MCTS sử dụng công thức PUCT tích hợp Policy Prior P(s, a) và Value Head V(s)
dành cho Tác tử Cấp 7 và Cấp 8.
Hỗ trợ:
  - Tách biệt eval_mode (tắt nhiễu) và selfplay_mode (bật Dirichlet noise)
  - FPU Reduction (First Play Urgency) chống lãng phí simulation ở thế cờ bất lợi
  - Phát hiện hòa lặp lại 3 lần (Threefold Repetition) trong cây tìm kiếm
  - NN Evaluation Cache theo khóa Zobrist 64-bit
  - Batched Leaf Evaluation kèm Virtual Loss cho tăng tốc GPU
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""

from collections import OrderedDict
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import chess
import numpy as np
import torch

from chess_ai.core.board_tensor import board_to_tensor, move_to_action_index
from chess_ai.core.zobrist import compute_zobrist_hash
from chess_ai.search.mcts_node import MCTSNode


EvalCache = OrderedDict[int, Tuple[Dict[chess.Move, float], float]]


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


def _build_history_zobrist_counts(board: chess.Board) -> Dict[int, int]:
    """
    Trích xuất bảng đếm khóa Zobrist của các thế cờ lịch sử kể từ nước đi không thể đảo ngược gần nhất
    (dựa trên halfmove_clock) để phát hiện hòa lặp lại 3 lần trong cây MCTS mà không cần copy(stack=True)
    ở mỗi simulation.
    """
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


def _logits_to_legal_priors(
    logits_np: np.ndarray,
    moves: List[chess.Move],
    turn: chess.Color,
) -> Dict[chess.Move, float]:
    """Áp Legal Action Masking + Softmax ổn định số học lên vector logits (4096,)."""
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
    return {m: float(p) for m, p in zip(moves, probs)}


@torch.no_grad()
def evaluate_board_with_model(
    board: chess.Board,
    model: Any,
    legal_moves: Optional[List[chess.Move]] = None,
    device: Optional[torch.device] = None,
    uniform_prior: bool = False,
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

    if uniform_prior:
        uniform = 1.0 / len(moves)
        return {m: uniform for m in moves}, float(np.clip(value, -1.0, 1.0))

    priors = _logits_to_legal_priors(logits_np, moves, board.turn)
    return priors, float(np.clip(value, -1.0, 1.0))


@torch.no_grad()
def evaluate_boards_batch_with_model(
    boards: List[chess.Board],
    legal_moves_list: List[List[chess.Move]],
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    uniform_prior: bool = False,
) -> List[Tuple[Dict[chess.Move, float], float]]:
    """Đánh giá đồng thời một batch bàn cờ (B, 18, 8, 8) bằng một lượt forward PyTorch duy nhất."""
    if not boards:
        return []
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    tensors = np.stack([board_to_tensor(b) for b in boards], axis=0)
    x = torch.from_numpy(tensors).to(device)
    logits_t, value_t = model(x)
    logits_batch = logits_t.detach().cpu().numpy()
    values_batch = value_t.squeeze(-1).detach().cpu().numpy()

    results: List[Tuple[Dict[chess.Move, float], float]] = []
    for i, (b, moves) in enumerate(zip(boards, legal_moves_list)):
        if not moves:
            results.append(({}, 0.0))
            continue
        if uniform_prior:
            uniform = 1.0 / len(moves)
            priors = {m: uniform for m in moves}
        else:
            priors = _logits_to_legal_priors(logits_batch[i], moves, b.turn)
        val = float(np.clip(float(values_batch[i]), -1.0, 1.0))
        results.append((priors, val))
    return results


def _cache_get(
    cache: Optional[EvalCache],
    zobrist_key: int,
) -> Optional[Tuple[Dict[chess.Move, float], float]]:
    if cache is None:
        return None
    item = cache.get(zobrist_key)
    if item is not None:
        cache.move_to_end(zobrist_key)
    return item


def _cache_put(
    cache: Optional[EvalCache],
    zobrist_key: int,
    priors: Dict[chess.Move, float],
    value: float,
    max_size: int,
) -> None:
    if cache is None or max_size <= 0:
        return
    cache[zobrist_key] = (priors, value)
    cache.move_to_end(zobrist_key)
    while len(cache) > max_size:
        cache.popitem(last=False)


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


def _select_puct_child(
    node: MCTSNode,
    c_puct: float,
    fpu_reduction: float = 0.2,
) -> Tuple[chess.Move, MCTSNode]:
    """
    Chọn nhánh con theo công thức PUCT có FPU Reduction:
        score(s, a) = Q_eff(s, a) + c_puct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a))
    Với nút chưa thăm (N = 0), Q_eff = max(-1.0, V_parent - fpu_reduction), trong đó
    V_parent = -node.q_value (theo góc nhìn bên tới lượt tại `node`).
    """
    total_child_visits = sum(c.visit_count for c in node.children.values())
    sqrt_parent = math.sqrt(max(1, total_child_visits))

    if node.visit_count > 0:
        parent_val_for_turn = -node.q_value
        fpu_value = max(-1.0, parent_val_for_turn - fpu_reduction)
    else:
        fpu_value = 0.0

    best_score = -float("inf")
    best_item: Optional[Tuple[chess.Move, MCTSNode]] = None

    for move, child in node.children.items():
        q_val = child.q_value if child.visit_count > 0 else fpu_value
        u_val = c_puct * child.prior * (sqrt_parent / (1.0 + child.visit_count))
        puct = q_val + u_val
        if puct > best_score:
            best_score = puct
            best_item = (move, child)

    assert best_item is not None
    return best_item


def _apply_virtual_loss(node: MCTSNode, vloss: float = 1.0) -> None:
    """Áp virtual loss dọc đường đi từ nút lá lên gốc để các luồng trong cùng batch phân tán sang nhánh khác."""
    curr: Optional[MCTSNode] = node
    while curr is not None:
        curr.visit_count += 1
        curr.value_sum -= vloss
        curr = curr.parent


def _revert_virtual_loss(node: MCTSNode, vloss: float = 1.0) -> None:
    """Hoàn tác virtual loss trước khi lan truyền ngược giá trị thật."""
    curr: Optional[MCTSNode] = node
    while curr is not None:
        curr.visit_count -= 1
        curr.value_sum += vloss
        curr = curr.parent


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
    - pi_4096: phân phối chuẩn hóa từ N(s_root, a) / sum_b N(s_root, b).
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
    fpu_reduction: float = 0.2,
    eval_cache: Optional[EvalCache] = None,
    max_cache_size: int = 8192,
    batch_size: int = 1,
    uniform_prior: bool = False,
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
    history_counts = _build_history_zobrist_counts(board)
    cache_hits = 0

    if root is None:
        root = MCTSNode(zobrist_key=root_zobrist)
    else:
        root.parent = None
        if root.zobrist_key is None:
            root.zobrist_key = root_zobrist

    # Đánh giá nút gốc (tận dụng cache hoặc subtree đã mở rộng khi ở eval_mode)
    if root.is_expanded() and eval_mode:
        cached_root = _cache_get(eval_cache, root_zobrist)
        if cached_root is not None:
            cache_hits += 1
            root_value = cached_root[1]
        else:
            root_value = -root.q_value if root.visit_count > 0 else 0.0
    else:
        cached_root = _cache_get(eval_cache, root_zobrist)
        if cached_root is not None:
            cache_hits += 1
            root_priors, root_value = cached_root
        else:
            root_priors, root_value = evaluate_board_with_model(
                board=board,
                model=model,
                legal_moves=legal_moves,
                device=device,
                uniform_prior=uniform_prior,
            )
            _cache_put(eval_cache, root_zobrist, root_priors, root_value, max_cache_size)

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
    eff_batch_size = max(1, int(batch_size)) if isinstance(model, torch.nn.Module) else 1
    root_fpu = 0.0 if not eval_mode else float(fpu_reduction)

    while completed_sims < max_sims:
        if deadline is not None and completed_sims > 0 and time.perf_counter() >= deadline:
            break

        if eff_batch_size == 1:
            node = root
            work_board = board.copy(stack=False)
            depth = 0
            path_counts: Dict[int, int] = {}

            # 1. Selection: đi xuống theo PUCT (có FPU reduction) cho tới khi gặp nút lá
            while node.is_expanded():
                node_fpu = root_fpu if node is root else float(fpu_reduction)
                move, child = _select_puct_child(node, c_puct, fpu_reduction=node_fpu)
                work_board.push(move)
                depth += 1
                if child.zobrist_key is None:
                    child.zobrist_key = compute_zobrist_hash(work_board)
                path_counts[child.zobrist_key] = path_counts.get(child.zobrist_key, 0) + 1
                node = child

            if depth > max_tree_depth:
                max_tree_depth = depth

            # 2. Kiểm tra hòa lặp lại 3 lần / kết thúc / Expansion & Evaluation
            z_key = node.zobrist_key if node.zobrist_key is not None else compute_zobrist_hash(work_board)
            total_rep = history_counts.get(z_key, 0) + path_counts.get(z_key, 0)
            if total_rep >= 3 or (depth >= 2 and total_rep >= 2):
                leaf_value = 0.0
            else:
                term_val = _is_fast_terminal(work_board)
                if term_val is not None:
                    leaf_value = term_val
                else:
                    node_moves = list(work_board.legal_moves)
                    if not node_moves:
                        leaf_value = 0.0
                    else:
                        cached = _cache_get(eval_cache, z_key)
                        if cached is not None:
                            cache_hits += 1
                            priors, leaf_value = cached
                        else:
                            priors, leaf_value = evaluate_board_with_model(
                                board=work_board,
                                model=model,
                                legal_moves=node_moves,
                                device=device,
                                uniform_prior=uniform_prior,
                            )
                            _cache_put(eval_cache, z_key, priors, leaf_value, max_cache_size)
                        node.expand(node_moves, priors=priors)

            # 3. Backpropagation (chuẩn dấu Negamax)
            _backpropagate(node, leaf_value)
            completed_sims += 1
        else:
            # Batched Leaf Evaluation với Virtual Loss (tăng tốc mạnh trên GPU)
            target_batch = min(eff_batch_size, max_sims - completed_sims)
            pending_nodes: List[MCTSNode] = []
            pending_boards: List[chess.Board] = []
            pending_moves: List[List[chess.Move]] = []
            pending_keys: List[int] = []
            immediate_updates: List[Tuple[MCTSNode, float]] = []

            for _ in range(target_batch):
                node = root
                work_board = board.copy(stack=False)
                depth = 0
                path_counts = {}

                while node.is_expanded():
                    node_fpu = root_fpu if node is root else float(fpu_reduction)
                    move, child = _select_puct_child(node, c_puct, fpu_reduction=node_fpu)
                    work_board.push(move)
                    depth += 1
                    if child.zobrist_key is None:
                        child.zobrist_key = compute_zobrist_hash(work_board)
                    path_counts[child.zobrist_key] = path_counts.get(child.zobrist_key, 0) + 1
                    node = child

                if depth > max_tree_depth:
                    max_tree_depth = depth

                _apply_virtual_loss(node)
                z_key = node.zobrist_key if node.zobrist_key is not None else compute_zobrist_hash(work_board)
                total_rep = history_counts.get(z_key, 0) + path_counts.get(z_key, 0)

                if total_rep >= 3 or (depth >= 2 and total_rep >= 2):
                    immediate_updates.append((node, 0.0))
                    continue

                term_val = _is_fast_terminal(work_board)
                if term_val is not None:
                    immediate_updates.append((node, term_val))
                    continue

                node_moves = list(work_board.legal_moves)
                if not node_moves:
                    immediate_updates.append((node, 0.0))
                    continue

                cached = _cache_get(eval_cache, z_key)
                if cached is not None:
                    cache_hits += 1
                    priors, leaf_val = cached
                    if not node.is_expanded():
                        node.expand(node_moves, priors=priors)
                    immediate_updates.append((node, leaf_val))
                    continue

                pending_nodes.append(node)
                pending_boards.append(work_board)
                pending_moves.append(node_moves)
                pending_keys.append(z_key)

            # Hoàn tác virtual loss & lan truyền ngược cho các lá terminal/cache
            for node, val in immediate_updates:
                _revert_virtual_loss(node)
                _backpropagate(node, val)
                completed_sims += 1

            # Forward 1 lượt duy nhất cho toàn bộ batch lá mới
            if pending_nodes:
                batch_outs = evaluate_boards_batch_with_model(
                    boards=pending_boards,
                    legal_moves_list=pending_moves,
                    model=model,
                    device=device,
                    uniform_prior=uniform_prior,
                )
                for node, moves, z_key, (priors, leaf_val) in zip(
                    pending_nodes, pending_moves, pending_keys, batch_outs
                ):
                    _cache_put(eval_cache, z_key, priors, leaf_val, max_cache_size)
                    if not node.is_expanded():
                        node.expand(moves, priors=priors)
                    _revert_virtual_loss(node)
                    _backpropagate(node, leaf_val)
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
        "cache_hits": float(cache_hits),
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
    fpu_reduction: float = 0.2,
    batch_size: int = 1,
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
        fpu_reduction=fpu_reduction,
        batch_size=batch_size,
    )
    return best_move, stats
