"""
chess_ai/training/selfplay.py
Vòng lặp tự chơi (Self-play Loop), cập nhật mạng nơ-ron trên Replay Buffer
và cơ chế đấu loại trực tiếp Checkpoint Arena Gating (ngưỡng thắng >= 55%) cho Cấp 8.
Phụ trách: TV2 phối hợp TV1
"""

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import chess
import numpy as np
import torch

from chess_ai.agents.level5_supervised import build_resnet_from_config
from chess_ai.config import load_yaml_config
from chess_ai.core.board_tensor import board_to_tensor
from chess_ai.core.evaluation import evaluate_board
from chess_ai.core.zobrist import compute_zobrist_hash
from chess_ai.models.dual_resnet import DualHeadResNet
from chess_ai.models.loss import DualHeadLoss
from chess_ai.search.mcts_node import MCTSNode
from chess_ai.search.mcts_puct import run_puct_search
from chess_ai.training.replay_buffer import SelfPlayReplayBuffer
from chess_ai.utils.epd import load_epd_file


def _resolve_game_outcome_white(board: chess.Board, adjudicate_cp_threshold: float = 500.0) -> float:
    """
    Trả về kết quả ván cờ theo góc nhìn quân Trắng:
    +1.0 nếu Trắng thắng, -1.0 nếu Đen thắng, 0.0 nếu Hòa.
    Nếu ván đấu bị cắt sớm do chạm trần max_plies (chưa is_game_over), phân xử thắng/thua
    nếu một bên đang có ưu thế lượng giá áp đảo >= adjudicate_cp_threshold (mặc định 500 cp ~ 1 Xe).
    """
    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == chess.WHITE else -1.0

    if adjudicate_cp_threshold > 0.0:
        rel_cp = evaluate_board(board)
        white_cp = rel_cp if board.turn == chess.WHITE else -rel_cp
        if white_cp >= adjudicate_cp_threshold:
            return 1.0
        if white_cp <= -adjudicate_cp_threshold:
            return -1.0
    return 0.0


def play_selfplay_game(
    model: DualHeadResNet,
    mcts_sims: int = 200,
    c_puct: float = 1.414,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    temp_initial: float = 1.0,
    temp_final: float = 0.1,
    temp_decay_plies: int = 20,
    max_plies: int = 160,
    initial_fen: Optional[str] = None,
    subtree_reuse: bool = True,
    device: Optional[torch.device] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Chạy 1 ván tự chơi (Self-play) bằng PUCT ở chế độ selfplay_mode (eval_mode=False):
    - Bật nhiễu Dirichlet tại nút gốc (alpha=0.3, eps=0.25).
    - Hạ nhiệt độ chọn nước: temp_initial (1.0) trong temp_decay_plies (20) đầu, sau đó temp_final (0.1).
    - Thu thập toàn bộ bộ ba (state_tensor [18, 8, 8], pi_mcts [4096], z_t [-1, 0, +1]).
    """
    t0 = time.perf_counter()
    local_rng = rng if rng is not None else np.random.default_rng()
    board = chess.Board(initial_fen) if initial_fen else chess.Board()

    states_list: List[np.ndarray] = []
    policies_list: List[np.ndarray] = []
    turns_list: List[chess.Color] = []

    root: Optional[MCTSNode] = None
    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        temp = temp_initial if ply < temp_decay_plies else temp_final

        reusable_root: Optional[MCTSNode] = None
        if subtree_reuse and root is not None:
            target_key = compute_zobrist_hash(board)
            reusable_root = root.find_reusable_root(target_key)

        state_tensor = board_to_tensor(board)
        move, stats, new_root = run_puct_search(
            board=board,
            model=model,
            simulations=mcts_sims,
            c_puct=c_puct,
            eval_mode=False,
            temperature=temp,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            root=reusable_root,
            device=device,
            rng=local_rng,
        )

        states_list.append(state_tensor)
        policies_list.append(np.asarray(stats["pi"], dtype=np.float32))
        turns_list.append(board.turn)

        board.push(move)
        ply += 1
        root = new_root if subtree_reuse else None

    white_outcome = _resolve_game_outcome_white(board)
    n_samples = len(states_list)

    if n_samples == 0:
        empty_s = np.zeros((0, 18, 8, 8), dtype=np.float32)
        empty_p = np.zeros((0, 4096), dtype=np.float32)
        empty_v = np.zeros((0,), dtype=np.float32)
        return empty_s, empty_p, empty_v, {"plies": 0, "white_outcome": white_outcome}

    values_arr = np.array(
        [white_outcome if turn == chess.WHITE else -white_outcome for turn in turns_list],
        dtype=np.float32,
    )
    states_arr = np.stack(states_list, axis=0).astype(np.float32)
    policies_arr = np.stack(policies_list, axis=0).astype(np.float32)

    game_info: Dict[str, Any] = {
        "plies": int(ply),
        "samples": int(n_samples),
        "white_outcome": float(white_outcome),
        "result_str": board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2",
        "time_sec": float(time.perf_counter() - t0),
    }
    return states_arr, policies_arr, values_arr, game_info


def train_challenger_on_buffer(
    challenger_model: DualHeadResNet,
    replay_buffer: SelfPlayReplayBuffer,
    batch_size: int = 128,
    train_steps: int = 20,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Cập nhật trọng số mạng ứng viên (Challenger) trên các mini-batch lấy từ SelfPlayReplayBuffer
    bằng hàm mất mát kép DualHeadLoss (Soft Cross-Entropy + MSE) và AdamW.
    """
    if len(replay_buffer) == 0:
        return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "steps": 0.0}

    if device is None:
        try:
            device = next(challenger_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    challenger_model.to(device)
    challenger_model.train()

    criterion = DualHeadLoss(policy_weight=1.0, value_weight=1.0)
    optimizer = torch.optim.AdamW(
        challenger_model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    steps_run = max(1, int(train_steps))

    for _ in range(steps_run):
        s_np, p_np, v_np = replay_buffer.sample_batch(batch_size=batch_size)
        states_t = torch.from_numpy(s_np).to(device=device, dtype=torch.float32)
        policies_t = torch.from_numpy(p_np).to(device=device, dtype=torch.float32)
        values_t = torch.from_numpy(v_np).to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        logits_pred, value_pred = challenger_model(states_t)
        loss, parts = criterion(logits_pred, value_pred, policies_t, values_t)
        loss.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(challenger_model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss_sum += parts["total_loss"]
        policy_loss_sum += parts["policy_loss"]
        value_loss_sum += parts["value_loss"]

    challenger_model.eval()
    return {
        "total_loss": float(total_loss_sum / steps_run),
        "policy_loss": float(policy_loss_sum / steps_run),
        "value_loss": float(value_loss_sum / steps_run),
        "steps": float(steps_run),
    }


def _play_arena_duel_game(
    white_model: DualHeadResNet,
    black_model: DualHeadResNet,
    mcts_sims: int,
    c_puct: float,
    max_plies: int = 160,
    initial_fen: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> float:
    """
    Đấu 1 ván đánh giá giữa white_model và black_model ở chế độ eval_mode=True (tắt nhiễu, tau=0).
    Trả về kết quả theo góc nhìn Trắng (+1.0 Trắng thắng, 0.0 Hòa, -1.0 Đen thắng).
    """
    board = chess.Board(initial_fen) if initial_fen else chess.Board()
    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        active_model = white_model if board.turn == chess.WHITE else black_model
        move, _, _ = run_puct_search(
            board=board,
            model=active_model,
            simulations=mcts_sims,
            c_puct=c_puct,
            eval_mode=True,
            temperature=0.0,
            device=device,
        )
        board.push(move)
        ply += 1

    return _resolve_game_outcome_white(board)


def evaluate_arena_gating(
    challenger_model: DualHeadResNet,
    champion_model: DualHeadResNet,
    arena_games: int = 40,
    mcts_sims: int = 200,
    c_puct: float = 1.414,
    win_threshold: float = 0.55,
    max_plies: int = 160,
    initial_fens: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    iter_label: str = "",
) -> Dict[str, Any]:
    """
    Đấu loại trực tiếp giữa Challenger và Champion (chia đều số ván cầm Trắng / Đen).
    Tính tỷ lệ điểm quy đổi: score_rate = (wins + 0.5 * draws) / total_games.
    Chỉ chấp nhận Challenger (accepted = True) khi score_rate >= win_threshold (0.55).
    """
    challenger_model.eval()
    champion_model.eval()

    total_games = max(1, int(arena_games))
    wins = 0
    draws = 0
    losses = 0

    for g_idx in range(total_games):
        fen = None
        if initial_fens:
            fen = initial_fens[(g_idx // 2) % len(initial_fens)]

        challenger_is_white = (g_idx % 2 == 0)
        if challenger_is_white:
            res_white = _play_arena_duel_game(
                white_model=challenger_model,
                black_model=champion_model,
                mcts_sims=mcts_sims,
                c_puct=c_puct,
                max_plies=max_plies,
                initial_fen=fen,
                device=device,
            )
            challenger_outcome = res_white
        else:
            res_white = _play_arena_duel_game(
                white_model=champion_model,
                black_model=challenger_model,
                mcts_sims=mcts_sims,
                c_puct=c_puct,
                max_plies=max_plies,
                initial_fen=fen,
                device=device,
            )
            challenger_outcome = -res_white

        if challenger_outcome > 0.5:
            wins += 1
        elif challenger_outcome < -0.5:
            losses += 1
        else:
            draws += 1

        if verbose:
            curr_rate = (wins + 0.5 * draws) / (g_idx + 1)
            prefix = f"  [{iter_label}] " if iter_label else "  "
            print(
                f"\r{prefix}Pha 3 (Arena Gating): Ván {g_idx+1}/{total_games} | "
                f"W:{wins} D:{draws} L:{losses} (Score: {curr_rate*100:5.1f}%)",
                end="",
                flush=True,
            )

    if verbose:
        print("", flush=True)

    score_rate = float(wins + 0.5 * draws) / float(total_games)
    accepted = bool(score_rate >= float(win_threshold))

    return {
        "games": int(total_games),
        "wins": int(wins),
        "draws": int(draws),
        "losses": int(losses),
        "score_rate": float(score_rate),
        "win_threshold": float(win_threshold),
        "accepted": accepted,
    }


def should_promote_challenger(wins: int, draws: int, total_games: int, threshold: float = 0.55) -> Tuple[bool, float]:
    """Hàm kiểm tra điều kiện cổng Arena Gating >= 55% (phục vụ kiểm thử TC54)."""
    if total_games <= 0:
        return False, 0.0
    score_rate = float(wins + 0.5 * draws) / float(total_games)
    return bool(score_rate >= float(threshold)), score_rate


def run_selfplay_pipeline(
    config_path: str = "configs/search_params.yaml",
    warm_start_checkpoint: str = "checkpoints/best_model.pt",
    output_dir: str = "results/training/",
    checkpoint_dir: str = "checkpoints",
    iterations: int = 3,
    selfplay_games_override: Optional[int] = None,
    mcts_sims_override: Optional[int] = None,
    arena_games_override: Optional[int] = None,
    train_steps_per_iter: int = 20,
    batch_size: int = 128,
    learning_rate: float = 5e-4,
    max_plies: int = 160,
    resume_path: Optional[str] = None,
    initial_fens: Optional[List[str]] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Thực thi chu trình Self-play -> Training -> Arena Gating cho Cấp 8.
    Lưu checkpoint vô địch tại `checkpoints/l8_alphazero_best.pt`,
    checkpoint tạm thời để chạy tiếp tại `checkpoints/l8_alphazero_latest.pt`,
    và lịch sử huấn luyện tại `results/training/selfplay_metrics.json`.
    """
    search_cfg = load_yaml_config(Path(config_path).name)
    az_cfg = search_cfg.get("alphazero_level8", {})
    puct_cfg = search_cfg.get("guided_mcts", {})
    sp_mode_cfg = puct_cfg.get("selfplay_mode", {})

    warm_start_enabled = bool(az_cfg.get("warm_start_from_level5", True))
    games_per_iter = int(
        selfplay_games_override
        if selfplay_games_override is not None
        else az_cfg.get("selfplay_games_per_iter", 50)
    )
    mcts_sims = int(
        mcts_sims_override
        if mcts_sims_override is not None
        else az_cfg.get("mcts_sims_per_move", 200)
    )
    buffer_capacity = int(az_cfg.get("replay_buffer_size", 50000))
    arena_games = int(
        arena_games_override
        if arena_games_override is not None
        else az_cfg.get("arena_eval_games", 40)
    )
    win_threshold = float(az_cfg.get("arena_win_threshold", 0.55))

    c_puct = float(puct_cfg.get("c_puct", 1.414))
    dir_alpha = float(sp_mode_cfg.get("dirichlet_alpha", 0.3))
    dir_eps = float(sp_mode_cfg.get("dirichlet_epsilon", 0.25))
    temp_init = float(sp_mode_cfg.get("temp_initial", 1.0))
    temp_final = float(sp_mode_cfg.get("temp_final", 0.1))
    temp_decay = int(sp_mode_cfg.get("temp_decay_plies", 20))

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = load_yaml_config("model_resnet.yaml")
    champion_model = build_resnet_from_config(model_cfg)
    warm_started = False

    if warm_start_enabled and warm_start_checkpoint and Path(warm_start_checkpoint).exists():
        payload = torch.load(warm_start_checkpoint, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "model_config" in payload:
            model_cfg = payload["model_config"]
            champion_model = build_resnet_from_config(model_cfg)
        state_dict = (
            payload["model_state_dict"]
            if isinstance(payload, dict) and "model_state_dict" in payload
            else payload
        )
        champion_model.load_state_dict(state_dict)
        warm_started = True

    champion_model.to(device)
    champion_model.eval()

    replay_buffer = SelfPlayReplayBuffer(capacity=buffer_capacity, seed=seed)
    history: List[Dict[str, Any]] = []
    start_iter = 1
    champion_version = 0

    if resume_path and Path(resume_path).exists():
        resume_payload = torch.load(resume_path, map_location=device, weights_only=False)
        if isinstance(resume_payload, dict):
            if "model_config" in resume_payload:
                model_cfg = resume_payload["model_config"]
                champion_model = build_resnet_from_config(model_cfg).to(device)
            if "model_state_dict" in resume_payload:
                champion_model.load_state_dict(resume_payload["model_state_dict"])
            if "replay_buffer" in resume_payload:
                replay_buffer.load_state_dict(resume_payload["replay_buffer"])
            history = list(resume_payload.get("history", []))
            start_iter = int(resume_payload.get("iteration", 0)) + 1
            champion_version = int(resume_payload.get("champion_version", 0))
            champion_model.eval()

    if initial_fens is None:
        for cand_epd in (Path("resources/openings_50.epd"), Path("../resources/openings_50.epd")):
            if cand_epd.exists():
                try:
                    initial_fens = [rec.fen for rec in load_epd_file(str(cand_epd))]
                    break
                except Exception:
                    initial_fens = None

    ckpt_dir_path = Path(checkpoint_dir)
    ckpt_dir_path.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir_path / "l8_alphazero_best.pt"
    latest_ckpt_path = ckpt_dir_path / "l8_alphazero_latest.pt"

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir_path / "selfplay_metrics.json"

    if not best_ckpt_path.exists():
        torch.save(
            {
                "iteration": start_iter - 1,
                "champion_version": champion_version,
                "model_state_dict": champion_model.state_dict(),
                "model_config": model_cfg,
                "warm_started": warm_started,
            },
            best_ckpt_path,
        )

    if verbose:
        print(
            f"Bắt đầu huấn luyện Cấp 8 (AlphaZero Self-Play) trên thiết bị: {device} | "
            f"Warm-start L5: {warm_started} ({warm_start_checkpoint}) | "
            f"Iterations: {start_iter}->{iterations} | "
            f"Self-Play: {games_per_iter} ván/iter ({mcts_sims} sims/nước, max_plies={max_plies}) | "
            f"Arena Gating: {arena_games} ván (ngưỡng >= {win_threshold*100:.0f}%)",
            flush=True,
        )

    for it in range(start_iter, int(iterations) + 1):
        it_t0 = time.perf_counter()
        sp_games_stats: List[Dict[str, Any]] = []
        samples_added = 0
        iter_tag = f"Iter {it}/{iterations}"

        # Pha 1: Self-Play sinh dữ liệu
        for g in range(games_per_iter):
            if initial_fens and g > 0:
                fen = initial_fens[((it - 1) * games_per_iter + g) % len(initial_fens)]
            else:
                fen = None
            s_arr, p_arr, v_arr, g_info = play_selfplay_game(
                model=champion_model,
                mcts_sims=mcts_sims,
                c_puct=c_puct,
                dirichlet_alpha=dir_alpha,
                dirichlet_epsilon=dir_eps,
                temp_initial=temp_init,
                temp_final=temp_final,
                temp_decay_plies=temp_decay,
                max_plies=max_plies,
                initial_fen=fen,
                device=device,
                rng=rng,
            )
            replay_buffer.push_game_samples(s_arr, p_arr, v_arr)
            samples_added += int(s_arr.shape[0])
            sp_games_stats.append(g_info)
            if verbose:
                print(
                    f"\r  [{iter_tag}] Pha 1 (Self-Play): Ván {g+1}/{games_per_iter} "
                    f"({g_info['plies']} plies, kq={g_info['white_outcome']:+.0f}) | "
                    f"Buffer: {len(replay_buffer):,} mẫu",
                    end="",
                    flush=True,
                )
        if verbose:
            print("", flush=True)

        # Pha 2: Cập nhật mạng Challenger trên Replay Buffer
        challenger_model = copy.deepcopy(champion_model).to(device)
        train_stats = train_challenger_on_buffer(
            challenger_model=challenger_model,
            replay_buffer=replay_buffer,
            batch_size=min(batch_size, max(1, len(replay_buffer))),
            train_steps=train_steps_per_iter,
            learning_rate=learning_rate,
            device=device,
        )
        if verbose:
            print(
                f"  [{iter_tag}] Pha 2 (Refinement): {int(train_stats['steps'])} steps | "
                f"Loss: {train_stats['total_loss']:.4f} "
                f"(Policy: {train_stats['policy_loss']:.4f}, Value: {train_stats['value_loss']:.4f})",
                flush=True,
            )

        # Pha 3: Đấu loại Arena Gating (>= 55%)
        arena_fens = (
            [initial_fens[((it - 1) * max(1, arena_games // 2) + k) % len(initial_fens)] for k in range(len(initial_fens))]
            if initial_fens
            else None
        )
        arena_stats = evaluate_arena_gating(
            challenger_model=challenger_model,
            champion_model=champion_model,
            arena_games=arena_games,
            mcts_sims=mcts_sims,
            c_puct=c_puct,
            win_threshold=win_threshold,
            max_plies=max_plies,
            initial_fens=arena_fens,
            device=device,
            verbose=verbose,
            iter_label=iter_tag,
        )

        if arena_stats["accepted"]:
            champion_model.load_state_dict(challenger_model.state_dict())
            champion_model.eval()
            champion_version += 1
            torch.save(
                {
                    "iteration": it,
                    "champion_version": champion_version,
                    "model_state_dict": champion_model.state_dict(),
                    "model_config": model_cfg,
                    "arena_score_rate": arena_stats["score_rate"],
                    "warm_started": warm_started,
                },
                best_ckpt_path,
            )

        iter_record: Dict[str, Any] = {
            "iteration": int(it),
            "selfplay_games": int(games_per_iter),
            "samples_added": int(samples_added),
            "buffer_size": int(len(replay_buffer)),
            "train_total_loss": round(float(train_stats["total_loss"]), 5),
            "train_policy_loss": round(float(train_stats["policy_loss"]), 5),
            "train_value_loss": round(float(train_stats["value_loss"]), 5),
            "arena": arena_stats,
            "champion_version": int(champion_version),
            "elapsed_sec": round(float(time.perf_counter() - it_t0), 2),
        }
        history.append(iter_record)

        torch.save(
            {
                "iteration": int(it),
                "champion_version": int(champion_version),
                "model_state_dict": champion_model.state_dict(),
                "model_config": model_cfg,
                "replay_buffer": replay_buffer.state_dict(),
                "history": history,
                "warm_started": warm_started,
            },
            latest_ckpt_path,
        )

        report_payload: Dict[str, Any] = {
            "experiment": "Level8_AlphaZero_SelfPlay_Gating",
            "device": str(device),
            "warm_started": warm_started,
            "warm_start_checkpoint": str(warm_start_checkpoint),
            "iterations_completed": int(it),
            "champion_version": int(champion_version),
            "best_checkpoint": str(best_ckpt_path),
            "latest_checkpoint": str(latest_ckpt_path),
            "buffer_size": int(len(replay_buffer)),
            "history": history,
        }
        metrics_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        if verbose:
            status_tag = "★ ACCEPTED (Cập nhật Champion mới)" if arena_stats["accepted"] else "REJECTED (Giữ nguyên Champion)"
            print(
                f"  ==> [{iter_tag} Hoàn tất] Arena Score: {arena_stats['score_rate']*100:.1f}% "
                f"(W:{arena_stats['wins']} D:{arena_stats['draws']} L:{arena_stats['losses']}) -> {status_tag} | "
                f"Champion v{champion_version} | {iter_record['elapsed_sec']:.1f}s\n",
                flush=True,
            )

    return {
        "device": str(device),
        "warm_started": warm_started,
        "iterations_completed": len(history),
        "champion_version": int(champion_version),
        "best_checkpoint": str(best_ckpt_path),
        "latest_checkpoint": str(latest_ckpt_path),
        "buffer_size": int(len(replay_buffer)),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện tự chơi AlphaZero-inspired (Level 8).")
    parser.add_argument("--config", type=str, default="configs/search_params.yaml", help="Đường dẫn cấu hình YAML")
    parser.add_argument(
        "--warm-start",
        type=str,
        default="checkpoints/best_model.pt",
        help="Checkpoint Level 5 dùng để khởi tạo ấm (Warm-start)",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Thư mục lưu checkpoint Level 8")
    parser.add_argument("--output-dir", type=str, default="results/training/", help="Thư mục lưu selfplay_metrics.json")
    parser.add_argument("--iterations", type=int, default=3, help="Tổng số vòng lặp Self-Play")
    parser.add_argument("--games-per-iter", type=int, default=None, help="Ghi đè số ván tự chơi mỗi vòng")
    parser.add_argument("--sims", type=int, default=None, help="Ghi đè số lượt mô phỏng MCTS mỗi nước đi")
    parser.add_argument("--arena-games", type=int, default=None, help="Ghi đè số ván đấu loại Arena Gating")
    parser.add_argument("--train-steps", type=int, default=20, help="Số bước cập nhật mini-batch mỗi vòng")
    parser.add_argument("--batch-size", type=int, default=128, help="Kích thước batch huấn luyện")
    parser.add_argument("--max-plies", type=int, default=160, help="Giới hạn số plies tối đa mỗi ván tự chơi")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn l8_alphazero_latest.pt để chạy tiếp")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_selfplay_pipeline(
        config_path=args.config,
        warm_start_checkpoint=args.warm_start,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        iterations=args.iterations,
        selfplay_games_override=args.games_per_iter,
        mcts_sims_override=args.sims,
        arena_games_override=args.arena_games,
        train_steps_per_iter=args.train_steps,
        batch_size=args.batch_size,
        max_plies=args.max_plies,
        resume_path=args.resume,
        seed=args.seed,
        verbose=True,
    )


if __name__ == "__main__":
    main()
