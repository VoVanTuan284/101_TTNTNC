"""
chess_ai/training/train_rl_krk.py
Kịch bản huấn luyện tác tử Học tăng cường Cấp 6 (Tabular Q-Learning cho KRK/KQK/KPK
và Policy Gradient REINFORCE đối chiếu trên KRK), lưu bảng Q ra checkpoints/l6_krk_qtable.pkl
và xuất dữ liệu đường cong học tập ra results/training/krk_learning_curve.json.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import pickle
import platform
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import chess
import numpy as np
import torch
import torch.nn as nn

from chess_ai.agents.level6_reinforcement import (
    CanonicalMoveKey,
    EndgameStateKey,
    encode_endgame_state_and_move,
)
from chess_ai.config import load_yaml_config
from chess_ai.core.chess_env import ChessEnv


def generate_endgame_fen(
    endgame_type: str,
    rng: random.Random,
    curriculum_easy: bool = False,
) -> str:
    """Sinh thế cờ tàn ngẫu nhiên hợp lệ (KRK, KQK hoặc KPK) trên ChessEnv 8x8."""
    piece_char = {"KRK": chess.ROOK, "KQK": chess.QUEEN, "KPK": chess.PAWN}[endgame_type]
    while True:
        if curriculum_easy and endgame_type in ("KRK", "KQK"):
            # Đặt Vua Đen ở cạnh/góc bàn cờ và Vua Trắng ở gần để dễ tìm thấy chiếu hết ban đầu
            edge_squares = [
                sq for sq in range(64)
                if chess.square_rank(sq) in (0, 7) or chess.square_file(sq) in (0, 7)
            ]
            bk = rng.choice(edge_squares)
            wk_candidates = [
                sq for sq in range(64)
                if 2 <= chess.square_distance(sq, bk) <= 3
            ]
            wk = rng.choice(wk_candidates)
            wp = rng.randrange(64)
        elif curriculum_easy and endgame_type == "KPK":
            wp_rank = rng.randint(4, 6)
            wp_file = rng.randrange(8)
            wp = chess.square(wp_file, wp_rank)
            wk_candidates = [
                sq for sq in range(64)
                if sq != wp and chess.square_distance(sq, wp) <= 2
            ]
            wk = rng.choice(wk_candidates)
            bk = rng.randrange(64)
        else:
            wk = rng.randrange(64)
            bk = rng.randrange(64)
            if endgame_type == "KPK":
                wp = chess.square(rng.randrange(8), rng.randint(1, 6))
            else:
                wp = rng.randrange(64)

        if bk == wk or wp in (wk, bk) or chess.square_distance(wk, bk) <= 1:
            continue
        # Tránh sinh thế cờ mất ngay Xe/Hậu/Tốt không bảo vệ ở nước đầu tiên
        if chess.square_distance(wp, bk) <= 1 and chess.square_distance(wp, wk) > 1:
            continue

        board = chess.Board(None)
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(wp, chess.Piece(piece_char, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE

        if board.is_valid() and not board.is_checkmate() and not board.is_stalemate():
            return board.fen()


def _endgame_potential(board: chess.Board, endgame_type: str) -> float:
    """Hàm thế năng hình học (Potential-based Reward Shaping) bảo toàn chính sách tối ưu."""
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return 0.0

    if endgame_type == "KPK":
        p_mask = board.pieces_mask(chess.PAWN, chess.WHITE)
        if p_mask:
            wp = chess.lsb(p_mask)
            return 0.04 * chess.square_rank(wp) - 0.02 * chess.square_distance(wk, wp)
        return 0.0

    bk_f, bk_r = chess.square_file(bk), chess.square_rank(bk)
    bk_edge_dist = min(bk_f, 7 - bk_f, bk_r, 7 - bk_r)
    king_dist = chess.square_distance(wk, bk)
    return -0.03 * bk_edge_dist - 0.02 * king_dist


def _opponent_reply(board: chess.Board, rng: random.Random) -> chess.Move:
    """Nước đáp trả của Vua Đen: ưu tiên bắt quân không bảo vệ, ngược lại chọn ngẫu nhiên."""
    captures = list(board.generate_legal_captures())
    if captures:
        return rng.choice(captures)
    return rng.choice(list(board.legal_moves))


def _terminal_reward(
    board: chess.Board,
    result: Optional[float],
    win_reward: float,
    draw_penalty: float,
) -> float:
    """Phân tách thưởng kết cục: +1.0 chiếu hết | -1.0 mất quân (insufficient material) | -0.2 hòa khác."""
    if result == 1.0:
        return win_reward
    if board.is_insufficient_material():
        return -win_reward
    return draw_penalty


def _train_tabular_q_single_endgame(
    endgame_type: str,
    episodes: int,
    gamma: float,
    alpha: float,
    step_penalty: float,
    win_reward: float,
    draw_penalty: float,
    eps_start: float,
    eps_end: float,
    max_steps: int,
    seed: int,
    eval_every: int = 1000,
    shared_kqk_table: Optional[Dict[Tuple[EndgameStateKey, CanonicalMoveKey], float]] = None,
    verbose: bool = True,
) -> Tuple[Dict[Tuple[EndgameStateKey, CanonicalMoveKey], float], List[Dict[str, Any]]]:
    """Huấn luyện Tabular Q-Learning trên ChessEnv cho một thế cờ tàn cụ thể."""
    rng = random.Random(seed)
    q_table: Dict[Tuple[EndgameStateKey, CanonicalMoveKey], float] = defaultdict(float)
    curve: List[Dict[str, Any]] = []

    recent_wins = 0
    recent_steps = 0
    tick_every = max(1, episodes // 50)

    for ep in range(1, episodes + 1):
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - ep / float(max(1, int(episodes * 0.80))))
        curriculum_easy = ep <= int(episodes * 0.40)
        env = ChessEnv(fen=generate_endgame_fen(endgame_type, rng, curriculum_easy=curriculum_easy))

        steps = 0
        won = False

        while not env.is_terminal() and steps < max_steps:
            board = env.board
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            # Nếu KPK vừa phong Hậu -> dùng bảng KQK đã học (hoặc tiếp tục học)
            if endgame_type == "KPK" and shared_kqk_table is not None and board.pieces_mask(chess.QUEEN, chess.WHITE):
                best_m = legal_moves[0]
                best_q = -float("inf")
                for mv in legal_moves:
                    enc = encode_endgame_state_and_move(board, mv)
                    if enc is not None:
                        qv = shared_kqk_table.get((enc[0], enc[1]), -1.0)
                        if qv > best_q:
                            best_q = qv
                            best_m = mv
                env.step(best_m)
                steps += 1
                if not env.is_terminal():
                    env.step(_opponent_reply(env.board, rng))
                    steps += 1
                if env.is_terminal() and env.result() == 1.0:
                    won = True
                continue

            encoded_pairs = []
            for mv in legal_moves:
                enc = encode_endgame_state_and_move(board, mv)
                if enc is not None:
                    encoded_pairs.append((mv, enc[0], enc[1]))

            if not encoded_pairs:
                break

            prev_pot = _endgame_potential(board, endgame_type)

            if rng.random() < eps:
                chosen_mv, s_key, m_key = rng.choice(encoded_pairs)
            else:
                best_q = max(q_table[(sk, mk)] for _, sk, mk in encoded_pairs)
                best_items = [item for item in encoded_pairs if q_table[(item[1], item[2])] == best_q]
                chosen_mv, s_key, m_key = rng.choice(best_items)

            env.step(chosen_mv)
            steps += 1

            # Đối thủ Đen đáp trả nếu chưa hết ván
            if not env.is_terminal():
                env.step(_opponent_reply(env.board, rng))
                steps += 1

            if env.is_terminal():
                res = env.result()
                if res == 1.0:
                    won = True
                reward = _terminal_reward(env.board, res, win_reward, draw_penalty)
                target = reward
            elif steps >= max_steps:
                reward = draw_penalty
                target = reward
            else:
                new_pot = _endgame_potential(env.board, endgame_type)
                reward = step_penalty + (gamma * new_pot - prev_pot)
                next_legal = list(env.board.legal_moves)
                lookup_table = (
                    shared_kqk_table
                    if (endgame_type == "KPK" and shared_kqk_table is not None and env.board.pieces_mask(chess.QUEEN, chess.WHITE))
                    else q_table
                )
                next_qs = []
                for nmv in next_legal:
                    nenc = encode_endgame_state_and_move(env.board, nmv)
                    if nenc is not None:
                        next_qs.append(lookup_table.get((nenc[0], nenc[1]), 0.0))
                max_next_q = max(next_qs) if next_qs else 0.0
                target = reward + gamma * max_next_q

            old_q = q_table[(s_key, m_key)]
            q_table[(s_key, m_key)] = old_q + alpha * (target - old_q)

        if won:
            recent_wins += 1
        recent_steps += steps

        if verbose and (ep % tick_every == 0 or ep == episodes):
            pct = int(ep * 100 / episodes)
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"\r  [{endgame_type}] [{bar}] {pct:3d}% ({ep}/{episodes})", end="", flush=True)

        if ep % eval_every == 0 or ep == episodes:
            denom = eval_every if ep % eval_every == 0 else (ep % eval_every)
            win_r = round(recent_wins / denom, 4)
            avg_p = round(recent_steps / denom, 2)
            curve.append(
                {
                    "episode": ep,
                    "win_rate": win_r,
                    "avg_plies": avg_p,
                    "epsilon": round(eps, 4),
                    "q_states": len(q_table),
                }
            )
            if verbose:
                print(
                    f"\r  [{endgame_type}] Ep {ep:5d}/{episodes} | WinRate: {win_r*100:5.1f}% "
                    f"| AvgPlies: {avg_p:5.1f} | Eps: {eps:.2f} | States: {len(q_table)}",
                    flush=True,
                )
            recent_wins = 0
            recent_steps = 0

    return dict(q_table), curve


class _ReinforcePolicyNet(nn.Module):
    """Mạng chính sách nhỏ đánh giá cặp (trạng thái KRK, nước đi ứng viên) cho REINFORCE."""

    def __init__(self, in_dim: int = 12, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _krk_action_features(board: chess.Board, move: chess.Move) -> List[float]:
    """Trích xuất 12 đặc trưng hình học chuẩn hóa cho nước đi trong thế cờ KRK."""
    wk = board.king(chess.WHITE) or 0
    bk = board.king(chess.BLACK) or 0
    wr_mask = board.pieces_mask(chess.ROOK, chess.WHITE)
    wr = chess.lsb(wr_mask) if wr_mask else 0

    wk_f, wk_r = chess.square_file(wk), chess.square_rank(wk)
    bk_f, bk_r = chess.square_file(bk), chess.square_rank(bk)
    wr_f, wr_r = chess.square_file(wr), chess.square_rank(wr)
    to_f, to_r = chess.square_file(move.to_square), chess.square_rank(move.to_square)

    is_rook_move = 1.0 if move.from_square == wr else 0.0
    n_wr_f, n_wr_r = (to_f, to_r) if is_rook_move else (wr_f, wr_r)
    n_wk_f, n_wk_r = (wk_f, wk_r) if is_rook_move else (to_f, to_r)

    king_cheb = max(abs(n_wk_f - bk_f), abs(n_wk_r - bk_r)) / 7.0
    king_manh = (abs(n_wk_f - bk_f) + abs(n_wk_r - bk_r)) / 14.0
    bk_edge = min(bk_f, 7 - bk_f, bk_r, 7 - bk_r) / 3.5
    rook_dist = (abs(n_wr_f - bk_f) + abs(n_wr_r - bk_r)) / 14.0
    gives_check = 1.0 if board.gives_check(move) else 0.0
    rook_same_line = 1.0 if (n_wr_f == bk_f or n_wr_r == bk_r) else 0.0

    return [
        n_wk_f / 7.0,
        n_wk_r / 7.0,
        n_wr_f / 7.0,
        n_wr_r / 7.0,
        bk_f / 7.0,
        bk_r / 7.0,
        king_cheb,
        king_manh,
        bk_edge,
        rook_dist,
        gives_check,
        rook_same_line,
    ]


def _train_reinforce_krk(
    episodes: int,
    gamma: float,
    step_penalty: float,
    win_reward: float,
    draw_penalty: float,
    max_steps: int,
    seed: int,
    eval_every: int = 500,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Huấn luyện Policy Gradient (REINFORCE với moving-average baseline) trên KRK."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    policy = _ReinforcePolicyNet()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    curve: List[Dict[str, Any]] = []
    baseline = 0.0
    recent_wins = 0
    recent_steps = 0
    tick_every = max(1, episodes // 50)

    for ep in range(1, episodes + 1):
        curriculum_easy = ep <= int(episodes * 0.40)
        env = ChessEnv(fen=generate_endgame_fen("KRK", rng, curriculum_easy=curriculum_easy))
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        rewards: List[float] = []
        steps = 0
        won = False

        while not env.is_terminal() and steps < max_steps:
            board = env.board
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            prev_pot = _endgame_potential(board, "KRK")
            feats = torch.tensor(
                [_krk_action_features(board, mv) for mv in legal_moves],
                dtype=torch.float32,
            )
            logits = policy(feats)
            dist = torch.distributions.Categorical(logits=logits)
            idx = dist.sample()
            log_probs.append(dist.log_prob(idx))
            entropies.append(dist.entropy())

            env.step(legal_moves[int(idx.item())])
            steps += 1
            if not env.is_terminal():
                env.step(_opponent_reply(env.board, rng))
                steps += 1

            if env.is_terminal():
                res = env.result()
                if res == 1.0:
                    won = True
                rewards.append(_terminal_reward(env.board, res, win_reward, draw_penalty))
            elif steps >= max_steps:
                rewards.append(draw_penalty)
            else:
                new_pot = _endgame_potential(env.board, "KRK")
                rewards.append(step_penalty + (gamma * new_pot - prev_pot))

        if log_probs:
            returns: List[float] = []
            g = 0.0
            for r in reversed(rewards):
                g = r + gamma * g
                returns.insert(0, g)
            ret_t = torch.tensor(returns, dtype=torch.float32)
            ep_return = float(ret_t[0].item())
            baseline = 0.95 * baseline + 0.05 * ep_return
            adv = ret_t - baseline
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            entropy_bonus = torch.stack(entropies).mean()
            loss = -(torch.stack(log_probs) * adv).mean() - 0.02 * entropy_bonus
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if won:
            recent_wins += 1
        recent_steps += steps

        if verbose and (ep % tick_every == 0 or ep == episodes):
            pct = int(ep * 100 / episodes)
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"\r  [REINFORCE KRK] [{bar}] {pct:3d}% ({ep}/{episodes})", end="", flush=True)

        if ep % eval_every == 0 or ep == episodes:
            denom = eval_every if ep % eval_every == 0 else (ep % eval_every)
            win_r = round(recent_wins / denom, 4)
            avg_p = round(recent_steps / denom, 2)
            curve.append(
                {
                    "episode": ep,
                    "win_rate": win_r,
                    "avg_plies": avg_p,
                }
            )
            if verbose:
                print(
                    f"\r  [REINFORCE KRK] Ep {ep:5d}/{episodes} | WinRate: {win_r*100:5.1f}% | AvgPlies: {avg_p:5.1f}",
                    flush=True,
                )
            recent_wins = 0
            recent_steps = 0

    return curve


def _evaluate_greedy_policy(
    endgame_type: str,
    q_table: Dict[Tuple[EndgameStateKey, CanonicalMoveKey], float],
    num_games: int = 100,
    max_steps: int = 80,
    seed: int = 999,
    shared_kqk_table: Optional[Dict[Tuple[EndgameStateKey, CanonicalMoveKey], float]] = None,
) -> Dict[str, Any]:
    """Đánh giá chính sách greedy (epsilon = 0) sau huấn luyện để ghi số liệu báo cáo."""
    rng = random.Random(seed)
    wins = 0
    total_plies = 0

    for _ in range(num_games):
        env = ChessEnv(fen=generate_endgame_fen(endgame_type, rng))
        steps = 0
        while not env.is_terminal() and steps < max_steps:
            board = env.board
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            active_table = (
                shared_kqk_table
                if (endgame_type == "KPK" and shared_kqk_table is not None and board.pieces_mask(chess.QUEEN, chess.WHITE))
                else q_table
            )
            best_mv = legal_moves[0]
            best_q = -float("inf")
            for mv in legal_moves:
                enc = encode_endgame_state_and_move(board, mv)
                if enc is not None:
                    qv = active_table.get((enc[0], enc[1]), -1.0)
                    if qv > best_q:
                        best_q = qv
                        best_mv = mv
            env.step(best_mv)
            steps += 1
            if not env.is_terminal():
                env.step(_opponent_reply(env.board, rng))
                steps += 1
        if env.is_terminal() and env.result() == 1.0:
            wins += 1
        total_plies += steps

    return {
        "eval_games": num_games,
        "greedy_win_rate": round(wins / max(1, num_games), 4),
        "greedy_avg_plies": round(total_plies / max(1, num_games), 2),
    }


def train_krk_rl(
    config_path: str = "configs/search_params.yaml",
    output_qtable: str = "checkpoints/l6_krk_qtable.pkl",
    output_metrics: str = "results/training/krk_learning_curve.json",
    episodes_override: Optional[int] = None,
    reinforce_episodes: int = 2000,
    endgames: Tuple[str, ...] = ("KQK", "KRK", "KPK"),
    seed: int = 42,
    verbose: bool = True,
    eval_games: int = 100,
) -> Dict[str, Any]:
    """Thực thi huấn luyện RL (Tabular Q-Learning + REINFORCE) và lưu bảng Q + Learning Curve."""
    t0 = time.perf_counter()
    cfg = load_yaml_config(config_path)["rl_level6"]

    episodes = episodes_override if episodes_override is not None else int(cfg.get("episodes", 20000))
    gamma = float(cfg.get("gamma", 0.99))
    alpha = max(0.25, float(cfg.get("learning_rate", 0.1)))
    step_penalty = -0.005
    win_reward = float(cfg.get("win_reward", 1.0))
    draw_penalty = float(cfg.get("draw_penalty", -0.2))
    eps_start = float(cfg.get("epsilon_start", 1.0))
    eps_end = float(cfg.get("epsilon_end", 0.05))
    max_steps = 100
    eval_every = max(100, episodes // 20)

    q_tables: Dict[str, Dict[Tuple[EndgameStateKey, CanonicalMoveKey], float]] = {}
    q_curves: Dict[str, List[Dict[str, Any]]] = {}
    stage_summary: Dict[str, Any] = {}

    if verbose:
        print(f"Bắt đầu huấn luyện Cấp 6 RL: Q-Learning {list(endgames)} ({episodes} ep/thế) + REINFORCE KRK ({reinforce_episodes} ep)", flush=True)

    for eg in endgames:
        t_stage = time.perf_counter()
        shared_kqk = q_tables.get("KQK") if eg == "KPK" else None
        table, curve = _train_tabular_q_single_endgame(
            endgame_type=eg,
            episodes=episodes,
            gamma=gamma,
            alpha=alpha,
            step_penalty=step_penalty,
            win_reward=win_reward,
            draw_penalty=draw_penalty,
            eps_start=eps_start,
            eps_end=eps_end,
            max_steps=max_steps,
            seed=seed,
            eval_every=eval_every,
            shared_kqk_table=shared_kqk,
            verbose=verbose,
        )
        stage_elapsed = round(time.perf_counter() - t_stage, 2)
        eval_stats = _evaluate_greedy_policy(
            endgame_type=eg,
            q_table=table,
            num_games=eval_games,
            max_steps=max_steps,
            seed=seed + 1000,
            shared_kqk_table=shared_kqk,
        )
        q_tables[eg] = table
        q_curves[eg] = curve
        stage_summary[f"tabular_q_{eg}"] = {
            "elapsed_sec": stage_elapsed,
            "q_states": len(table),
            "final_train_win_rate": curve[-1]["win_rate"] if curve else 0.0,
            "final_train_avg_plies": curve[-1]["avg_plies"] if curve else 0.0,
            **eval_stats,
        }

    t_reinf = time.perf_counter()
    reinforce_eval_every = max(50, reinforce_episodes // 10)
    reinforce_curve = _train_reinforce_krk(
        episodes=reinforce_episodes,
        gamma=gamma,
        step_penalty=step_penalty,
        win_reward=win_reward,
        draw_penalty=draw_penalty,
        max_steps=max_steps,
        seed=seed,
        eval_every=reinforce_eval_every,
        verbose=verbose,
    )
    stage_summary["reinforce_KRK"] = {
        "elapsed_sec": round(time.perf_counter() - t_reinf, 2),
        "final_train_win_rate": reinforce_curve[-1]["win_rate"] if reinforce_curve else 0.0,
        "final_train_avg_plies": reinforce_curve[-1]["avg_plies"] if reinforce_curve else 0.0,
    }

    q_out = Path(output_qtable)
    q_out.parent.mkdir(parents=True, exist_ok=True)
    with open(q_out, "wb") as f:
        pickle.dump({"q_tables": q_tables, "endgames": list(endgames)}, f)

    metrics_payload: Dict[str, Any] = {
        "experiment": "Level6_RL_Endgames",
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "hardware_environment": {
            "device": "cpu",
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
        },
        "hyperparameters": {
            "config_path": config_path,
            "tabular_q_episodes_per_endgame": episodes,
            "reinforce_krk_episodes": reinforce_episodes,
            "gamma": gamma,
            "learning_rate": alpha,
            "reinforce_lr": 0.001,
            "step_penalty": step_penalty,
            "win_reward": win_reward,
            "draw_penalty": draw_penalty,
            "epsilon_start": eps_start,
            "epsilon_end": eps_end,
            "max_steps": max_steps,
            "seed": seed,
        },
        "checkpoint_path": str(q_out),
        "checkpoint_size_kb": round(q_out.stat().st_size / 1024.0, 2),
        "endgames": list(endgames),
        "stage_summary": stage_summary,
        "tabular_q_curves": q_curves,
        "reinforce_krk_curve": reinforce_curve,
    }
    m_out = Path(output_metrics)
    m_out.parent.mkdir(parents=True, exist_ok=True)
    with open(m_out, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện RL Cấp 6 (Tabular Q-Learning KQK/KRK/KPK + REINFORCE KRK).")
    parser.add_argument("--config", default="configs/search_params.yaml", help="Đường dẫn tệp YAML.")
    parser.add_argument("--episodes", type=int, default=None, help="Số episode Q-Learning cho mỗi thế cờ tàn.")
    parser.add_argument("--reinforce-episodes", type=int, default=2000, help="Số episode REINFORCE trên KRK.")
    args = parser.parse_args()

    res = train_krk_rl(
        config_path=args.config,
        episodes_override=args.episodes,
        reinforce_episodes=args.reinforce_episodes,
    )
    print(f"Huấn luyện Cấp 6 hoàn tất trong {res['elapsed_sec']}s.")
    for eg, curve in res["tabular_q_curves"].items():
        last = curve[-1]
        print(f"  - Tabular Q [{eg}]: WinRate={last['win_rate']*100:.1f}% | AvgPlies={last['avg_plies']} | States={last['q_states']}")
    r_last = res["reinforce_krk_curve"][-1]
    print(f"  - REINFORCE [KRK]: WinRate={r_last['win_rate']*100:.1f}% | AvgPlies={r_last['avg_plies']}")


if __name__ == "__main__":
    main()
