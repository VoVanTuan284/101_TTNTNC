"""
chess_ai/training/train_rl_krk.py
Kịch bản huấn luyện tác tử Học tăng cường Cấp 6:
  - Afterstate Value Learning V(s_after) kết hợp Backward TD(lambda) & Staged Curriculum
    cho 3 thế cờ tàn chuẩn FIDE (KQK, KRK, KPK) trên không gian đối xứng D4 (~35k trạng thái).
  - Policy Gradient Batch REINFORCE đối chiếu trên KRK (18 đặc trưng hình học & chiến thuật),
    lưu cả bảng V-table và trọng số REINFORCE ra checkpoints/l6_krk_qtable.pkl,
    xuất dữ liệu đường cong học tập ra results/training/krk_learning_curve.json.
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
import torch

from chess_ai.agents.level6_reinforcement import (
    EndgameStateKey,
    REINFORCE_FEATURE_DIM,
    ReinforcePolicyNet,
    encode_afterstate,
    is_piece_hanging_after_move,
    krk_action_features,
)
from chess_ai.config import load_yaml_config
from chess_ai.core.chess_env import ChessEnv


_EDGE_SQUARES: List[int] = [
    sq for sq in range(64)
    if chess.square_rank(sq) in (0, 7) or chess.square_file(sq) in (0, 7)
]


def generate_endgame_fen(
    endgame_type: str,
    rng: random.Random,
    curriculum_easy: bool = False,
    curriculum_stage: Optional[int] = None,
) -> str:
    """
    Sinh thế cờ tàn ngẫu nhiên hợp lệ (KRK, KQK hoặc KPK) trên ChessEnv 8x8.
    - curriculum_stage = 1: rất gần chiếu hết / gần phong Hậu (học tín hiệu cuối ván nhanh).
    - curriculum_stage = 2: Vua Đen ở cạnh bàn cờ / Tốt ở hàng 5-6 (giống curriculum_easy=True).
    - curriculum_stage = 3 (hoặc None & curriculum_easy=False): ngẫu nhiên toàn bàn 8x8.
    """
    if curriculum_stage is None:
        curriculum_stage = 2 if curriculum_easy else 3

    piece_char = {"KRK": chess.ROOK, "KQK": chess.QUEEN, "KPK": chess.PAWN}[endgame_type]
    edge_squares = _EDGE_SQUARES

    while True:
        if curriculum_stage == 1 and endgame_type in ("KRK", "KQK"):
            # Gần chiếu hết: Vua Đen ở cạnh/góc, Vua Trắng đối mặt cách đúng 2 ô
            bk = rng.choice(edge_squares)
            wk_candidates = [
                sq for sq in range(64)
                if chess.square_distance(sq, bk) == 2
            ]
            wk = rng.choice(wk_candidates)
            wp = rng.randrange(64)
        elif curriculum_stage == 2 and endgame_type in ("KRK", "KQK"):
            # Trung bình: Vua Đen ở cạnh/góc, Vua Trắng cách 2-3 ô
            bk = rng.choice(edge_squares)
            wk_candidates = [
                sq for sq in range(64)
                if 2 <= chess.square_distance(sq, bk) <= 3
            ]
            wk = rng.choice(wk_candidates)
            wp = rng.randrange(64)
        elif curriculum_stage == 1 and endgame_type == "KPK":
            # Tốt ở hàng 6-7, Vua Trắng đứng sát hỗ trợ, Vua Đen ở xa hơn một chút
            wp_rank = rng.randint(5, 6)
            wp_file = rng.randrange(8)
            wp = chess.square(wp_file, wp_rank)
            wk_candidates = [
                sq for sq in range(64)
                if sq != wp and chess.square_distance(sq, wp) == 1
            ]
            wk = rng.choice(wk_candidates)
            bk_candidates = [
                sq for sq in range(64)
                if sq != wp and chess.square_distance(sq, wk) >= 2 and chess.square_distance(sq, wp) >= 2
            ]
            bk = rng.choice(bk_candidates)
        elif curriculum_stage == 2 and endgame_type == "KPK":
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
            return 0.05 * chess.square_rank(wp) - 0.025 * chess.square_distance(wk, wp)
        if board.pieces_mask(chess.QUEEN, chess.WHITE):
            return 0.35
        return 0.0

    bk_f, bk_r = chess.square_file(bk), chess.square_rank(bk)
    bk_edge_dist = min(bk_f, 7 - bk_f, bk_r, 7 - bk_r)
    king_dist = chess.square_distance(wk, bk)
    return -0.035 * bk_edge_dist - 0.02 * king_dist


def _opponent_reply(
    board: chess.Board,
    rng: random.Random,
    center_bias_prob: float = 0.5,
) -> chess.Move:
    """
    Nước đáp trả của Vua Đen:
    - Luôn ưu tiên bắt quân không bảo vệ nếu có.
    - Với xác suất center_bias_prob: kéo Vua về trung tâm (chống bị ép góc) để tác tử học ép Vua thật sự.
    - Ngược lại: chọn ngẫu nhiên hợp lệ để đa dạng hóa trạng thái.
    """
    captures = list(board.generate_legal_captures())
    if captures:
        return rng.choice(captures)
    legal = list(board.legal_moves)
    if rng.random() < center_bias_prob:
        def _center_dist(mv: chess.Move) -> float:
            f = chess.square_file(mv.to_square)
            r = chess.square_rank(mv.to_square)
            return abs(f - 3.5) + abs(r - 3.5)

        min_d = min(_center_dist(m) for m in legal)
        best = [m for m in legal if _center_dist(m) == min_d]
        return rng.choice(best)
    return rng.choice(legal)


def _fast_endgame_status(
    board: chess.Board,
    rep_count: int = 0,
) -> Tuple[bool, Optional[float], str]:
    """
    Kiểm tra kết thúc ván cờ tàn siêu nhanh O(1) (tránh quét toàn bộ move_stack và tạo FEN mỗi bước).
    Trả về (is_done, result_for_white, reason).
    """
    if board.occupied.bit_count() <= 2:
        return True, 0.0, "insufficient"
    if not any(board.generate_legal_moves()):
        if board.is_check():
            return True, (1.0 if board.turn == chess.BLACK else -1.0), "checkmate"
        return True, 0.0, "stalemate"
    if rep_count >= 3:
        return True, 0.0, "repetition"
    if board.halfmove_clock >= 100:
        return True, 0.0, "fifty_moves"
    return False, None, ""


def _terminal_reward_from_reason(
    reason: str,
    result: Optional[float],
    win_reward: float,
    draw_penalty: float,
) -> float:
    """Phân tách thưởng kết cục: +1.0 chiếu hết | -1.0 mất quân | -0.8 hòa pat | -0.5 lặp nước | draw_penalty."""
    if result == 1.0:
        return win_reward
    if reason == "insufficient":
        return -win_reward
    if reason == "stalemate":
        return -0.8 * win_reward
    if reason == "repetition":
        return -0.5 * win_reward
    return draw_penalty


def _terminal_reward(
    board: chess.Board,
    result: Optional[float],
    win_reward: float,
    draw_penalty: float,
) -> float:
    """Phân tách thưởng kết cục: +1.0 chiếu hết | -1.0 mất quân | -0.8 hòa pat | draw_penalty hòa khác."""
    if result == 1.0:
        return win_reward
    if board.is_insufficient_material():
        return -win_reward
    if board.is_stalemate():
        return -0.8 * win_reward
    if board.is_repetition(3):
        return -0.5 * win_reward
    return draw_penalty


def _score_afterstate_candidate(
    board: chess.Board,
    move: chess.Move,
    endgame_type: str,
    v_table: Dict[EndgameStateKey, float],
    shared_kqk_table: Optional[Dict[EndgameStateKey, float]] = None,
    seen_afterstates: Optional[Dict[EndgameStateKey, int]] = None,
) -> Tuple[float, Optional[EndgameStateKey]]:
    """Đánh giá nhanh giá trị kỳ vọng của một nước đi thông qua Afterstate V(s_after)."""
    board.push(move)
    has_reply = any(board.generate_legal_moves())
    if not has_reply:
        is_mate = board.is_check()
        board.pop()
        return (1.5 if is_mate else -1.0), None

    after_key = encode_afterstate(board)
    val = 0.0
    if after_key is not None:
        if after_key[0] == endgame_type:
            val = v_table.get(after_key, 0.0)
            if seen_afterstates is not None and seen_afterstates.get(after_key, 0) >= 1:
                val -= 0.4
        elif endgame_type == "KPK" and after_key[0] == "KQK":
            if is_piece_hanging_after_move(board, chess.WHITE):
                val = -1.0
            else:
                kqk_v = shared_kqk_table.get(after_key, 0.0) if shared_kqk_table else 0.0
                val = max(0.85, kqk_v)
    board.pop()
    return val, after_key


def _train_afterstate_td_single_endgame(
    endgame_type: str,
    episodes: int,
    gamma: float,
    alpha_start: float,
    alpha_end: float,
    td_lambda: float,
    step_penalty: float,
    win_reward: float,
    draw_penalty: float,
    eps_start: float,
    eps_end: float,
    max_steps: int,
    seed: int,
    eval_every: int = 1000,
    shared_kqk_table: Optional[Dict[EndgameStateKey, float]] = None,
    verbose: bool = True,
) -> Tuple[Dict[EndgameStateKey, float], List[Dict[str, Any]]]:
    """
    Huấn luyện Afterstate Value Learning V(s_after) kết hợp Backward TD(lambda)
    và 3-Stage Curriculum Learning trên ChessEnv.
    """
    rng = random.Random(seed)
    v_table: Dict[EndgameStateKey, float] = defaultdict(float)
    curve: List[Dict[str, Any]] = []

    recent_wins = 0
    recent_steps = 0
    tick_every = max(1, episodes // 50)

    for ep in range(1, episodes + 1):
        progress = ep / float(max(1, episodes))
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1.0 - ep / float(max(1, int(episodes * 0.80))))
        alpha = alpha_end + (alpha_start - alpha_end) * (1.0 - progress)

        if progress <= 0.25:
            stage = 1
        elif progress <= 0.55:
            stage = 2
        else:
            stage = 3

        env = ChessEnv(fen=generate_endgame_fen(endgame_type, rng, curriculum_stage=stage))
        board = env.board

        # Lưu quỹ đạo các bước (after_key, reward) để cập nhật ngược Backward TD(lambda)
        trajectory: List[Tuple[Optional[EndgameStateKey], float]] = []
        seen_afterstates: Dict[EndgameStateKey, int] = defaultdict(int)
        steps = 0
        won = False

        while steps < max_steps:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            # Nếu KPK vừa phong Hậu -> dùng bảng KQK đã học để dứt điểm
            if endgame_type == "KPK" and shared_kqk_table is not None and board.pieces_mask(chess.QUEEN, chess.WHITE):
                best_m = legal_moves[0]
                best_v = -float("inf")
                for mv in legal_moves:
                    sc, _ = _score_afterstate_candidate(board, mv, "KQK", shared_kqk_table, None, seen_afterstates)
                    if sc > best_v:
                        best_v = sc
                        best_m = mv
                board.push(best_m)
                steps += 1
                ak_kqk = encode_afterstate(board)
                rep_kqk = 0
                if ak_kqk is not None:
                    seen_afterstates[ak_kqk] += 1
                    rep_kqk = seen_afterstates[ak_kqk]
                done, res, _ = _fast_endgame_status(board, rep_kqk)
                if not done:
                    board.push(_opponent_reply(board, rng))
                    steps += 1
                    done, res, _ = _fast_endgame_status(board, rep_kqk)
                if done:
                    if res == 1.0:
                        won = True
                    break
                continue

            prev_pot = _endgame_potential(board, endgame_type)

            if rng.random() < eps:
                chosen_mv = rng.choice(legal_moves)
            else:
                scored = [
                    (_score_afterstate_candidate(board, mv, endgame_type, v_table, shared_kqk_table, seen_afterstates)[0], mv)
                    for mv in legal_moves
                ]
                max_sc = max(sc for sc, _ in scored)
                best_moves = [mv for sc, mv in scored if sc == max_sc]
                chosen_mv = rng.choice(best_moves)

            # Đẩy nước Trắng -> thu được trạng thái afterstate s_after
            board.push(chosen_mv)
            steps += 1

            after_key = encode_afterstate(board)
            rep_count = 0
            if after_key is not None:
                seen_afterstates[after_key] += 1
                rep_count = seen_afterstates[after_key]

            # Với KPK vừa phong Hậu: ghi nhận after_key là None (không lưu vào bảng KPK) nhưng thưởng ngay nếu an toàn
            promoted_safely = False
            if endgame_type == "KPK" and board.pieces_mask(chess.QUEEN, chess.WHITE):
                after_key = None
                promoted_safely = not is_piece_hanging_after_move(board, chess.WHITE)

            done, res, reason = _fast_endgame_status(board, rep_count)

            # Đối thủ Đen đáp trả nếu chưa hết ván
            if not done:
                board.push(_opponent_reply(board, rng))
                steps += 1
                done, res, reason = _fast_endgame_status(board, rep_count)

            if done:
                if res == 1.0:
                    won = True
                reward = _terminal_reward_from_reason(reason, res, win_reward, draw_penalty)
                trajectory.append((after_key, reward))
                break
            elif steps >= max_steps:
                reward = draw_penalty
                trajectory.append((after_key, reward))
                break
            elif promoted_safely:
                # Thưởng ngay khi phong Hậu an toàn để tín hiệu lan truyền ngược cực rõ cho KPK
                promo_reward = 0.85 * win_reward
                trajectory.append((after_key, promo_reward))
                # Tiếp tục vòng lặp để chơi KQK cho tới khi chiếu hết (tính win_rate trung thực)
            else:
                new_pot = _endgame_potential(board, endgame_type)
                reward = step_penalty + (gamma * new_pot - prev_pot)
                if rep_count >= 2:
                    reward -= 0.15
                trajectory.append((after_key, reward))

        # Cập nhật ngược Backward TD(lambda) từ cuối ván về đầu ván
        g_return = 0.0
        next_v = 0.0
        for s_key, r in reversed(trajectory):
            td_target = r + gamma * ((1.0 - td_lambda) * next_v + td_lambda * g_return)
            if s_key is not None:
                old_v = v_table[s_key]
                v_table[s_key] = old_v + alpha * (td_target - old_v)
                next_v = v_table[s_key]
            else:
                next_v = td_target
            g_return = td_target

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
                    "q_states": len(v_table),
                }
            )
            if verbose:
                print(
                    f"\r  [{endgame_type}] Ep {ep:5d}/{episodes} | WinRate: {win_r*100:5.1f}% "
                    f"| AvgPlies: {avg_p:5.1f} | Eps: {eps:.2f} | States: {len(v_table)}",
                    flush=True,
                )
            recent_wins = 0
            recent_steps = 0

    return dict(v_table), curve


def _evaluate_reinforce_greedy(
    policy: ReinforcePolicyNet,
    num_games: int = 100,
    max_steps: int = 100,
    seed: int = 999,
) -> Dict[str, Any]:
    """Đánh giá chính sách greedy của mạng REINFORCE trên KRK."""
    rng = random.Random(seed)
    policy.eval()
    wins = 0
    total_plies = 0

    with torch.no_grad():
        for _ in range(num_games):
            env = ChessEnv(fen=generate_endgame_fen("KRK", rng, curriculum_stage=3))
            board = env.board
            seen_states: Dict[EndgameStateKey, int] = defaultdict(int)
            steps = 0
            while steps < max_steps:
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                feats = torch.tensor(
                    [krk_action_features(board, mv) for mv in legal_moves],
                    dtype=torch.float32,
                )
                logits = policy(feats)
                best_idx = int(torch.argmax(logits).item())
                board.push(legal_moves[best_idx])
                steps += 1
                ak = encode_afterstate(board)
                rep = 0
                if ak is not None:
                    seen_states[ak] += 1
                    rep = seen_states[ak]
                done, res, _ = _fast_endgame_status(board, rep)
                if not done:
                    board.push(_opponent_reply(board, rng))
                    steps += 1
                    done, res, _ = _fast_endgame_status(board, rep)
                if done:
                    if res == 1.0:
                        wins += 1
                    break
            total_plies += steps

    return {
        "eval_games": num_games,
        "greedy_win_rate": round(wins / max(1, num_games), 4),
        "greedy_avg_plies": round(total_plies / max(1, num_games), 2),
    }


def _train_reinforce_krk(
    episodes: int,
    gamma: float,
    step_penalty: float,
    win_reward: float,
    draw_penalty: float,
    max_steps: int,
    seed: int,
    eval_every: int = 500,
    batch_size: int = 16,
    verbose: bool = True,
) -> Tuple[ReinforcePolicyNet, List[Dict[str, Any]]]:
    """
    Huấn luyện Batch Policy Gradient (REINFORCE với 18 đặc trưng & batch-level advantage normalization)
    trên thế cờ tàn KRK. Trả về (policy_net, learning_curve).
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    policy = ReinforcePolicyNet(in_dim=REINFORCE_FEATURE_DIM, hidden=64)
    optimizer = torch.optim.Adam(policy.parameters(), lr=2e-3)

    curve: List[Dict[str, Any]] = []
    recent_wins = 0
    recent_steps = 0
    tick_every = max(1, episodes // 50)

    batch_log_probs: List[torch.Tensor] = []
    batch_entropies: List[torch.Tensor] = []
    batch_returns: List[float] = []
    episodes_in_batch = 0

    for ep in range(1, episodes + 1):
        progress = ep / float(max(1, episodes))
        if progress <= 0.30:
            stage = 1
        elif progress <= 0.60:
            stage = 2
        else:
            stage = 3

        env = ChessEnv(fen=generate_endgame_fen("KRK", rng, curriculum_stage=stage))
        board = env.board
        seen_states: Dict[EndgameStateKey, int] = defaultdict(int)
        ep_log_probs: List[torch.Tensor] = []
        ep_entropies: List[torch.Tensor] = []
        ep_rewards: List[float] = []
        steps = 0
        won = False

        while steps < max_steps:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            prev_pot = _endgame_potential(board, "KRK")
            feats = torch.tensor(
                [krk_action_features(board, mv) for mv in legal_moves],
                dtype=torch.float32,
            )
            logits = policy(feats)
            dist = torch.distributions.Categorical(logits=logits)
            idx = dist.sample()
            ep_log_probs.append(dist.log_prob(idx))
            ep_entropies.append(dist.entropy())

            board.push(legal_moves[int(idx.item())])
            steps += 1
            ak = encode_afterstate(board)
            rep = 0
            if ak is not None:
                seen_states[ak] += 1
                rep = seen_states[ak]
            done, res, reason = _fast_endgame_status(board, rep)
            if not done:
                board.push(_opponent_reply(board, rng))
                steps += 1
                done, res, reason = _fast_endgame_status(board, rep)

            if done:
                if res == 1.0:
                    won = True
                ep_rewards.append(_terminal_reward_from_reason(reason, res, win_reward, draw_penalty))
                break
            elif steps >= max_steps:
                ep_rewards.append(draw_penalty)
                break
            else:
                new_pot = _endgame_potential(board, "KRK")
                r_step = step_penalty + (gamma * new_pot - prev_pot)
                if rep >= 2:
                    r_step -= 0.15
                ep_rewards.append(r_step)

        if ep_log_probs:
            g = 0.0
            ep_ret_rev: List[float] = []
            for r in reversed(ep_rewards):
                g = r + gamma * g
                ep_ret_rev.append(g)
            ep_ret_rev.reverse()

            batch_log_probs.extend(ep_log_probs)
            batch_entropies.extend(ep_entropies)
            batch_returns.extend(ep_ret_rev)
            episodes_in_batch += 1

        if episodes_in_batch >= batch_size or (ep == episodes and batch_log_probs):
            ret_t = torch.tensor(batch_returns, dtype=torch.float32)
            # Chuẩn hóa trên toàn batch (không chuẩn hóa per-episode để tránh thưởng nhầm ván thua)
            adv = ret_t - ret_t.mean()
            if adv.numel() > 1 and float(adv.std().item()) > 1e-6:
                adv = adv / (adv.std() + 1e-8)

            entropy_coef = 0.03 * max(0.2, 1.0 - progress)
            entropy_bonus = torch.stack(batch_entropies).mean()
            loss = -(torch.stack(batch_log_probs) * adv).mean() - entropy_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            batch_log_probs.clear()
            batch_entropies.clear()
            batch_returns.clear()
            episodes_in_batch = 0

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

    policy.eval()
    return policy, curve


def _evaluate_greedy_policy(
    endgame_type: str,
    v_table: Dict[EndgameStateKey, float],
    num_games: int = 100,
    max_steps: int = 100,
    seed: int = 999,
    shared_kqk_table: Optional[Dict[EndgameStateKey, float]] = None,
) -> Dict[str, Any]:
    """Đánh giá chính sách greedy (epsilon = 0) của Afterstate V-table trên thế cờ ngẫu nhiên toàn bàn."""
    rng = random.Random(seed)
    wins = 0
    total_plies = 0

    for _ in range(num_games):
        env = ChessEnv(fen=generate_endgame_fen(endgame_type, rng, curriculum_stage=3))
        board = env.board
        seen_afterstates: Dict[EndgameStateKey, int] = defaultdict(int)
        steps = 0
        while steps < max_steps:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            active_type = (
                "KQK"
                if (endgame_type == "KPK" and shared_kqk_table is not None and board.pieces_mask(chess.QUEEN, chess.WHITE))
                else endgame_type
            )
            active_table = shared_kqk_table if (active_type == "KQK" and shared_kqk_table is not None) else v_table

            best_mv = legal_moves[0]
            best_v = -float("inf")
            for mv in legal_moves:
                sc, _ = _score_afterstate_candidate(board, mv, active_type, active_table, shared_kqk_table, seen_afterstates)
                if sc > best_v:
                    best_v = sc
                    best_mv = mv

            board.push(best_mv)
            steps += 1
            ak = encode_afterstate(board)
            rep = 0
            if ak is not None:
                seen_afterstates[ak] += 1
                rep = seen_afterstates[ak]
            done, res, _ = _fast_endgame_status(board, rep)
            if not done:
                board.push(_opponent_reply(board, rng))
                steps += 1
                done, res, _ = _fast_endgame_status(board, rep)
            if done:
                if res == 1.0:
                    wins += 1
                break
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
    """Thực thi huấn luyện RL Cấp 6 (Afterstate TD(lambda) + Batch REINFORCE) và lưu checkpoint + Learning Curve."""
    t0 = time.perf_counter()
    cfg = load_yaml_config(config_path)["rl_level6"]

    episodes = episodes_override if episodes_override is not None else int(cfg.get("episodes", 20000))
    gamma = float(cfg.get("gamma", 0.99))
    alpha_start = max(0.30, float(cfg.get("learning_rate", 0.1)))
    alpha_end = 0.08
    td_lambda = 0.70
    step_penalty = -0.005
    win_reward = float(cfg.get("win_reward", 1.0))
    draw_penalty = float(cfg.get("draw_penalty", -0.2))
    eps_start = float(cfg.get("epsilon_start", 1.0))
    eps_end = float(cfg.get("epsilon_end", 0.05))
    max_steps = 100
    eval_every = max(100, episodes // 20)

    v_tables: Dict[str, Dict[EndgameStateKey, float]] = {}
    q_curves: Dict[str, List[Dict[str, Any]]] = {}
    stage_summary: Dict[str, Any] = {}

    if verbose:
        print(
            f"Bắt đầu huấn luyện Cấp 6 RL: Afterstate TD(lambda) {list(endgames)} "
            f"({episodes} ep/thế) + Batch REINFORCE KRK ({reinforce_episodes} ep)",
            flush=True,
        )

    for eg in endgames:
        t_stage = time.perf_counter()
        shared_kqk = v_tables.get("KQK") if eg == "KPK" else None
        table, curve = _train_afterstate_td_single_endgame(
            endgame_type=eg,
            episodes=episodes,
            gamma=gamma,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            td_lambda=td_lambda,
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
            v_table=table,
            num_games=eval_games,
            max_steps=max_steps,
            seed=seed + 1000,
            shared_kqk_table=shared_kqk,
        )
        v_tables[eg] = table
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
    reinforce_net, reinforce_curve = _train_reinforce_krk(
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
    reinforce_eval = _evaluate_reinforce_greedy(
        policy=reinforce_net,
        num_games=eval_games,
        max_steps=max_steps,
        seed=seed + 2000,
    )
    stage_summary["reinforce_KRK"] = {
        "elapsed_sec": round(time.perf_counter() - t_reinf, 2),
        "final_train_win_rate": reinforce_curve[-1]["win_rate"] if reinforce_curve else 0.0,
        "final_train_avg_plies": reinforce_curve[-1]["avg_plies"] if reinforce_curve else 0.0,
        **reinforce_eval,
    }

    q_out = Path(output_qtable)
    q_out.parent.mkdir(parents=True, exist_ok=True)
    with open(q_out, "wb") as f:
        pickle.dump(
            {
                "algorithm": "afterstate_td_lambda_and_batch_reinforce",
                "v_tables": v_tables,
                "endgames": list(endgames),
                "reinforce_state_dict": reinforce_net.state_dict(),
                "reinforce_in_dim": REINFORCE_FEATURE_DIM,
                "reinforce_hidden": 64,
            },
            f,
        )

    metrics_payload: Dict[str, Any] = {
        "experiment": "Level6_RL_Endgames",
        "algorithm": "afterstate_td_lambda_and_batch_reinforce",
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
            "learning_rate": alpha_start,
            "learning_rate_end": alpha_end,
            "td_lambda": td_lambda,
            "reinforce_lr": 0.002,
            "reinforce_batch_size": 16,
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
    parser = argparse.ArgumentParser(description="Huấn luyện RL Cấp 6 (Afterstate TD(lambda) KQK/KRK/KPK + Batch REINFORCE KRK).")
    parser.add_argument("--config", default="configs/search_params.yaml", help="Đường dẫn tệp YAML.")
    parser.add_argument("--episodes", type=int, default=None, help="Số episode Afterstate TD cho mỗi thế cờ tàn.")
    parser.add_argument("--reinforce-episodes", type=int, default=2000, help="Số episode Batch REINFORCE trên KRK.")
    args = parser.parse_args()

    res = train_krk_rl(
        config_path=args.config,
        episodes_override=args.episodes,
        reinforce_episodes=args.reinforce_episodes,
    )
    print(f"Huấn luyện Cấp 6 hoàn tất trong {res['elapsed_sec']}s.")
    for eg in res["endgames"]:
        st = res["stage_summary"][f"tabular_q_{eg}"]
        print(
            f"  - Afterstate TD [{eg}]: TrainWR={st['final_train_win_rate']*100:.1f}% | "
            f"GreedyWR={st['greedy_win_rate']*100:.1f}% | AvgPlies={st['greedy_avg_plies']} | States={st['q_states']}"
        )
    r_st = res["stage_summary"]["reinforce_KRK"]
    print(
        f"  - Batch REINFORCE [KRK]: TrainWR={r_st['final_train_win_rate']*100:.1f}% | "
        f"GreedyWR={r_st['greedy_win_rate']*100:.1f}% | AvgPlies={r_st['greedy_avg_plies']}"
    )


if __name__ == "__main__":
    main()
