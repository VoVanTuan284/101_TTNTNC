"""
chess_ai/arena/metrics.py
Module tính toán thống kê khoa học:
- Điểm số chuẩn hóa S = (W + 0.5 * D) / N.
- Khoảng tin cậy Wilson Score 95%.
- Khoảng tin cậy Paired-Opening Bootstrap 95% (10,000 resamples).
- Độ trễ p95 (ms), tốc độ NPS / SPS, và các chỉ số phong cách thi đấu (RQ4).
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

import math
from typing import Any, Dict, Sequence, Tuple
import numpy as np


def compute_normalized_score(wins: int, draws: int, losses: int) -> float:
    """Tính tỷ lệ điểm chuẩn hóa S = (W + 0.5 * D) / (W + D + L)."""
    total = wins + draws + losses
    if total <= 0:
        raise ValueError("Tổng số ván đấu (W + D + L) phải lớn hơn 0.")
    return float((wins + 0.5 * draws) / total)


def wilson_confidence_interval(
    score: float,
    total_games: int,
    z: float = 1.96,
) -> Tuple[float, float]:
    """Tính khoảng tin cậy Wilson Score 95% [ci_low, ci_high] trong [0.0, 1.0]."""
    if total_games <= 0:
        raise ValueError("total_games phải lớn hơn 0.")
    p = min(1.0, max(0.0, float(score)))
    n = float(total_games)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p)) / n + z2 / (4.0 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def paired_bootstrap_confidence_interval(
    pair_scores: Sequence[float],
    num_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Tính khoảng tin cậy Bootstrap 95% trên 50 cặp ván khai cuộc đối xứng."""
    arr = np.asarray(pair_scores, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Danh sách pair_scores không được rỗng.")
    if num_resamples <= 0:
        raise ValueError("num_resamples phải lớn hơn 0.")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(num_resamples, arr.size))
    sample_means = arr[indices].mean(axis=1)

    alpha = 1.0 - confidence
    low, high = np.quantile(sample_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return max(0.0, float(low)), min(1.0, float(high))


def compute_latency_stats(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """Tính độ trễ trung bình, p95 và tối đa (ms)."""
    if not latencies_ms:
        return {"avg_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "avg_ms": float(np.mean(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr)),
    }


def compute_style_metrics(match_records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Tính các chỉ số định lượng phong cách thi đấu (RQ4) từ danh sách log trận đấu."""
    if not match_records:
        return {
            "avg_game_length_plies": 0.0,
            "checkmate_rate": 0.0,
            "draw_rate": 0.0,
            "castling_rate": 0.0,
        }

    total = len(match_records)
    total_plies = sum(int(m.get("total_plies", 0)) for m in match_records)
    checkmates = sum(1 for m in match_records if m.get("termination") == "checkmate")
    draws = sum(1 for m in match_records if m.get("result") == "1/2-1/2")

    castling_games = 0
    for m in match_records:
        moves = m.get("moves_uci", [])[:30]
        if any(uci in ("e1g1", "e1c1", "e8g8", "e8c8") for uci in moves):
            castling_games += 1

    return {
        "avg_game_length_plies": total_plies / total,
        "checkmate_rate": checkmates / total,
        "draw_rate": draws / total,
        "castling_rate": castling_games / total,
    }
