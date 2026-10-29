"""
chess_ai/arena/metrics.py
Module tính toán thống kê khoa học:
- Điểm số chuẩn hóa S = (W + 0.5 * D) / N.
- Khoảng tin cậy Wilson Score 95%.
- Khoảng tin cậy Paired-Opening Bootstrap 95% (10,000 resamples).
- Độ trễ p95 (ms), tốc độ NPS / SPS, và các chỉ số phong cách thi đấu (RQ4).
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

from typing import Sequence, Tuple


def compute_normalized_score(wins: int, draws: int, losses: int) -> float:
    """Tính tỷ lệ điểm chuẩn hóa S = (W + 0.5 * D) / (W + D + L)."""
    raise NotImplementedError


def wilson_confidence_interval(
    score: float,
    total_games: int,
    z: float = 1.96,
) -> Tuple[float, float]:
    """Tính khoảng tin cậy Wilson Score 95% [ci_low, ci_high] trong [0.0, 1.0]."""
    raise NotImplementedError


def paired_bootstrap_confidence_interval(
    pair_scores: Sequence[float],
    num_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Tính khoảng tin cậy Bootstrap 95% trên 50 cặp ván khai cuộc đối xứng."""
    raise NotImplementedError

