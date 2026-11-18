"""
tests/test_experiments_phase1_2.py
Kiểm thử Giai đoạn 1 & 2 của phần thực nghiệm:
- load_epd_file trên benchmark_positions.epd (WAC-50) và openings_50.epd (50 khai cuộc)
- metrics.py (Normalized Score, Wilson 95% CI, Paired-Opening Bootstrap 95% CI, Latency, Style)
- ablation_search.py (Ablation 1 trên mẫu nhỏ WAC)
"""

import json
from pathlib import Path
import pytest

from chess_ai.arena.metrics import (
    compute_latency_stats,
    compute_normalized_score,
    compute_style_metrics,
    paired_bootstrap_confidence_interval,
    wilson_confidence_interval,
)
from chess_ai.experiments.ablation_search import run_ablation_search
from chess_ai.utils.epd import load_epd_file


def test_load_epd_wac50_and_openings50() -> None:
    wac_records = load_epd_file("resources/benchmark_positions.epd")
    assert len(wac_records) == 50
    assert wac_records[0].record_id == "WAC.001"
    assert len(wac_records[0].best_moves) >= 1

    opening_records = load_epd_file("resources/openings_50.epd")
    assert len(opening_records) == 50
    assert opening_records[0].record_id == "OPEN.001"
    assert opening_records[-1].record_id == "OPEN.050"


def test_metrics_wilson_and_bootstrap() -> None:
    score = compute_normalized_score(wins=60, draws=20, losses=20)
    assert score == pytest.approx(0.70)

    ci_low, ci_high = wilson_confidence_interval(score, 100)
    assert 0.0 <= ci_low < score < ci_high <= 1.0

    # Biên 0 và 1 vẫn nằm hợp lệ trong [0, 1]
    assert wilson_confidence_interval(0.0, 100)[0] == pytest.approx(0.0)
    assert wilson_confidence_interval(1.0, 100)[1] == pytest.approx(1.0)

    pair_scores = [0.75] * 25 + [0.5] * 25
    b_low, b_high = paired_bootstrap_confidence_interval(pair_scores, num_resamples=1000, seed=42)
    assert 0.5 <= b_low <= 0.625 <= b_high <= 0.75


def test_latency_and_style_metrics() -> None:
    lat = compute_latency_stats([10.0, 20.0, 30.0, 40.0, 100.0])
    assert lat["avg_ms"] == pytest.approx(40.0)
    assert lat["max_ms"] == pytest.approx(100.0)
    assert lat["p95_ms"] >= 40.0

    style = compute_style_metrics(
        [
            {"total_plies": 60, "termination": "checkmate", "result": "1-0", "moves_uci": ["e2e4", "e1g1"]},
            {"total_plies": 80, "termination": "stalemate", "result": "1/2-1/2", "moves_uci": ["d2d4"]},
        ]
    )
    assert style["avg_game_length_plies"] == pytest.approx(70.0)
    assert style["checkmate_rate"] == pytest.approx(0.5)
    assert style["draw_rate"] == pytest.approx(0.5)
    assert style["castling_rate"] == pytest.approx(0.5)


def test_ablation_search_smoke(tmp_path: Path) -> None:
    out_json = tmp_path / "ablation1_smoke.json"
    res = run_ablation_search(
        epd_path="resources/benchmark_positions.epd",
        output_path=str(out_json),
        max_positions=3,
        base_depth=2,
    )
    assert out_json.exists()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["num_positions"] == 3
    assert len(loaded["part_1a_progressive_configs"]) == 5
    assert len(loaded["part_1b_depth_sensitivity"]) == 3
    assert res["part_1a_progressive_configs"][-1]["name"] == "Config_D"
