"""
tests/test_arena_metrics.py
Nhóm 4 (TC52 - TC61): Kiểm thử động cơ giải đấu Arena trên 50 thế cờ khai cuộc cân bằng,
tính toán điểm S, Khoảng tin cậy Wilson Score 95%, Paired-Opening Bootstrap 95%,
xuất PGN/JSONL và kiểm định tính hợp lệ của bộ 50 thế cờ WAC (resources/benchmark_positions.epd).
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

import pytest

from chess_ai.arena.metrics import compute_move_agreement_rate


def test_compute_move_agreement_rate() -> None:
    """Kiểm tra tính toán Move Agreement Rate cho RQ4."""
    assert compute_move_agreement_rate(["e2e4", "d2d4", "g1f3", "c2c4"], ["e2e4", "e2e3", "g1f3", "c2c3"]) == 0.5
    assert compute_move_agreement_rate([], []) == 0.0
    with pytest.raises(ValueError):
        compute_move_agreement_rate(["e2e4"], ["e2e4", "d2d4"])

