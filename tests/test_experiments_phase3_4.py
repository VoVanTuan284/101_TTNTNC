"""
tests/test_experiments_phase3_4.py
Kiểm thử Giai đoạn 3 & 4 của phần thực nghiệm:
- Agent registry (create_agent)
- play_single_match & export_game_to_pgn / append_match_jsonl
- run_round_robin_tournament (smoke 2 khai cuộc = 4 ván)
- run_ablation_mcts (smoke Phần 2A Cấp 4 & graceful fallback khi TV2 chưa xong Cấp 5/7)
- generate_figures_from_logs
"""

import json
from pathlib import Path
import chess

from chess_ai.agents.registry import create_agent, list_full_board_agents
from chess_ai.arena.match import play_single_match
from chess_ai.arena.pgn_exporter import append_match_jsonl, export_game_to_pgn
from chess_ai.arena.tournament import run_round_robin_tournament
from chess_ai.experiments.ablation_mcts import run_ablation_mcts
from scripts.reproduce_experiment import generate_figures_from_logs


def test_registry_create_classical_agents() -> None:
    full_agents = list_full_board_agents()
    assert "level1" in full_agents
    assert "level6" not in full_agents
    for aid in ("level1", "level2", "level3", "level4"):
        agent = create_agent(aid)
        assert agent is not None


def test_single_match_and_pgn_export(tmp_path: Path) -> None:
    a1 = create_agent("level1", seed=42)
    a2 = create_agent("level2", seed=42)
    rec = play_single_match(a1, a2, max_plies=20, seed=42)
    assert rec["result"] in ("1-0", "0-1", "1/2-1/2")
    assert rec["total_plies"] <= 20
    assert len(rec["moves_uci"]) == rec["total_plies"]

    moves = [chess.Move.from_uci(u) for u in rec["moves_uci"]]
    pgn = export_game_to_pgn(moves, {"White": rec["white"], "Black": rec["black"], "Result": rec["result"]})
    assert "[White " in pgn

    jsonl_file = tmp_path / "matches.jsonl"
    append_match_jsonl(str(jsonl_file), rec)
    assert jsonl_file.exists()


def test_round_robin_smoke(tmp_path: Path) -> None:
    a1 = create_agent("level1", seed=42)
    a2 = create_agent("level2", seed=42)
    report = run_round_robin_tournament(
        agents=[a1, a2],
        budget_mode="workload_control",
        num_openings=2,
        output_dir=str(tmp_path),
    )
    assert report["games_per_pair"] == 4
    assert len(report["matchups"]) == 1
    assert report["matchups"][0]["newly_played_games"] == 4
    assert report["matchups"][0]["reused_games"] == 0
    assert (tmp_path / "matches.jsonl").exists()
    assert (tmp_path / "games.pgn").exists()
    assert (tmp_path / "tournament_summary.json").exists()

    # Kiểm tra chạy tiếp nối (resume=True): chạy lại cùng cặp a1 vs a2 phải tái sử dụng 100% (0 ván mới)
    report_resume = run_round_robin_tournament(
        agents=[a1, a2],
        budget_mode="workload_control",
        num_openings=2,
        output_dir=str(tmp_path),
        resume=True,
    )
    assert report_resume["matchups"][0]["newly_played_games"] == 0
    assert report_resume["matchups"][0]["reused_games"] == 4
    lines = (tmp_path / "matches.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4  # Không bị nhân đôi số dòng trong matches.jsonl


def test_ablation_mcts_smoke(tmp_path: Path) -> None:
    out_json = tmp_path / "ablation2_smoke.json"
    res = run_ablation_mcts(
        output_path=str(out_json),
        sim_budgets=[5, 10],
        num_openings_2a=1,
        num_openings_2b=1,
        include_puct=True,
    )
    assert out_json.exists()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(loaded["part_2a_uct_level4"]) == 2
    assert loaded["part_2a_uct_level4"][0]["games"] == 2
    assert res["part_2a_puct_level7"]["status"] in ("completed", "pending_tv2_level7")


def test_reproduce_figures_from_ablation1(tmp_path: Path) -> None:
    figs = generate_figures_from_logs(output_dir=str(tmp_path))
    assert "ablation1_progressive.png" in figs
    assert (tmp_path / "ablation1_progressive.png").exists()
