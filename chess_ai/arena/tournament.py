"""
chess_ai/arena/tournament.py
Động cơ điều phối giải đấu Round-Robin 100 ván/cặp (50 thế cờ khai cuộc cân bằng x 2 màu)
hỗ trợ hai chế độ ngân sách: Workload-Control và Time-Control.
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import chess

from chess_ai.agents.base_agent import BaseAgent
from chess_ai.agents.registry import create_agent, list_full_board_agents
from chess_ai.arena.match import play_single_match
from chess_ai.arena.metrics import (
    compute_normalized_score,
    paired_bootstrap_confidence_interval,
    wilson_confidence_interval,
)
from chess_ai.arena.pgn_exporter import (
    append_match_jsonl,
    append_pgn_file,
    export_game_to_pgn,
)
from chess_ai.config import load_yaml_config
from chess_ai.utils.epd import EPDRecord, load_epd_file


def _default_workload_for_agent(agent: BaseAgent, search_cfg: Dict[str, Any]) -> Optional[int]:
    """Lấy ngân sách mặc định (nodes hoặc simulations) từ search_params.yaml theo tên tác tử."""
    name = agent.name.lower()
    if "alphabeta" in name or "level3" in name:
        return int(search_cfg["alpha_beta"].get("max_nodes", 10000))
    if "mcts_uct" in name or "level4" in name:
        return int(search_cfg["vanilla_mcts"].get("default_simulations", 400))
    if "guidedmcts" in name or "alphazero" in name or "level7" in name or "level8" in name:
        return int(search_cfg["guided_mcts"].get("default_simulations", 400))
    return None


def _run_pair_worker(task: Tuple[Any, ...]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Worker chạy một cặp ván đổi màu trên cùng 1 thế cờ khai cuộc (phục vụ multiprocessing)."""
    (
        factory_a,
        factory_b,
        opening_id,
        opening_fen,
        seed,
        time_limit_sec,
        wl_a,
        wl_b,
        max_plies,
        budget_mode,
    ) = task
    agent_a = factory_a()
    agent_b = factory_b()

    g1 = play_single_match(
        white_agent=agent_a,
        black_agent=agent_b,
        initial_fen=opening_fen,
        time_limit_sec=time_limit_sec,
        max_plies=max_plies,
        seed=seed,
        white_workload=wl_a,
        black_workload=wl_b,
    )
    g1["opening_id"] = opening_id
    g1["budget_mode"] = budget_mode
    g1["match_id"] = f"{agent_a.name}_vs_{agent_b.name}_{opening_id}_w"

    g2 = play_single_match(
        white_agent=agent_b,
        black_agent=agent_a,
        initial_fen=opening_fen,
        time_limit_sec=time_limit_sec,
        max_plies=max_plies,
        seed=seed,
        white_workload=wl_b,
        black_workload=wl_a,
    )
    g2["opening_id"] = opening_id
    g2["budget_mode"] = budget_mode
    g2["match_id"] = f"{agent_a.name}_vs_{agent_b.name}_{opening_id}_b"
    return g1, g2


def run_paired_matchup(
    agent_a: BaseAgent,
    agent_b: BaseAgent,
    openings: List[EPDRecord],
    seeds: List[int],
    budget_mode: str = "workload_control",
    time_limit_sec: Optional[float] = None,
    workload_a: Optional[int] = None,
    workload_b: Optional[int] = None,
    max_plies: int = 400,
    jsonl_path: Optional[str] = None,
    pgn_path: Optional[str] = None,
    bootstrap_resamples: int = 10000,
    workers: int = 1,
    factory_a: Optional[Callable[[], BaseAgent]] = None,
    factory_b: Optional[Callable[[], BaseAgent]] = None,
) -> Dict[str, Any]:
    """
    Cho cặp (agent_a, agent_b) đấu đối xứng qua toàn bộ danh sách thế cờ khai cuộc (2 ván/khai cuộc).
    Trả về thống kê W/D/L, điểm S, Wilson 95% CI và Paired-Opening Bootstrap 95% CI cho agent_a.
    """
    pair_results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    if workers > 1 and budget_mode == "workload_control" and factory_a is not None and factory_b is not None:
        tasks = []
        for idx, rec in enumerate(openings):
            seed = seeds[(idx // 5) % len(seeds)]
            tasks.append(
                (
                    factory_a,
                    factory_b,
                    rec.record_id,
                    rec.fen,
                    seed,
                    time_limit_sec,
                    workload_a,
                    workload_b,
                    max_plies,
                    budget_mode,
                )
            )
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pair_results = list(pool.map(_run_pair_worker, tasks))
    else:
        for idx, rec in enumerate(openings):
            seed = seeds[(idx // 5) % len(seeds)]
            g1 = play_single_match(
                white_agent=agent_a,
                black_agent=agent_b,
                initial_fen=rec.fen,
                time_limit_sec=time_limit_sec,
                max_plies=max_plies,
                seed=seed,
                white_workload=workload_a,
                black_workload=workload_b,
            )
            g1["opening_id"] = rec.record_id
            g1["budget_mode"] = budget_mode
            g1["match_id"] = f"{agent_a.name}_vs_{agent_b.name}_{rec.record_id}_w"

            g2 = play_single_match(
                white_agent=agent_b,
                black_agent=agent_a,
                initial_fen=rec.fen,
                time_limit_sec=time_limit_sec,
                max_plies=max_plies,
                seed=seed,
                white_workload=workload_b,
                black_workload=workload_a,
            )
            g2["opening_id"] = rec.record_id
            g2["budget_mode"] = budget_mode
            g2["match_id"] = f"{agent_a.name}_vs_{agent_b.name}_{rec.record_id}_b"
            pair_results.append((g1, g2))

    wins = draws = losses = 0
    white_pts = 0.0
    black_pts = 0.0
    pair_scores: List[float] = []
    a_latencies: List[float] = []
    b_latencies: List[float] = []

    for g1, g2 in pair_results:
        for g, a_is_white in ((g1, True), (g2, False)):
            if jsonl_path:
                append_match_jsonl(jsonl_path, g)
            if pgn_path:
                moves = [chess.Move.from_uci(u) for u in g["moves_uci"]]
                pgn_txt = export_game_to_pgn(
                    moves=moves,
                    headers={
                        "Event": f"101_TTNTNC Arena ({budget_mode})",
                        "Site": g["opening_id"],
                        "White": g["white"],
                        "Black": g["black"],
                        "Result": g["result"],
                    },
                    initial_fen=g["opening_fen"],
                )
                append_pgn_file(pgn_path, pgn_txt)

            if a_is_white:
                a_latencies.append(g["white_avg_latency_ms"])
                b_latencies.append(g["black_avg_latency_ms"])
            else:
                a_latencies.append(g["black_avg_latency_ms"])
                b_latencies.append(g["white_avg_latency_ms"])

        s_a_w = float(g1["white_score"])
        s_a_b = float(g2["black_score"])
        white_pts += s_a_w
        black_pts += s_a_b
        pair_scores.append(0.5 * (s_a_w + s_a_b))

        for s in (s_a_w, s_a_b):
            if s == 1.0:
                wins += 1
            elif s == 0.5:
                draws += 1
            else:
                losses += 1

    total_games = wins + draws + losses
    score = compute_normalized_score(wins, draws, losses)
    w_ci = wilson_confidence_interval(score, total_games)
    b_ci = paired_bootstrap_confidence_interval(pair_scores, num_resamples=bootstrap_resamples)
    num_pairs = len(openings)

    return {
        "agent_a": agent_a.name,
        "agent_b": agent_b.name,
        "budget_mode": budget_mode,
        "games": total_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": round(score, 4),
        "white_score": round(white_pts / num_pairs, 4) if num_pairs > 0 else 0.0,
        "black_score": round(black_pts / num_pairs, 4) if num_pairs > 0 else 0.0,
        "wilson_ci_95": [round(w_ci[0], 4), round(w_ci[1], 4)],
        "bootstrap_ci_95": [round(b_ci[0], 4), round(b_ci[1], 4)],
        "agent_a_avg_latency_ms": round(sum(a_latencies) / len(a_latencies), 3) if a_latencies else 0.0,
        "agent_b_avg_latency_ms": round(sum(b_latencies) / len(b_latencies), 3) if b_latencies else 0.0,
    }


def run_round_robin_tournament(
    agents: List[BaseAgent],
    config_path: str = "configs/default_config.yaml",
    budget_mode: str = "workload_control",
    num_openings: Optional[int] = None,
    output_dir: Optional[str] = None,
    workers: int = 1,
) -> Dict[str, Any]:
    """Vận hành giải đấu đối kháng giữa danh sách tác tử và lưu vết ra results/round_robin/."""
    cfg = load_yaml_config(config_path)
    search_cfg = load_yaml_config("search_params.yaml")
    arena_cfg = cfg["arena"]
    paths_cfg = cfg["paths"]

    openings = load_epd_file(paths_cfg["openings_epd"])
    if num_openings is not None:
        openings = openings[:num_openings]
    else:
        openings = openings[: int(arena_cfg.get("num_openings", 50))]

    seeds: List[int] = [int(s) for s in cfg.get("agent_seeds", [42])]
    max_plies = int(arena_cfg.get("max_plies", 400))
    bootstrap_resamples = int(arena_cfg.get("bootstrap_resamples", 10000))

    time_limit_sec: Optional[float] = None
    if budget_mode == "time_control":
        time_limit_sec = float(arena_cfg.get("time_limit_sec", 1.0))

    out_base = Path(output_dir if output_dir is not None else paths_cfg["round_robin_dir"])
    out_base.mkdir(parents=True, exist_ok=True)
    jsonl_path = str(out_base / "matches.jsonl")
    pgn_path = str(out_base / "games.pgn")

    # Xóa log cũ nếu chạy giải mới tại thư mục đích
    for p in (Path(jsonl_path), Path(pgn_path)):
        if p.exists():
            p.unlink()

    matchups: List[Dict[str, Any]] = []
    for agent_a, agent_b in combinations(agents, 2):
        wl_a = _default_workload_for_agent(agent_a, search_cfg) if budget_mode == "workload_control" else None
        wl_b = _default_workload_for_agent(agent_b, search_cfg) if budget_mode == "workload_control" else None
        summary = run_paired_matchup(
            agent_a=agent_a,
            agent_b=agent_b,
            openings=openings,
            seeds=seeds,
            budget_mode=budget_mode,
            time_limit_sec=time_limit_sec,
            workload_a=wl_a,
            workload_b=wl_b,
            max_plies=max_plies,
            jsonl_path=jsonl_path,
            pgn_path=pgn_path,
            bootstrap_resamples=bootstrap_resamples,
            workers=workers,
        )
        matchups.append(summary)

    report = {
        "tournament": "101_TTNTNC_Round_Robin",
        "budget_mode": budget_mode,
        "num_openings": len(openings),
        "games_per_pair": len(openings) * 2,
        "agents": [a.name for a in agents],
        "matchups": matchups,
    }
    summary_path = out_base / "tournament_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy giải đấu Round-Robin Arena cho các cấp độ tác tử.")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Đường dẫn cấu hình YAML.")
    parser.add_argument(
        "--budget-mode",
        default="workload_control",
        choices=["workload_control", "time_control"],
        help="Chế độ kiểm soát ngân sách.",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["level1", "level2", "level3", "level4"],
        help="Danh sách mã tác tử tham gia (ví dụ: level1 level2 level3 level4).",
    )
    parser.add_argument("--num-openings", type=int, default=None, help="Giới hạn số khai cuộc để chạy nhanh.")
    parser.add_argument("--output-dir", default=None, help="Thư mục lưu kết quả giải đấu.")
    parser.add_argument("--workers", type=int, default=1, help="Số tiến trình song song (chỉ dùng cho workload_control).")
    args = parser.parse_args()

    valid_ids = set(list_full_board_agents())
    selected_agents: List[BaseAgent] = []
    for aid in args.agents:
        if aid not in valid_ids:
            raise ValueError(f"Tác tử '{aid}' không hỗ trợ thi đấu giải 8x8.")
        selected_agents.append(create_agent(aid))

    report = run_round_robin_tournament(
        agents=selected_agents,
        config_path=args.config,
        budget_mode=args.budget_mode,
        num_openings=args.num_openings,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(f"Giải đấu hoàn tất ({report['games_per_pair']} ván/cặp, chế độ {report['budget_mode']}).")
    for m in report["matchups"]:
        print(
            f"{m['agent_a']} vs {m['agent_b']}: "
            f"W/D/L={m['wins']}/{m['draws']}/{m['losses']} | "
            f"S={m['score']:.3f} | Wilson95={m['wilson_ci_95']} | Boot95={m['bootstrap_ci_95']}"
        )


if __name__ == "__main__":
    main()
