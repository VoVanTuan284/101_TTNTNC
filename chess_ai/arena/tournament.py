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


def _load_existing_matches_index(jsonl_path: Optional[str]) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    """
    Đọc tệp matches.jsonl hiện có và lập chỉ mục theo khóa:
    (white_agent_name, black_agent_name, opening_id, budget_mode) -> bản_ghi_ván_đấu.
    Giúp chạy tiếp nối (resume) mà không bao giờ phải đánh lại các ván đã hoàn tất.
    """
    index: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    if not jsonl_path:
        return index
    p = Path(jsonl_path)
    if not p.exists():
        return index
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = (
                        str(rec.get("white", "")),
                        str(rec.get("black", "")),
                        str(rec.get("opening_id", "")),
                        str(rec.get("budget_mode", "")),
                    )
                    if all(key):
                        index[key] = rec
                except Exception:
                    continue
    except Exception:
        pass
    return index


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
    existing_index: Optional[Dict[Tuple[str, str, str, str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Cho cặp (agent_a, agent_b) đấu đối xứng qua toàn bộ danh sách thế cờ khai cuộc (2 ván/khai cuộc).
    Nếu existing_index (hoặc jsonl_path) đã có kết quả của cặp ván ở khai cuộc nào thì tự động tái sử dụng,
    chỉ chạy các khai cuộc còn thiếu và ghi nối tiếp (append) những ván mới vào jsonl_path / pgn_path.
    """
    cache_idx = existing_index if existing_index is not None else _load_existing_matches_index(jsonl_path)

    cached_by_pos: Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    missing_indices: List[int] = []

    for idx, rec in enumerate(openings):
        key_w = (agent_a.name, agent_b.name, rec.record_id, budget_mode)
        key_b = (agent_b.name, agent_a.name, rec.record_id, budget_mode)
        if key_w in cache_idx and key_b in cache_idx:
            cached_by_pos[idx] = (cache_idx[key_w], cache_idx[key_b])
        else:
            missing_indices.append(idx)

    newly_played_by_pos: Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]] = {}

    if missing_indices:
        if workers > 1 and budget_mode == "workload_control" and factory_a is not None and factory_b is not None:
            tasks = []
            for idx in missing_indices:
                rec = openings[idx]
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
                computed = list(pool.map(_run_pair_worker, tasks))
            for idx, pair in zip(missing_indices, computed):
                newly_played_by_pos[idx] = pair
        else:
            for idx in missing_indices:
                rec = openings[idx]
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
                newly_played_by_pos[idx] = (g1, g2)

    wins = draws = losses = 0
    white_pts = 0.0
    black_pts = 0.0
    pair_scores: List[float] = []
    a_latencies: List[float] = []
    b_latencies: List[float] = []

    for idx in range(len(openings)):
        is_new = idx in newly_played_by_pos
        g1, g2 = newly_played_by_pos[idx] if is_new else cached_by_pos[idx]
        for g, a_is_white in ((g1, True), (g2, False)):
            if is_new:
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
                key_g = (
                    str(g.get("white", "")),
                    str(g.get("black", "")),
                    str(g.get("opening_id", "")),
                    str(g.get("budget_mode", "")),
                )
                cache_idx[key_g] = g

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
        "reused_games": len(cached_by_pos) * 2,
        "newly_played_games": len(newly_played_by_pos) * 2,
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


def _find_agent_id_for_instance(agent: BaseAgent) -> Optional[str]:
    for aid in list_full_board_agents():
        try:
            probe = create_agent(aid)
            if probe.name == agent.name:
                return aid
        except Exception:
            continue
    return None


def run_round_robin_tournament(
    agents: List[BaseAgent],
    config_path: str = "configs/default_config.yaml",
    budget_mode: str = "workload_control",
    num_openings: Optional[int] = None,
    output_dir: Optional[str] = None,
    workers: int = 1,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Vận hành giải đấu đối kháng giữa danh sách tác tử và lưu vết ra results/round_robin/.
    - Khi resume=True (mặc định): đọc matches.jsonl và tournament_summary.json hiện có,
      tự động bỏ qua mọi ván/cặp đấu đã hoàn tất và chỉ đánh nối tiếp các cặp/ván còn thiếu,
      đồng thời gộp đầy đủ kết quả cũ + mới vào tournament_summary.json.
    - Khi resume=False: xóa sạch matches.jsonl và games.pgn để chạy lại từ đầu.
    """
    from functools import partial

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
    summary_path = out_base / "tournament_summary.json"

    if not resume:
        for p in (Path(jsonl_path), Path(pgn_path), summary_path):
            if p.exists():
                p.unlink()

    existing_index = _load_existing_matches_index(jsonl_path) if resume else {}

    # Nạp các cặp đấu đã lưu từ tournament_summary.json trước đó (để không làm mất các cặp không nằm trong danh sách --agents lần chạy này)
    merged_matchups_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    all_agent_names: List[str] = []
    if resume and summary_path.exists():
        try:
            prev_report = json.loads(summary_path.read_text(encoding="utf-8"))
            if prev_report.get("budget_mode") == budget_mode and prev_report.get("num_openings") == len(openings):
                for prev_a in prev_report.get("agents", []):
                    if prev_a not in all_agent_names:
                        all_agent_names.append(prev_a)
                for pm in prev_report.get("matchups", []):
                    pair_key = (str(pm.get("agent_a", "")), str(pm.get("agent_b", "")))
                    merged_matchups_map[pair_key] = pm
        except Exception:
            pass

    for a in agents:
        if a.name not in all_agent_names:
            all_agent_names.append(a.name)

    for agent_a, agent_b in combinations(agents, 2):
        wl_a = _default_workload_for_agent(agent_a, search_cfg) if budget_mode == "workload_control" else None
        wl_b = _default_workload_for_agent(agent_b, search_cfg) if budget_mode == "workload_control" else None
        aid_a = _find_agent_id_for_instance(agent_a)
        aid_b = _find_agent_id_for_instance(agent_b)
        fac_a = partial(create_agent, aid_a) if aid_a is not None else None
        fac_b = partial(create_agent, aid_b) if aid_b is not None else None

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
            factory_a=fac_a,
            factory_b=fac_b,
            existing_index=existing_index,
        )
        # Xóa khóa ngược chiều nếu có để tránh trùng cặp (A, B) và (B, A)
        rev_key = (agent_b.name, agent_a.name)
        if rev_key in merged_matchups_map:
            del merged_matchups_map[rev_key]
        merged_matchups_map[(agent_a.name, agent_b.name)] = summary

    matchups = list(merged_matchups_map.values())

    report = {
        "tournament": "101_TTNTNC_Round_Robin",
        "budget_mode": budget_mode,
        "num_openings": len(openings),
        "games_per_pair": len(openings) * 2,
        "agents": all_agent_names,
        "matchups": matchups,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    root_summary = Path("results/tournament_summary.json")
    if root_summary.resolve() != summary_path.resolve() and output_dir is None:
        root_summary.parent.mkdir(parents=True, exist_ok=True)
        with open(root_summary, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy giải đấu Round-Robin Arena cho các cấp độ tác tử (hỗ trợ chạy tiếp nối).")
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
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Xóa sạch kết quả cũ (matches.jsonl, games.pgn, tournament_summary.json) để chạy lại từ đầu.",
    )
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
        resume=not args.reset,
    )
    print(f"Giải đấu hoàn tất ({report['games_per_pair']} ván/cặp, chế độ {report['budget_mode']}, resume={not args.reset}).")
    for m in report["matchups"]:
        reused = m.get("reused_games", 0)
        played = m.get("newly_played_games", m["games"])
        print(
            f"{m['agent_a']} vs {m['agent_b']} [mới: {played}, tái sử dụng: {reused}]: "
            f"W/D/L={m['wins']}/{m['draws']}/{m['losses']} | "
            f"S={m['score']:.3f} | Wilson95={m['wilson_ci_95']} | Boot95={m['bootstrap_ci_95']}"
        )


if __name__ == "__main__":
    main()
