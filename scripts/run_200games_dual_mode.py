"""
scripts/run_200games_dual_mode.py
Thực nghiệm chuẩn hóa tối đa theo yêu cầu đề tài:
  1. Profiling RAM (RSS) & VRAM (CUDA) thực tế cho toàn bộ 8 cấp độ AI (Level 1 -> Level 8).
  2. Chạy đủ 200 ván/cặp (50 khai cuộc chuẩn x 2 màu Trắng/Đen x 2 seeds = 200 ván, chia đều 50/50)
     theo HAI CHẾ ĐỘ NGÂN SÁCH ĐỘC LẬP:
     - Chế độ 1 (Workload-Control / Cấu hình mặc định minh bạch):
       (1) Level3_AlphaBeta vs Level2_Heuristic (200 ván)
       (2) Level3_AlphaBeta vs Level5_Supervised (200 ván)
       (3) Level7_GuidedMCTS vs Level5_Supervised (200 ván)
       (4) Level3_AlphaBeta vs Level7_GuidedMCTS (200 ván)
     - Chế độ 2 (Time-Control / Ngân sách thời gian công bằng: 0.15s/nước):
       (5) Level3_AlphaBeta vs Level2_Heuristic (200 ván, 0.15s/nước)
       (6) Level3_AlphaBeta vs Level7_GuidedMCTS (200 ván, 0.15s/nước)
  3. Báo cáo đầy đủ W-D-L, Score, Wilson 95% CI, Paired Bootstrap 95% CI, Mean & P95 Latency,
     Avg Nodes/Simulations, RAM/VRAM và xuất biểu đồ fig_dual_mode_200games.png.
  Hỗ trợ đa tiến trình (ProcessPoolExecutor) và resume tự động không mất ván đã chạy.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from chess_ai.agents.registry import create_agent
from chess_ai.arena.match import play_single_match
from chess_ai.arena.metrics import paired_bootstrap_confidence_interval, wilson_confidence_interval
from chess_ai.arena.pgn_exporter import append_match_jsonl
from chess_ai.core.chess_env import ChessEnv
from chess_ai.utils.epd import load_epd_file

try:
    import psutil
except ImportError:
    psutil = None


def _get_rss_mb() -> float:
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return 0.0


def profile_resources_8_levels() -> List[Dict[str, Any]]:
    """Đo RAM (delta RSS / tiến trình) & VRAM (CUDA peak) và độ trễ 1 nước tiêu biểu cho 8 cấp độ."""
    l5_ckpt = str(PROJECT_ROOT / "checkpoints" / "best_model.pt")
    l6_ckpt = str(PROJECT_ROOT / "checkpoints" / "l6_krk_qtable.pkl")
    l8_ckpt = str(PROJECT_ROOT / "checkpoints" / "l8_alphazero_best.pt")
    if not Path(l8_ckpt).exists():
        l8_ckpt = l5_ckpt

    specs = [
        ("Level1_Random", "level1", {}, None, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        ("Level2_Heuristic", "level2", {}, None, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        ("Level3_AlphaBeta", "level3", {"default_depth": 3, "max_nodes": 4000}, 4000, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        ("Level4_MCTS_UCT", "level4", {"simulations": 100}, 100, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        ("Level5_Supervised", "level5", {"checkpoint_path": l5_ckpt}, None, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        ("Level6_RL_Endgame", "level6", {}, None, "4k3/R7/4K3/8/8/8/8/8 w - - 0 1"),
        ("Level7_GuidedMCTS", "level7", {"checkpoint_path": l5_ckpt, "simulations": 100}, 100, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        ("Level8_AlphaZero", "level8", {"checkpoint_path": l8_ckpt, "simulations": 100}, 100, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
    ]

    results = []
    has_cuda = torch.cuda.is_available()
    for display_name, agent_id, kwargs, wl, fen in specs:
        gc.collect()
        if has_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        rss_before = _get_rss_mb()
        t0 = time.perf_counter()
        ag = create_agent(agent_id, **kwargs)
        load_ms = (time.perf_counter() - t0) * 1000.0
        env = ChessEnv(fen=fen)
        t1 = time.perf_counter()
        _ = ag.select_move(env, max_workload=wl)
        move_ms = (time.perf_counter() - t1) * 1000.0
        rss_after = _get_rss_mb()
        vram_peak = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if has_cuda else 0.0
        uses_gpu = agent_id in ("level5", "level7", "level8") and has_cuda

        ckpt_size_mb = 0.0
        if "checkpoint_path" in kwargs and Path(kwargs["checkpoint_path"]).exists():
            ckpt_size_mb = Path(kwargs["checkpoint_path"]).stat().st_size / (1024 * 1024)
        elif agent_id == "level6" and Path(l6_ckpt).exists():
            ckpt_size_mb = Path(l6_ckpt).stat().st_size / (1024 * 1024)

        results.append({
            "agent": display_name,
            "device": "CUDA (GTX 1650)" if uses_gpu else "CPU",
            "checkpoint_mb": round(ckpt_size_mb, 2),
            "process_rss_mb": round(rss_after, 1),
            "delta_ram_mb": round(max(0.0, rss_after - rss_before), 1),
            "vram_peak_mb": round(vram_peak if uses_gpu else 0.0, 1),
            "load_time_ms": round(load_ms, 1),
            "sample_move_ms": round(move_ms, 1),
        })
        del ag
        gc.collect()
    return results


_WORKER_CACHE: Dict[str, Any] = {}


def _make_agent(spec_name: str, seed: int, device: Optional[str] = "cpu") -> Any:
    """Khởi tạo hoặc tái sử dụng agent trong cùng worker (dùng CPU 1-thread để chạy 4 worker song song không OOM VRAM)."""
    torch.set_num_threads(1)
    key = f"{spec_name}__{device}"
    if key in _WORKER_CACHE:
        ag = _WORKER_CACHE[key]
        ag.reset()
        return ag
    l5_ckpt = str(PROJECT_ROOT / "checkpoints" / "best_model.pt")
    if spec_name == "Level2_Heuristic":
        ag = create_agent("level2", seed=seed)
    elif spec_name == "Level3_AlphaBeta":
        ag = create_agent("level3", default_depth=3, max_nodes=4000)
    elif spec_name == "Level5_Supervised":
        ag = create_agent("level5", checkpoint_path=l5_ckpt, device=device, seed=seed)
    elif spec_name == "Level7_GuidedMCTS":
        ag = create_agent("level7", checkpoint_path=l5_ckpt, simulations=100, device=device, seed=seed)
    else:
        raise ValueError(f"Unknown spec: {spec_name}")
    ag.name = spec_name
    _WORKER_CACHE[key] = ag
    return ag


def _run_pair_task(task: Tuple[Any, ...]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    (
        matchup_id,
        agent_a_name,
        agent_b_name,
        mode,
        opening_id,
        opening_fen,
        seed_round,
        seed_val,
        wl_a,
        wl_b,
        time_limit_sec,
        max_plies,
    ) = task
    ag_a = _make_agent(agent_a_name, seed_val)
    ag_b = _make_agent(agent_b_name, seed_val + 17)

    g1 = play_single_match(
        white_agent=ag_a,
        black_agent=ag_b,
        initial_fen=opening_fen,
        time_limit_sec=time_limit_sec,
        max_plies=max_plies,
        seed=seed_val,
        white_workload=wl_a,
        black_workload=wl_b,
    )
    g1["matchup_id"] = matchup_id
    g1["opening_id"] = opening_id
    g1["seed_round"] = seed_round
    g1["budget_mode"] = mode
    g1["match_id"] = f"{matchup_id}__{opening_id}__s{seed_round}__w"

    g2 = play_single_match(
        white_agent=ag_b,
        black_agent=ag_a,
        initial_fen=opening_fen,
        time_limit_sec=time_limit_sec,
        max_plies=max_plies,
        seed=seed_val + 1,
        white_workload=wl_b,
        black_workload=wl_a,
    )
    g2["matchup_id"] = matchup_id
    g2["opening_id"] = opening_id
    g2["seed_round"] = seed_round
    g2["budget_mode"] = mode
    g2["match_id"] = f"{matchup_id}__{opening_id}__s{seed_round}__b"
    return g1, g2


def summarize_matchup(
    matchup_id: str,
    agent_a: str,
    agent_b: str,
    mode: str,
    budget_desc: str,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    wins = draws = losses = 0
    w_pts = b_pts = 0.0
    w_games = b_games = 0
    a_mean_lats, a_p95_lats, b_mean_lats, b_p95_lats = [], [], [], []
    a_nodes, b_nodes, plies_list = [], [], []
    pair_scores_map: Dict[Tuple[str, int], List[float]] = {}

    for r in records:
        plies_list.append(int(r.get("total_plies", 0)))
        res = r["result"]
        key = (str(r.get("opening_id", "")), int(r.get("seed_round", 0)))
        if r["white"] == agent_a:
            w_games += 1
            sc = 1.0 if res == "1-0" else (0.5 if res == "1/2-1/2" else 0.0)
            w_pts += sc
            a_mean_lats.append(float(r.get("white_avg_latency_ms", 0.0)))
            a_p95_lats.append(float(r.get("white_p95_latency_ms", 0.0)))
            b_mean_lats.append(float(r.get("black_avg_latency_ms", 0.0)))
            b_p95_lats.append(float(r.get("black_p95_latency_ms", 0.0)))
            a_nodes.append(float(r.get("white_avg_nps", 0.0) or r.get("white_avg_sps", 0.0)))
            b_nodes.append(float(r.get("black_avg_nps", 0.0) or r.get("black_avg_sps", 0.0)))
        else:
            b_games += 1
            sc = 1.0 if res == "0-1" else (0.5 if res == "1/2-1/2" else 0.0)
            b_pts += sc
            a_mean_lats.append(float(r.get("black_avg_latency_ms", 0.0)))
            a_p95_lats.append(float(r.get("black_p95_latency_ms", 0.0)))
            b_mean_lats.append(float(r.get("white_avg_latency_ms", 0.0)))
            b_p95_lats.append(float(r.get("white_p95_latency_ms", 0.0)))
            a_nodes.append(float(r.get("black_avg_nps", 0.0) or r.get("black_avg_sps", 0.0)))
            b_nodes.append(float(r.get("white_avg_nps", 0.0) or r.get("white_avg_sps", 0.0)))

        if sc == 1.0:
            wins += 1
        elif sc == 0.5:
            draws += 1
        else:
            losses += 1
        pair_scores_map.setdefault(key, []).append(sc)

    n = len(records)
    score = (wins + 0.5 * draws) / max(1, n)
    w_lo, w_hi = wilson_confidence_interval(score, n)
    pair_avgs = [float(np.mean(v)) for v in pair_scores_map.values() if v]
    b_lo, b_hi = paired_bootstrap_confidence_interval(pair_avgs, seed=42) if pair_avgs else (w_lo, w_hi)

    return {
        "matchup_id": matchup_id,
        "agent_a": agent_a,
        "agent_b": agent_b,
        "mode": mode,
        "budget_desc": budget_desc,
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_pct": round(score * 100.0, 2),
        "white_score_pct": round(w_pts * 100.0 / max(1, w_games), 2),
        "black_score_pct": round(b_pts * 100.0 / max(1, b_games), 2),
        "wilson_ci_95": [round(w_lo * 100.0, 2), round(w_hi * 100.0, 2)],
        "bootstrap_ci_95": [round(b_lo * 100.0, 2), round(b_hi * 100.0, 2)],
        "a_mean_latency_ms": round(float(np.mean(a_mean_lats)), 1) if a_mean_lats else 0.0,
        "a_p95_latency_ms": round(float(np.percentile(a_p95_lats, 95)), 1) if a_p95_lats else 0.0,
        "b_mean_latency_ms": round(float(np.mean(b_mean_lats)), 1) if b_mean_lats else 0.0,
        "b_p95_latency_ms": round(float(np.percentile(b_p95_lats, 95)), 1) if b_p95_lats else 0.0,
        "a_avg_nps_or_sps": round(float(np.mean(a_nodes)), 1) if a_nodes else 0.0,
        "b_avg_nps_or_sps": round(float(np.mean(b_nodes)), 1) if b_nodes else 0.0,
        "avg_plies": round(float(np.mean(plies_list)), 1) if plies_list else 0.0,
    }


def plot_dual_mode_200(summaries: List[Dict[str, Any]], out_path: Path) -> None:
    labels = [
        f"{s['agent_a'].replace('Level', 'L').split('_')[0]} vs {s['agent_b'].replace('Level', 'L').split('_')[0]}\n({s['mode']})"
        for s in summaries
    ]
    scores = np.array([s["score_pct"] for s in summaries])
    lo = np.array([s["wilson_ci_95"][0] for s in summaries])
    hi = np.array([s["wilson_ci_95"][1] for s in summaries])
    lat_a = np.array([s["a_mean_latency_ms"] for s in summaries])
    p95_a = np.array([s["a_p95_latency_ms"] for s in summaries])
    colors = ["#1B365D" if s["mode"] == "Workload" else "#D97706" for s in summaries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.0), dpi=180)
    x = np.arange(len(summaries))
    bars = ax1.bar(x, scores, color=colors, width=0.58, alpha=0.9)
    ax1.errorbar(x, scores, yerr=[scores - lo, hi - scores], fmt="none", ecolor="#0F172A", capsize=5, elinewidth=1.6)
    ax1.axhline(50.0, color="#DC2626", linestyle="--", linewidth=1.3, label="Mốc cân bằng (50%)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylim(0, 114); ax1.set_ylabel("Tỷ lệ điểm của tác tử A (%) — 200 ván/cặp")
    ax1.set_title("Kết quả Đối kháng 200 ván/cặp (Xanh: Workload | Cam: Time-Control 0,15s)", fontsize=10.5, fontweight="bold", color="#1B365D")
    ax1.grid(True, axis="y", alpha=0.3); ax1.legend(loc="lower right")
    for b, sc in zip(bars, scores):
        ax1.text(b.get_x() + b.get_width() / 2, min(107, sc + 4.5), f"{sc:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    ax2.bar(x - 0.18, lat_a, width=0.35, color="#2563EB", label="Độ trễ Trung bình (Mean ms)")
    ax2.bar(x + 0.18, p95_a, width=0.35, color="#EF4444", label="Độ trễ Phân vị P95 (ms)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("Độ trễ suy nghĩ của tác tử A (ms / nước)")
    ax2.set_title("So sánh Độ trễ Mean vs P95 qua 2 Chế độ Ngân sách", fontsize=10.5, fontweight="bold", color="#1B365D")
    ax2.grid(True, axis="y", alpha=0.3); ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    out_dir = PROJECT_ROOT / "results" / "dual_mode_200"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "resource_profile.json"
    summary_path = out_dir / "dual_mode_200_summary.json"
    fig_path = PROJECT_ROOT / "results" / "figures" / "fig_dual_mode_200games.png"

    if res_path.exists():
        print("1. Nạp lại kết quả Profiling RAM/VRAM đã lưu từ resource_profile.json...", flush=True)
        res_profile = json.loads(res_path.read_text(encoding="utf-8"))
    else:
        print("1. Profiling RAM (RSS) & VRAM (CUDA) thực tế cho cả 8 cấp độ AI...", flush=True)
        res_profile = profile_resources_8_levels()
        res_path.write_text(json.dumps(res_profile, ensure_ascii=False, indent=2), encoding="utf-8")

    openings = load_epd_file(PROJECT_ROOT / "resources" / "openings_50.epd")[:50]
    assert len(openings) == 50, f"Cần đủ 50 khai cuộc, hiện có {len(openings)}"

    # 6 cặp x 200 ván = 1.200 ván — TẤT CẢ chạy song song 4 workers (mỗi worker tái sử dụng model CPU 1-thread, tăng tốc ~4x, 0 OOM!)
    matchups = [
        # (id, A, B, mode, desc, wl_a, wl_b, time_limit, max_workers)
        ("M1_WL_L3_vs_L2", "Level3_AlphaBeta", "Level2_Heuristic", "Workload", "L3 max_nodes=4000 vs L2 1-ply", 4000, None, None, 4),
        ("M2_WL_L3_vs_L5", "Level3_AlphaBeta", "Level5_Supervised", "Workload", "L3 max_nodes=4000 vs L5 Policy-only", 4000, None, None, 4),
        ("M3_WL_L7_vs_L5", "Level7_GuidedMCTS", "Level5_Supervised", "Workload", "L7 100 sims vs L5 Policy-only", 100, None, None, 4),
        ("M4_WL_L3_vs_L7", "Level3_AlphaBeta", "Level7_GuidedMCTS", "Workload", "L3 max_nodes=4000 vs L7 100 sims", 4000, 100, None, 4),
        ("M5_TC_L3_vs_L2", "Level3_AlphaBeta", "Level2_Heuristic", "Time-Control", "Cùng 0.15s/nước (Time-Control)", None, None, 0.15, 4),
        ("M6_TC_L3_vs_L7", "Level3_AlphaBeta", "Level7_GuidedMCTS", "Time-Control", "Cùng 0.15s/nước (Time-Control)", None, None, 0.15, 4),
    ]

    summaries = []
    for idx, (mid, ag_a, ag_b, mode, desc, wl_a, wl_b, tlim, n_workers) in enumerate(matchups, 1):
        jsonl_path = out_dir / f"{mid}_matches.jsonl"
        existing: Dict[str, Dict[str, Any]] = {}
        if jsonl_path.exists():
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    existing[rec["match_id"]] = rec

        tasks = []
        for s_round, base_seed in enumerate((42, 2026), start=1):
            for op_i, op in enumerate(openings):
                id_w = f"{mid}__{op.record_id}__s{s_round}__w"
                id_b = f"{mid}__{op.record_id}__s{s_round}__b"
                if id_w in existing and id_b in existing:
                    continue
                tasks.append((
                    mid, ag_a, ag_b, mode, op.record_id, op.fen,
                    s_round, base_seed + op_i * 10, wl_a, wl_b, tlim, 160
                ))

        print(f"\n[{idx}/6] {mid} ({mode}: {desc}) — đã có {len(existing)}/200 ván, cần chạy {len(tasks)*2} ván (workers={n_workers})...", flush=True)
        t_m = time.perf_counter()
        if tasks:
            done_pairs = 0
            if n_workers == 1:
                # Chạy in-process tái sử dụng duy nhất 1 model CUDA (không tạo nhiều CUDA context, không OOM!)
                inst_a = _make_agent(ag_a, 42)
                inst_b = _make_agent(ag_b, 59)
                for t in tasks:
                    _, _, _, _, op_id, op_fen, s_round, s_val, wla, wlb, tl, mp = t
                    g1 = play_single_match(inst_a, inst_b, initial_fen=op_fen, time_limit_sec=tl, max_plies=mp, seed=s_val, white_workload=wla, black_workload=wlb)
                    g1.update({"matchup_id": mid, "opening_id": op_id, "seed_round": s_round, "budget_mode": mode, "match_id": f"{mid}__{op_id}__s{s_round}__w"})
                    g2 = play_single_match(inst_b, inst_a, initial_fen=op_fen, time_limit_sec=tl, max_plies=mp, seed=s_val + 1, white_workload=wlb, black_workload=wla)
                    g2.update({"matchup_id": mid, "opening_id": op_id, "seed_round": s_round, "budget_mode": mode, "match_id": f"{mid}__{op_id}__s{s_round}__b"})
                    append_match_jsonl(str(jsonl_path), g1)
                    append_match_jsonl(str(jsonl_path), g2)
                    existing[g1["match_id"]] = g1
                    existing[g2["match_id"]] = g2
                    done_pairs += 1
                    if done_pairs % 5 == 0 or done_pairs == len(tasks):
                        print(f"   -> {mid}: {len(existing)}/200 ván ({time.perf_counter() - t_m:.1f}s)", flush=True)
                del inst_a, inst_b
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futs = [pool.submit(_run_pair_task, t) for t in tasks]
                    for fut in as_completed(futs):
                        g1, g2 = fut.result()
                        append_match_jsonl(str(jsonl_path), g1)
                        append_match_jsonl(str(jsonl_path), g2)
                        existing[g1["match_id"]] = g1
                        existing[g2["match_id"]] = g2
                        done_pairs += 1
                        if done_pairs % 10 == 0 or done_pairs == len(tasks):
                            print(f"   -> {mid}: {len(existing)}/200 ván ({time.perf_counter() - t_m:.1f}s)", flush=True)

        recs_all = list(existing.values())
        sm = summarize_matchup(mid, ag_a, ag_b, mode, desc, recs_all)
        summaries.append(sm)
        print(
            f"   XONG {mid}: {sm['wins']}W-{sm['draws']}D-{sm['losses']}L ({sm['score_pct']:.1f}%, "
            f"Wilson [{sm['wilson_ci_95'][0]:.1f}%, {sm['wilson_ci_95'][1]:.1f}%], "
            f"Mean/P95 A: {sm['a_mean_latency_ms']}/{sm['a_p95_latency_ms']} ms)",
            flush=True,
        )

    out_payload = {
        "experiment": "Dual_Mode_200_Games_Per_Pair_And_Resource_Profile",
        "games_per_pair": 200,
        "total_games": sum(s["games"] for s in summaries),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "resource_profile": res_profile,
        "matchups": summaries,
    }
    summary_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_dual_mode_200(summaries, fig_path)
    print(f"\nHOÀN TẤT TOÀN BỘ 1.200 VÁN (200 ván/cặp, 2 chế độ) trong {out_payload['elapsed_sec']}s -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
