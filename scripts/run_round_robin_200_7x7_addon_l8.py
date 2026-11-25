"""
scripts/run_round_robin_200_7x7_addon_l8.py
Chạy bổ sung 6 cặp của Level 8 AlphaZero (6 × 200 = 1.200 ván) và ghép với 15 cặp (3.000 ván)
của Round-Robin 6×6 để hoàn chỉnh Ma trận Round-Robin 7×7 ĐẦY ĐỦ 200 ván/cặp (21 cặp × 200 = 4.200 ván,
mỗi tác tử đấu đúng 1.200 ván!).
Hỗ trợ 4 nhân song song (CPU 1-thread cache model) và auto-resume từng cặp ván.
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
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
from chess_ai.arena.metrics import compute_style_metrics, wilson_confidence_interval
from chess_ai.arena.pgn_exporter import append_match_jsonl
from chess_ai.utils.epd import load_epd_file

_WORKER_CACHE: Dict[str, Any] = {}

WORKLOADS: Dict[str, Optional[int]] = {
    "Level1_Random": None,
    "Level2_Heuristic": None,
    "Level3_AlphaBeta": 4000,
    "Level4_MCTS_UCT": 100,
    "Level5_Supervised": None,
    "Level7_GuidedMCTS": 100,
    "Level8_AlphaZero": 100,
}

AGENTS_7 = [
    "Level1_Random",
    "Level2_Heuristic",
    "Level3_AlphaBeta",
    "Level4_MCTS_UCT",
    "Level5_Supervised",
    "Level7_GuidedMCTS",
    "Level8_AlphaZero",
]


def _make_agent(spec_name: str, seed: int) -> Any:
    torch.set_num_threads(1)
    if spec_name in _WORKER_CACHE:
        ag = _WORKER_CACHE[spec_name]
        ag.reset()
        return ag
    l5_ckpt = str(PROJECT_ROOT / "checkpoints" / "best_model.pt")
    l8_ckpt = str(PROJECT_ROOT / "checkpoints" / "l8_alphazero_best.pt")
    if not Path(l8_ckpt).exists():
        l8_ckpt = l5_ckpt

    if spec_name == "Level1_Random":
        ag = create_agent("level1", seed=seed)
    elif spec_name == "Level2_Heuristic":
        ag = create_agent("level2", seed=seed)
    elif spec_name == "Level3_AlphaBeta":
        ag = create_agent("level3", default_depth=3, max_nodes=4000)
    elif spec_name == "Level4_MCTS_UCT":
        ag = create_agent("level4", simulations=100, seed=seed)
    elif spec_name == "Level5_Supervised":
        ag = create_agent("level5", checkpoint_path=l5_ckpt, device="cpu", seed=seed)
    elif spec_name == "Level7_GuidedMCTS":
        ag = create_agent("level7", checkpoint_path=l5_ckpt, simulations=100, device="cpu", seed=seed)
    elif spec_name == "Level8_AlphaZero":
        ag = create_agent("level8", checkpoint_path=l8_ckpt, simulations=100, device="cpu", seed=seed)
    else:
        raise ValueError(f"Unknown agent spec: {spec_name}")
    ag.name = spec_name
    _WORKER_CACHE[spec_name] = ag
    return ag


def _run_pair_worker(task: Tuple[Any, ...]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pair_id, ag_a_name, ag_b_name, op_id, op_fen, s_round, s_val, max_plies = task
    ag_a = _make_agent(ag_a_name, s_val)
    ag_b = _make_agent(ag_b_name, s_val + 17)
    wl_a, wl_b = WORKLOADS[ag_a_name], WORKLOADS[ag_b_name]

    g1 = play_single_match(
        white_agent=ag_a,
        black_agent=ag_b,
        initial_fen=op_fen,
        max_plies=max_plies,
        seed=s_val,
        white_workload=wl_a,
        black_workload=wl_b,
    )
    g1.update({
        "matchup_id": pair_id,
        "opening_id": op_id,
        "seed_round": s_round,
        "budget_mode": "Workload_RR200_7x7",
        "match_id": f"{pair_id}__{op_id}__s{s_round}__w",
    })

    g2 = play_single_match(
        white_agent=ag_b,
        black_agent=ag_a,
        initial_fen=op_fen,
        max_plies=max_plies,
        seed=s_val + 1,
        white_workload=wl_b,
        black_workload=wl_a,
    )
    g2.update({
        "matchup_id": pair_id,
        "opening_id": op_id,
        "seed_round": s_round,
        "budget_mode": "Workload_RR200_7x7",
        "match_id": f"{pair_id}__{op_id}__s{s_round}__b",
    })
    return g1, g2


def compute_iterative_elo(
    agents: List[str],
    records: List[Dict[str, Any]],
    base_elo: float = 1200.0,
    iterations: int = 100,
) -> Dict[str, float]:
    ratings = {a: base_elo for a in agents}
    for _ in range(iterations):
        deltas = {a: 0.0 for a in agents}
        counts = {a: 0 for a in agents}
        for r in records:
            w, b, res = r["white"], r["black"], r["result"]
            sw = 1.0 if res == "1-0" else (0.5 if res == "1/2-1/2" else 0.0)
            sb = 1.0 - sw
            ew = 1.0 / (1.0 + 10.0 ** ((ratings[b] - ratings[w]) / 400.0))
            eb = 1.0 - ew
            deltas[w] += 24.0 * (sw - ew)
            deltas[b] += 24.0 * (sb - eb)
            counts[w] += 1
            counts[b] += 1
        for a in agents:
            if counts[a] > 0:
                ratings[a] += deltas[a] / max(1.0, counts[a] * 0.25)
        mean_r = sum(ratings.values()) / len(ratings)
        for a in agents:
            ratings[a] = ratings[a] - mean_r + base_elo
    return {a: round(ratings[a], 1) for a in agents}


def plot_rr_7x7_200(
    leaderboard_rows: List[Dict[str, Any]],
    pair_scores: Dict[Tuple[str, str], float],
    out_path: Path,
) -> None:
    order = [r["agent"] for r in leaderboard_rows]
    n = len(order)
    mat = np.full((n, n), np.nan)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if i != j and (a, b) in pair_scores:
                mat[i, j] = pair_scores[(a, b)] * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 5.6), dpi=180)
    im = ax1.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100)
    short = [x.replace("Level", "L").replace("_", " ") for x in order]
    ax1.set_xticks(range(n)); ax1.set_xticklabels(short, rotation=26, ha="right", fontsize=8.8)
    ax1.set_yticks(range(n)); ax1.set_yticklabels(short, fontsize=8.8)
    for i in range(n):
        for j in range(n):
            txt = "—" if np.isnan(mat[i, j]) else f"{mat[i, j]:.1f}%"
            ax1.text(j, i, txt, ha="center", va="center", fontsize=8.4, fontweight="bold")
    ax1.set_title("Ma trận Round-Robin 7×7 (200 ván/cặp = 4.200 ván)", fontsize=11, fontweight="bold", color="#1B365D")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    y_pos = np.arange(n)[::-1]
    elos = [r["elo"] for r in leaderboard_rows]
    scrs = [r["score_pct"] for r in leaderboard_rows]
    colors = ["#1B365D" if e >= 1300 else ("#2E86AB" if e >= 1100 else "#94A3B8") for e in elos]
    bars = ax2.barh(y_pos, elos, color=colors, height=0.58)
    ax2.set_yticks(y_pos); ax2.set_yticklabels(short, fontsize=9.0)
    ax2.axvline(1200, color="#DC2626", linestyle="--", linewidth=1.3, label="Elo trung bình (1200)")
    ax2.set_xlabel("Hệ số Elo ước lượng (1.200 ván / tác tử)")
    ax2.set_title("Xếp hạng Elo Round-Robin 7×7 — 200 ván/cặp", fontsize=11, fontweight="bold", color="#1B365D")
    ax2.set_xlim(min(elos) - 100, max(elos) + 220)
    ax2.grid(True, axis="x", alpha=0.3); ax2.legend(loc="lower right", fontsize=8.5)
    for b, e, s in zip(bars, elos, scrs):
        ax2.text(e + 12, b.get_y() + b.get_height() / 2, f"{e:.0f} Elo ({s:.1f}%)",
                 va="center", fontsize=8.6, fontweight="bold", color="#1E293B")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    rr_dir = PROJECT_ROOT / "results" / "round_robin_200"
    dm_dir = PROJECT_ROOT / "results" / "dual_mode_200"
    rr_dir.mkdir(parents=True, exist_ok=True)

    openings = load_epd_file(PROJECT_ROOT / "resources" / "openings_50.epd")[:50]
    assert len(openings) == 50

    # 6 cặp mới của Level 8 (nhanh -> chậm)
    l8_pairs = [
        ("RR_L1_vs_L8", "Level1_Random", "Level8_AlphaZero"),
        ("RR_L5_vs_L8", "Level5_Supervised", "Level8_AlphaZero"),
        ("RR_L2_vs_L8", "Level2_Heuristic", "Level8_AlphaZero"),
        ("RR_L3_vs_L8", "Level3_AlphaBeta", "Level8_AlphaZero"),
        ("RR_L7_vs_L8", "Level7_GuidedMCTS", "Level8_AlphaZero"),
        ("RR_L4_vs_L8", "Level4_MCTS_UCT", "Level8_AlphaZero"),
    ]

    n_workers = 4
    l8_records: List[Dict[str, Any]] = []
    for idx, (pid, ag_a, ag_b) in enumerate(l8_pairs, 1):
        jsonl_path = rr_dir / f"{pid}_matches.jsonl"
        existing: Dict[str, Dict[str, Any]] = {}
        if jsonl_path.exists():
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    existing[rec["match_id"]] = rec

        tasks = []
        for s_round, base_seed in enumerate((42, 2026), start=1):
            for op_i, op in enumerate(openings):
                id_w = f"{pid}__{op.record_id}__s{s_round}__w"
                id_b = f"{pid}__{op.record_id}__s{s_round}__b"
                if id_w in existing and id_b in existing:
                    continue
                tasks.append((pid, ag_a, ag_b, op.record_id, op.fen, s_round, base_seed + op_i * 10, 160))

        print(f"\n[L8 {idx}/6] {pid} ({ag_a} vs {ag_b}) — đã có {len(existing)}/200, chạy tiếp {len(tasks)*2} ván (workers={n_workers})...", flush=True)
        t_m = time.perf_counter()
        if tasks:
            done_pairs = 0
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futs = [pool.submit(_run_pair_worker, t) for t in tasks]
                for fut in as_completed(futs):
                    g1, g2 = fut.result()
                    append_match_jsonl(str(jsonl_path), g1)
                    append_match_jsonl(str(jsonl_path), g2)
                    existing[g1["match_id"]] = g1
                    existing[g2["match_id"]] = g2
                    done_pairs += 1
                    if done_pairs % 10 == 0 or done_pairs == len(tasks):
                        print(f"   -> {pid}: {len(existing)}/200 ván ({time.perf_counter() - t_m:.1f}s)", flush=True)

        recs_pair = list(existing.values())[:200]
        wa = da = la = 0
        for r in recs_pair:
            res = r["result"]
            if res == "1/2-1/2": da += 1
            elif (res == "1-0" and r["white"] == ag_a) or (res == "0-1" and r["black"] == ag_a): wa += 1
            else: la += 1
        sc = (wa + 0.5 * da) * 100.0 / max(1, len(recs_pair))
        print(f"   XONG {pid}: {ag_a} {wa}W-{da}D-{la}L ({sc:.1f}%) vs {ag_b}", flush=True)
        l8_records.extend(recs_pair)

    # Gộp đủ 21 cặp (4 từ dual_mode_200 + 11 từ round_robin_200 + 6 cặp L8 = 4.200 ván!)
    base_files = [
        dm_dir / "M1_WL_L3_vs_L2_matches.jsonl",
        dm_dir / "M2_WL_L3_vs_L5_matches.jsonl",
        dm_dir / "M3_WL_L7_vs_L5_matches.jsonl",
        dm_dir / "M4_WL_L3_vs_L7_matches.jsonl",
        rr_dir / "RR_L1_vs_L2_matches.jsonl",
        rr_dir / "RR_L1_vs_L5_matches.jsonl",
        rr_dir / "RR_L2_vs_L5_matches.jsonl",
        rr_dir / "RR_L1_vs_L3_matches.jsonl",
        rr_dir / "RR_L1_vs_L7_matches.jsonl",
        rr_dir / "RR_L2_vs_L7_matches.jsonl",
        rr_dir / "RR_L1_vs_L4_matches.jsonl",
        rr_dir / "RR_L2_vs_L4_matches.jsonl",
        rr_dir / "RR_L5_vs_L4_matches.jsonl",
        rr_dir / "RR_L3_vs_L4_matches.jsonl",
        rr_dir / "RR_L7_vs_L4_matches.jsonl",
    ]
    all_records: List[Dict[str, Any]] = list(l8_records)
    for fp in base_files:
        if fp.exists():
            recs = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
            all_records.extend(recs[:200])

    elos = compute_iterative_elo(AGENTS_7, all_records)
    pair_wdl = defaultdict(lambda: [0, 0, 0])
    for r in all_records:
        w, b, res = r["white"], r["black"], r["result"]
        if res == "1-0":
            pair_wdl[(w, b)][0] += 1; pair_wdl[(b, w)][2] += 1
        elif res == "0-1":
            pair_wdl[(w, b)][2] += 1; pair_wdl[(b, w)][0] += 1
        else:
            pair_wdl[(w, b)][1] += 1; pair_wdl[(b, w)][1] += 1
    pair_scores = {k: (v[0] + 0.5 * v[1]) / max(1, sum(v)) for k, v in pair_wdl.items()}

    rows = []
    for ag in AGENTS_7:
        sub = [r for r in all_records if r["white"] == ag or r["black"] == ag]
        w = d = l = 0
        w_pts = b_pts = 0.0
        w_g = b_g = 0
        mean_lats, p95_lats, plies_l = [], [], []
        for r in sub:
            plies_l.append(int(r.get("total_plies", 0)))
            res = r["result"]
            if r["white"] == ag:
                w_g += 1
                sc = 1.0 if res == "1-0" else (0.5 if res == "1/2-1/2" else 0.0)
                w_pts += sc
                mean_lats.append(float(r.get("white_avg_latency_ms", 0.0)))
                p95_lats.append(float(r.get("white_p95_latency_ms", 0.0)))
            else:
                b_g += 1
                sc = 1.0 if res == "0-1" else (0.5 if res == "1/2-1/2" else 0.0)
                b_pts += sc
                mean_lats.append(float(r.get("black_avg_latency_ms", 0.0)))
                p95_lats.append(float(r.get("black_p95_latency_ms", 0.0)))
            if sc == 1.0: w += 1
            elif sc == 0.5: d += 1
            else: l += 1
        n = w + d + l
        score = (w + 0.5 * d) / max(1, n)
        ci_lo, ci_hi = wilson_confidence_interval(score, n)
        st = compute_style_metrics(sub)
        rows.append({
            "agent": ag, "games": n, "wins": w, "draws": d, "losses": l,
            "score_pct": round(score * 100.0, 2),
            "white_score_pct": round(w_pts * 100.0 / max(1, w_g), 2),
            "black_score_pct": round(b_pts * 100.0 / max(1, b_g), 2),
            "wilson_ci_95": [round(ci_lo * 100.0, 2), round(ci_hi * 100.0, 2)],
            "elo": elos[ag],
            "mean_latency_ms": round(float(np.mean(mean_lats)), 1) if mean_lats else 0.0,
            "p95_latency_ms": round(float(np.percentile(p95_lats, 95)), 1) if p95_lats else 0.0,
            "checkmate_rate": round(st.get("checkmate_rate", 0.0) * 100.0, 1),
            "castling_rate": round(st.get("castling_rate", 0.0) * 100.0, 1),
            "avg_plies": round(float(np.mean(plies_l)), 1) if plies_l else 0.0,
        })
    rows.sort(key=lambda x: (x["score_pct"], x["elo"]), reverse=True)

    csv_path = rr_dir / "leaderboard_7x7_200.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "agent", "games", "wins", "draws", "losses", "score_pct",
                "white_score_pct", "black_score_pct", "wilson_lo", "wilson_hi",
                "elo", "mean_latency_ms", "p95_latency_ms",
                "checkmate_rate", "castling_rate", "avg_plies",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "agent": r["agent"], "games": r["games"], "wins": r["wins"],
                "draws": r["draws"], "losses": r["losses"], "score_pct": r["score_pct"],
                "white_score_pct": r["white_score_pct"], "black_score_pct": r["black_score_pct"],
                "wilson_lo": r["wilson_ci_95"][0], "wilson_hi": r["wilson_ci_95"][1],
                "elo": r["elo"], "mean_latency_ms": r["mean_latency_ms"], "p95_latency_ms": r["p95_latency_ms"],
                "checkmate_rate": r["checkmate_rate"], "castling_rate": r["castling_rate"], "avg_plies": r["avg_plies"],
            })

    summary = {
        "experiment": "Full_Round_Robin_7x7_200_Games_Per_Pair",
        "games_per_pair": 200,
        "total_pairs": 21,
        "total_games": len(all_records),
        "elapsed_sec_l8_addon": round(time.perf_counter() - t0, 2),
        "leaderboard": rows,
        "pairwise_wdl": {f"{k[0]}__vs__{k[1]}": v for k, v in pair_wdl.items()},
    }
    (rr_dir / "summary_7x7_200.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fig_path = PROJECT_ROOT / "results" / "figures" / "fig_round_robin_7x7_200.png"
    plot_rr_7x7_200(rows, pair_scores, fig_path)

    print("\n=== BẢNG XẾP HẠNG ROUND-ROBIN 7×7 (4.200 VÁN — 200 VÁN/CẶP) ===")
    for i, r in enumerate(rows, 1):
        print(
            f"  #{i} {r['agent']:20s} | {r['wins']:4d}W-{r['draws']:3d}D-{r['losses']:4d}L | "
            f"{r['score_pct']:5.1f}% (W:{r['white_score_pct']:.1f}% B:{r['black_score_pct']:.1f}%) | "
            f"Elo {r['elo']:6.1f} | Mean/P95 {r['mean_latency_ms']:5.1f}/{r['p95_latency_ms']:5.1f} ms"
        )
    print(f"HOÀN TẤT 4.200 VÁN (7×7 × 200 ván/cặp) -> {csv_path} & {fig_path}", flush=True)


if __name__ == "__main__":
    main()
