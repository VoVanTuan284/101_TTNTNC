"""
scripts/run_full_leaderboard.py
Chạy giải đấu vòng tròn tổng hợp (Full Round-Robin Leaderboard) cho toàn bộ 7 tác tử hỗ trợ
bàn cờ 8×8 đầy đủ (Level 1, 2, 3, 4, 5, 7, 8 — Level 6 chuyên biệt cờ tàn 3 quân KQK/KRK/KPK).
Kết quả lưu độc lập tại results/full_leaderboard/ (KHÔNG ghi đè kết quả cũ).
"""

from collections import defaultdict
import csv
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess
import matplotlib
matplotlib.use("Agg")
import chess
import matplotlib.pyplot as plt
import numpy as np

from chess_ai.agents.registry import create_agent
from chess_ai.arena.match import play_single_match
from chess_ai.arena.metrics import compute_style_metrics, wilson_confidence_interval
from chess_ai.arena.pgn_exporter import append_match_jsonl, append_pgn_file, export_game_to_pgn
from chess_ai.utils.epd import load_epd_file


def compute_iterative_elo(
    agents: List[str],
    records: List[Dict[str, Any]],
    base_elo: float = 1200.0,
    iterations: int = 80,
) -> Dict[str, float]:
    """Ước lượng Elo toàn cục bằng phương pháp lặp Bradley-Terry / Logistic MLE."""
    ratings = {a: base_elo for a in agents}
    for _ in range(iterations):
        deltas = {a: 0.0 for a in agents}
        counts = {a: 0 for a in agents}
        for r in records:
            w, b, res = r["white"], r["black"], r["result"]
            if res == "1-0":
                sw, sb = 1.0, 0.0
            elif res == "0-1":
                sw, sb = 0.0, 1.0
            else:
                sw, sb = 0.5, 0.5
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


def build_agents() -> Tuple[Dict[str, Any], Dict[str, int | None]]:
    """Khởi tạo 7 tác tử 8×8 dùng trực tiếp checkpoints đã huấn luyện."""
    l5_ckpt = str(PROJECT_ROOT / "checkpoints" / "best_model.pt")
    l8_ckpt = str(PROJECT_ROOT / "checkpoints" / "l8_alphazero_best.pt")
    if not Path(l8_ckpt).exists():
        l8_ckpt = l5_ckpt

    agents = {
        "Level1_Random": create_agent("level1", seed=42),
        "Level2_Heuristic": create_agent("level2", seed=42),
        "Level3_AlphaBeta": create_agent("level3", default_depth=3, max_nodes=10000),
        "Level4_MCTS_UCT": create_agent("level4", simulations=100, seed=42),
        "Level5_Supervised": create_agent("level5", checkpoint_path=l5_ckpt, seed=42),
        "Level7_GuidedMCTS": create_agent("level7", checkpoint_path=l5_ckpt, simulations=200, seed=42),
        "Level8_AlphaZero": create_agent("level8", checkpoint_path=l8_ckpt, simulations=200, seed=42),
    }
    workloads = {
        "Level1_Random": None,
        "Level2_Heuristic": None,
        "Level3_AlphaBeta": 10000,
        "Level4_MCTS_UCT": 100,
        "Level5_Supervised": None,
        "Level7_GuidedMCTS": 200,
        "Level8_AlphaZero": 200,
    }
    # Đảm bảo tên chuẩn hóa đồng nhất
    for name, ag in agents.items():
        ag.name = name
    return agents, workloads


def plot_full_leaderboard(
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.4), dpi=180)
    im = ax1.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100)
    short_labels = [x.replace("Level", "L").replace("_", " ") for x in order]
    ax1.set_xticks(range(n)); ax1.set_xticklabels(short_labels, rotation=28, ha="right", fontsize=8.8)
    ax1.set_yticks(range(n)); ax1.set_yticklabels(short_labels, fontsize=8.8)
    for i in range(n):
        for j in range(n):
            txt = "—" if np.isnan(mat[i, j]) else f"{mat[i, j]:.0f}%"
            ax1.text(j, i, txt, ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax1.set_title("Ma trận đối đầu 7×7 (Tỷ lệ điểm % hàng thắng cột)", fontsize=11, fontweight="bold", color="#1B365D")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    y_pos = np.arange(n)[::-1]
    elos = [r["elo"] for r in leaderboard_rows]
    scrs = [r["score_pct"] for r in leaderboard_rows]
    colors = ["#1B365D" if e >= 1250 else ("#2E86AB" if e >= 1150 else "#94A3B8") for e in elos]
    bars = ax2.barh(y_pos, elos, color=colors, height=0.6)
    ax2.set_yticks(y_pos); ax2.set_yticklabels(short_labels, fontsize=9.0)
    ax2.axvline(1200, color="#DC2626", linestyle="--", linewidth=1.3, label="Mốc Elo trung bình (1200)")
    ax2.set_xlabel("Hệ số xếp hạng Elo ước lượng")
    ax2.set_title("Bảng xếp hạng Elo Tổng sắp Toàn hệ sinh thái (84 ván)", fontsize=11, fontweight="bold", color="#1B365D")
    ax2.set_xlim(min(elos) - 90, max(elos) + 170)
    ax2.grid(True, axis="x", alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8.5)
    for b, e, s in zip(bars, elos, scrs):
        ax2.text(e + 10, b.get_y() + b.get_height() / 2, f"{e:.0f} Elo ({s:.1f}%)",
                 va="center", fontsize=8.5, fontweight="bold", color="#1E293B")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    t_start = time.perf_counter()
    out_dir = PROJECT_ROOT / "results" / "full_leaderboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "matches.jsonl"
    pgn_path = out_dir / "games.pgn"
    csv_path = out_dir / "leaderboard.csv"
    summary_path = out_dir / "summary.json"
    fig_path = PROJECT_ROOT / "results" / "figures" / "fig_full_leaderboard.png"

    # Làm sạch bản cũ trong thư mục riêng này để đảm bảo nhất quán
    for p in (jsonl_path, pgn_path):
        if p.exists():
            p.unlink()

    openings_path = PROJECT_ROOT / "resources" / "openings_50.epd"
    if openings_path.exists():
        openings = load_epd_file(openings_path)[:2]  # 2 khai cuộc cân bằng x 2 màu = 4 ván/cặp -> 84 ván
    else:
        openings = []

    agents, workloads = build_agents()
    agent_names = list(agents.keys())
    pairs = list(combinations(agent_names, 2))
    total_games = len(pairs) * max(1, len(openings)) * 2
    print(f"Bắt đầu Full Round-Robin Leaderboard: {len(agent_names)} cấp độ, {len(pairs)} cặp, {total_games} ván...")

    records: List[Dict[str, Any]] = []
    game_idx = 0
    for a_name, b_name in pairs:
        ag_a, ag_b = agents[a_name], agents[b_name]
        wl_a, wl_b = workloads[a_name], workloads[b_name]
        for op_i, op in enumerate(openings):
            for swap in (False, True):
                game_idx += 1
                w_ag, b_ag = (ag_b, ag_a) if swap else (ag_a, ag_b)
                w_wl, b_wl = (wl_b, wl_a) if swap else (wl_a, wl_b)
                rec = play_single_match(
                    white_agent=w_ag,
                    black_agent=b_ag,
                    initial_fen=op.fen,
                    max_plies=180,
                    seed=1000 + game_idx,
                    white_workload=w_wl,
                    black_workload=b_wl,
                )
                rec["opening_id"] = op.record_id
                rec["budget_mode"] = "full_leaderboard_workload"
                rec["match_id"] = f"{w_ag.name}_vs_{b_ag.name}_{op.record_id}_{game_idx}"
                records.append(rec)
                append_match_jsonl(str(jsonl_path), rec)
                pgn_str = export_game_to_pgn(
                    moves=[chess.Move.from_uci(m) for m in rec["moves_uci"]],
                    headers={
                        "Event": "Chess AI 8x8 Full Round-Robin Leaderboard",
                        "White": rec["white"],
                        "Black": rec["black"],
                        "Result": rec["result"],
                        "Opening": op.record_id,
                    },
                    initial_fen=rec["opening_fen"],
                )
                append_pgn_file(str(pgn_path), pgn_str)
        elapsed_so_far = time.perf_counter() - t_start
        print(f"  [{game_idx:02d}/{total_games}] Xong cặp {a_name} vs {b_name} ({elapsed_so_far:.1f}s)", flush=True)

    # Tính bảng xếp hạng Elo & Wilson 95% CI & RQ4
    elos = compute_iterative_elo(agent_names, records)
    pair_wdl = defaultdict(lambda: [0, 0, 0])
    for r in records:
        w, b, res = r["white"], r["black"], r["result"]
        if res == "1-0":
            pair_wdl[(w, b)][0] += 1; pair_wdl[(b, w)][2] += 1
        elif res == "0-1":
            pair_wdl[(w, b)][2] += 1; pair_wdl[(b, w)][0] += 1
        else:
            pair_wdl[(w, b)][1] += 1; pair_wdl[(b, w)][1] += 1

    pair_scores = {k: (v[0] + 0.5 * v[1]) / max(1, sum(v)) for k, v in pair_wdl.items()}

    rows = []
    for ag in agent_names:
        sub = [r for r in records if r["white"] == ag or r["black"] == ag]
        w = d = l = 0
        lat_vals = []
        for r in sub:
            res = r["result"]
            if r["white"] == ag:
                lat_vals.append(float(r.get("white_avg_latency_ms", 0.0)))
                if res == "1-0": w += 1
                elif res == "0-1": l += 1
                else: d += 1
            else:
                lat_vals.append(float(r.get("black_avg_latency_ms", 0.0)))
                if res == "0-1": w += 1
                elif res == "1-0": l += 1
                else: d += 1
        n = w + d + l
        score = (w + 0.5 * d) / max(1, n)
        ci_lo, ci_hi = wilson_confidence_interval(w + 0.5 * d, n)
        st = compute_style_metrics(sub)
        rows.append({
            "agent": ag,
            "games": n,
            "wins": w,
            "draws": d,
            "losses": l,
            "score_pct": round(score * 100.0, 2),
            "wilson_ci_95": [round(ci_lo * 100.0, 2), round(ci_hi * 100.0, 2)],
            "elo": elos[ag],
            "avg_latency_ms": round(float(np.mean(lat_vals)) if lat_vals else 0.0, 1),
            "checkmate_rate": round(st.get("checkmate_rate", 0.0) * 100.0, 1),
            "castling_rate": round(st.get("castling_rate", 0.0) * 100.0, 1),
            "avg_plies": round(st.get("avg_plies", 0.0), 1),
        })

    rows.sort(key=lambda x: (x["score_pct"], x["elo"]), reverse=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "agent", "games", "wins", "draws", "losses", "score_pct",
                "wilson_lo", "wilson_hi", "elo", "avg_latency_ms",
                "checkmate_rate", "castling_rate", "avg_plies",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "agent": r["agent"], "games": r["games"], "wins": r["wins"],
                "draws": r["draws"], "losses": r["losses"], "score_pct": r["score_pct"],
                "wilson_lo": r["wilson_ci_95"][0], "wilson_hi": r["wilson_ci_95"][1],
                "elo": r["elo"], "avg_latency_ms": r["avg_latency_ms"],
                "checkmate_rate": r["checkmate_rate"], "castling_rate": r["castling_rate"],
                "avg_plies": r["avg_plies"],
            })

    summary = {
        "experiment": "Full_Round_Robin_Leaderboard_All_FullBoard_Levels",
        "total_games": len(records),
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "note_level6": "Level 6 (Afterstate TD(λ) + D4) chuyên biệt cờ tàn 3 quân (KQK 100%, KRK 98%, KPK 75%) nên đánh giá riêng ở Mục 4.4.",
        "leaderboard": rows,
        "pairwise_wdl": {f"{k[0]}__vs__{k[1]}": v for k, v in pair_wdl.items()},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_full_leaderboard(rows, pair_scores, fig_path)

    print("\n=== BẢNG XẾP HẠNG TỔNG SẮP (FULL LEADERBOARD) ===")
    for i, r in enumerate(rows, 1):
        print(f"  #{i} {r['agent']:20s} | {r['wins']:2d}W-{r['draws']:2d}D-{r['losses']:2d}L | {r['score_pct']:5.1f}% | Elo {r['elo']:6.1f} | {r['avg_latency_ms']:6.1f} ms")
    print(f"Hoàn tất trong {summary['elapsed_sec']}s -> {csv_path} & {fig_path}")


if __name__ == "__main__":
    main()
