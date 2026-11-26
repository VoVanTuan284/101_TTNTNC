"""
scripts/reproduce_experiment.py
Kịch bản tái lập thực nghiệm tự động hỗ trợ 2 chế độ:
- --from-logs: Tái tạo toàn bộ bảng kết quả, Wilson/Bootstrap CI và biểu đồ từ dữ liệu results/ (< 30s).
- --full: Chạy lại từ đầu toàn bộ Ablation 1, Ablation 2 và giải đấu Round-Robin 100 ván/cặp.
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chess_ai.agents.registry import create_agent, list_full_board_agents
from chess_ai.arena.tournament import run_round_robin_tournament
from chess_ai.config import load_yaml_config
from chess_ai.experiments.ablation_mcts import run_ablation_mcts
from chess_ai.experiments.ablation_search import run_ablation_search
import numpy as np


def _apply_dark_style(fig: Any, axes: List[Any]) -> None:
    """Áp dụng bảng màu Dark Slate đồng bộ với giao diện Pygame."""
    fig.patch.set_facecolor("#101622")
    for ax in axes:
        ax.set_facecolor("#161E2E")
        ax.tick_params(colors="#E2E8F0", labelsize=9)
        ax.xaxis.label.set_color("#E2E8F0")
        ax.yaxis.label.set_color("#E2E8F0")
        ax.title.set_color("#F8FAFC")
        for spine in ax.spines.values():
            spine.set_color("#334155")


def _plot_ablation1(data: Dict[str, Any], out_dir: Path) -> None:
    """Vẽ biểu đồ Ablation 1: Số nút duyệt vs Tactical Accuracy và Độ nhạy độ sâu."""
    part_1a = data.get("part_1a_progressive_configs", [])
    if part_1a:
        names = [r["name"] for r in part_1a]
        nodes = [r["avg_nodes"] for r in part_1a]
        accs = [r["tactical_accuracy"] * 100.0 for r in part_1a]

        fig, ax1 = plt.subplots(figsize=(8.6, 4.4), dpi=130)
        ax2 = ax1.twinx()
        _apply_dark_style(fig, [ax1, ax2])

        bars = ax1.bar(names, nodes, color="#3B82F6", alpha=0.88, width=0.52, label="Avg Nodes")
        ax1.set_ylabel("Số nút duyệt trung bình (Nodes)", color="#60A5FA", fontweight="bold")
        ax1.set_title("Ablation 1A: Hiệu quả Alpha-Beta, Ordering, TT & Quiescence (WAC-50)", fontweight="bold")
        ax1.grid(axis="y", linestyle="--", color="#334155", alpha=0.6)

        ax2.plot(names, accs, color="#4ADE80", marker="o", markersize=7, linewidth=2.5, label="Tactical Accuracy (%)")
        ax2.set_ylabel("Tactical Accuracy (%)", color="#4ADE80", fontweight="bold")
        ax2.set_ylim(0, 105)

        for b, v in zip(bars, nodes):
            ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8.5, color="#E2E8F0")
        for x_i, acc_v in enumerate(accs):
            ax2.text(x_i, acc_v + 3.0, f"{acc_v:.1f}%", ha="center", va="bottom", fontsize=8.5, color="#4ADE80", fontweight="bold")

        fig.tight_layout()
        fig.savefig(out_dir / "ablation1_progressive.png", facecolor=fig.get_facecolor())
        plt.close(fig)

    part_1b = data.get("part_1b_depth_sensitivity", [])
    if part_1b:
        depths = [f"d={r['depth']}" for r in part_1b]
        accs = [r["tactical_accuracy"] * 100.0 for r in part_1b]
        p95 = [r["p95_latency_ms"] for r in part_1b]

        fig, ax1 = plt.subplots(figsize=(8.2, 4.2), dpi=130)
        ax2 = ax1.twinx()
        _apply_dark_style(fig, [ax1, ax2])

        ax1.plot(depths, accs, marker="s", markersize=7, color="#4ADE80", linewidth=2.5, label="Accuracy (%)")
        ax1.set_ylabel("Tactical Accuracy (%)", color="#4ADE80", fontweight="bold")
        ax1.set_ylim(0, 105)
        ax1.set_title("Ablation 1B: Độ nhạy theo độ sâu tìm kiếm (Config_E)", fontweight="bold")
        ax1.grid(True, linestyle="--", color="#334155", alpha=0.6)

        ax2.plot(depths, p95, marker="^", markersize=7, linestyle="--", color="#F59E0B", linewidth=2.5, label="p95 Latency (ms)")
        ax2.set_ylabel("Độ trễ p95 (ms)", color="#F59E0B", fontweight="bold")

        fig.tight_layout()
        fig.savefig(out_dir / "ablation1_depth_sensitivity.png", facecolor=fig.get_facecolor())
        plt.close(fig)


def _plot_ablation2(data: Dict[str, Any], out_dir: Path) -> None:
    """Vẽ đường cong hiệu năng & độ trễ MCTS theo N_sim (Ablation 2A)."""
    uct_curve = data.get("part_2a_uct_level4", [])
    puct_curve = data.get("part_2a_puct_level7", {}).get("curve", [])
    if not uct_curve and not puct_curve:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 4.1), dpi=130)
    _apply_dark_style(fig, [ax1, ax2])

    if uct_curve:
        xs = [r["simulations"] for r in uct_curve]
        ys = [r["score"] * 100.0 for r in uct_curve]
        lats = [r.get("agent_a_avg_latency_ms", 0.0) for r in uct_curve]
        ci_low = [r.get("wilson_ci_95", [0.0, 0.0])[0] * 100.0 for r in uct_curve]
        ci_high = [r.get("wilson_ci_95", [0.0, 0.0])[1] * 100.0 for r in uct_curve]
        ax1.plot(xs, ys, marker="o", linewidth=2.4, color="#3B82F6", label="Cấp 4: Vanilla MCTS (UCT)")
        ax1.fill_between(xs, ci_low, ci_high, color="#3B82F6", alpha=0.2, label="Wilson 95% CI")
        ax2.bar([str(x) for x in xs], lats, color="#F59E0B", alpha=0.88, width=0.55, label="Avg Latency (ms)")
        for idx_b, lv in enumerate(lats):
            ax2.text(idx_b, lv, f"{lv:.0f}ms", ha="center", va="bottom", fontsize=8, color="#E2E8F0")

    if puct_curve:
        xs = [r["simulations"] for r in puct_curve]
        ys = [r["score"] * 100.0 for r in puct_curve]
        ax1.plot(xs, ys, marker="s", linewidth=2.4, color="#4ADE80", label="Cấp 7: Guided MCTS (PUCT)")

    ax1.set_xlabel("Ngân sách mô phỏng (N_sim)")
    ax1.set_ylabel("Điểm số quy đổi S (%)")
    ax1.set_ylim(-2, 105)
    ax1.set_title("Điểm S (%) & Wilson 95% CI", fontweight="bold", fontsize=11)
    ax1.grid(True, linestyle="--", color="#334155", alpha=0.6)
    ax1.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0", fontsize=8)

    ax2.set_xlabel("Ngân sách mô phỏng (N_sim)")
    ax2.set_ylabel("Độ trễ trung bình (ms)")
    ax2.set_title("Độ trễ suy nghĩ TB theo N_sim", fontweight="bold", fontsize=11)
    ax2.grid(axis="y", linestyle="--", color="#334155", alpha=0.6)

    fig.suptitle("Ablation 2: Khảo sát độ nhạy Ngân sách Mô phỏng MCTS", color="#F8FAFC", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "ablation2_mcts_sensitivity.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_rl_krk(data: Dict[str, Any], out_dir: Path) -> None:
    """Vẽ đường cong học tập (Win Rate & Avg Plies) của Cấp 6 RL trên KQK, KRK, KPK."""
    curves = data.get("tabular_q_curves", {})
    if not curves:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 4.1), dpi=130)
    _apply_dark_style(fig, [ax1, ax2])
    palette = {"KQK": "#4ADE80", "KRK": "#3B82F6", "KPK": "#F59E0B"}

    for eg_name, pts in curves.items():
        if not pts:
            continue
        xs = [p["episode"] for p in pts]
        wrs = [p["win_rate"] * 100.0 for p in pts]
        plies = [p["avg_plies"] for p in pts]
        c = palette.get(eg_name, "#E2E8F0")
        ax1.plot(xs, wrs, marker="o", markersize=4, linewidth=2.2, color=c, label=f"{eg_name} Win Rate (%)")
        ax2.plot(xs, plies, marker="s", markersize=4, linewidth=2.0, linestyle="--", color=c, label=f"{eg_name} Avg Plies")

    ax1.set_xlabel("Số Episodes huấn luyện")
    ax1.set_ylabel("Tỷ lệ Thắng - Win Rate (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Đường cong Hội tụ Tỷ lệ Thắng (%)", fontweight="bold", fontsize=11)
    ax1.grid(True, linestyle="--", color="#334155", alpha=0.6)
    ax1.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0", fontsize=8.5)

    ax2.set_xlabel("Số Episodes huấn luyện")
    ax2.set_ylabel("Số nước đi trung bình (Plies)")
    ax2.set_title("Số Plies Trung bình mỗi Episode", fontweight="bold", fontsize=11)
    ax2.grid(True, linestyle="--", color="#334155", alpha=0.6)
    ax2.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0", fontsize=8.5)

    fig.suptitle("Cấp 6 RL: Đường cong Học tăng cường Cờ tàn (KQK / KRK / KPK)", color="#F8FAFC", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "rl_krk_learning_curve.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_arena_matchup(data: Dict[str, Any], out_dir: Path) -> None:
    """Vẽ biểu đồ kết quả đấu cặp Arena (W/D/L, Điểm S kèm Wilson & Bootstrap 95% CI, Latency)."""
    if not data or "agent_a" not in data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 4.0), dpi=130)
    _apply_dark_style(fig, [ax1, ax2])

    a_name = data.get("agent_a", "Agent A")
    b_name = data.get("agent_b", "Agent B")
    wins = int(data.get("wins", 0))
    draws = int(data.get("draws", 0))
    losses = int(data.get("losses", 0))
    score = float(data.get("score", 0.0)) * 100.0
    w_score = float(data.get("white_score", 0.0)) * 100.0
    b_score = float(data.get("black_score", 0.0)) * 100.0
    w_ci = data.get("wilson_ci_95", [0.0, 0.0])

    # Biểu đồ 1: Cột kết quả Thắng / Hòa / Thua
    cats = ["Thắng (W)", "Hòa (D)", "Thua (L)"]
    counts = [wins, draws, losses]
    colors = ["#4ADE80", "#F59E0B", "#EF4444"]
    bars = ax1.bar(cats, counts, color=colors, width=0.52, alpha=0.9)
    for b, cnt in zip(bars, counts):
        ax1.text(b.get_x() + b.get_width() / 2, cnt, f"{cnt}", ha="center", va="bottom", color="#F8FAFC", fontweight="bold")
    ax1.set_ylabel("Số ván đấu")
    ax1.set_title(f"Phân bố W/D/L ({a_name})", fontweight="bold", fontsize=11)
    ax1.grid(axis="y", linestyle="--", color="#334155", alpha=0.6)

    # Biểu đồ 2: Điểm S tổng, cầm Trắng, cầm Đen & khoảng tin cậy 95%
    metrics_x = ["Tổng điểm S", "Khi cầm Trắng", "Khi cầm Đen"]
    metrics_y = [score, w_score, b_score]
    m_bars = ax2.bar(metrics_x, metrics_y, color=["#3B82F6", "#60A5FA", "#818CF8"], width=0.48, alpha=0.9)
    err_low = max(0.0, score - w_ci[0] * 100.0)
    err_high = max(0.0, w_ci[1] * 100.0 - score)
    ax2.errorbar([0], [score], yerr=[[err_low], [err_high]], fmt="none", ecolor="#F59E0B", elinewidth=2.2, capsize=6, label="Wilson 95% CI")
    ax2.axhline(50.0, color="#94A3B8", linestyle="--", linewidth=1.2, label="Cân bằng (50%)")
    for b, val in zip(m_bars, metrics_y):
        ax2.text(b.get_x() + b.get_width() / 2, val + 2.0, f"{val:.1f}%", ha="center", va="bottom", color="#F8FAFC", fontsize=8.5, fontweight="bold")
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("Tỷ lệ điểm quy đổi (%)")
    ax2.set_title("Điểm S (%) & Khoảng tin cậy 95%", fontweight="bold", fontsize=11)
    ax2.grid(axis="y", linestyle="--", color="#334155", alpha=0.6)
    ax2.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0", fontsize=8)

    fig.suptitle(f"Arena: {a_name} vs {b_name} ({data.get('games', 0)} ván)", color="#F8FAFC", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "arena_matchup.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_sl_training(data: Dict[str, Any], out_dir: Path) -> None:
    """Vẽ đường cong huấn luyện Level 5 (Loss & Top-1/Top-3 Accuracy)."""
    hist = data.get("history", [])
    if not hist:
        return
    epochs = [h["epoch"] for h in hist]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.1), dpi=130)
    _apply_dark_style(fig, [ax1, ax2])

    ax1.plot(epochs, [h["train_total_loss"] for h in hist], marker="o", markersize=3.5, color="#3B82F6", label="Train Total Loss")
    ax1.plot(epochs, [h["val_total_loss"] for h in hist], marker="s", markersize=3.5, color="#F59E0B", label="Val Total Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss")
    ax1.set_title("Level 5: Đường cong Total Loss", fontweight="bold", fontsize=11)
    ax1.grid(True, linestyle="--", color="#334155", alpha=0.6)
    ax1.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0", fontsize=8.5)

    ax2.plot(epochs, [h["val_top1_policy_acc"] * 100.0 for h in hist], marker="o", markersize=3.5, color="#4ADE80", label="Val Top-1 Acc (%)")
    ax2.plot(epochs, [h["val_top3_policy_acc"] * 100.0 for h in hist], marker="s", markersize=3.5, color="#60A5FA", label="Val Top-3 Acc (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Level 5: Policy Accuracy trên Validation", fontweight="bold", fontsize=11)
    ax2.grid(True, linestyle="--", color="#334155", alpha=0.6)
    ax2.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0", fontsize=8.5)

    fig.suptitle("Cấp 5 Supervised Dual-Head ResNet (1M mẫu)", color="#F8FAFC", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_level5_training_curves.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_selfplay_gating(data: Dict[str, Any], out_dir: Path) -> None:
    """Vẽ biểu đồ Arena Gating Score & Loss của Level 8 Self-Play."""
    hist = data.get("history", [])
    if not hist:
        return
    iters = [h["iteration"] for h in hist]
    scores = [h["arena"]["score_rate"] * 100.0 for h in hist]
    cis = [h["arena"].get("wilson_ci_95", [h["arena"]["score_rate"], h["arena"]["score_rate"]]) for h in hist]
    err_low = [max(0.0, s - ci[0] * 100.0) for s, ci in zip(scores, cis)]
    err_high = [max(0.0, ci[1] * 100.0 - s) for s, ci in zip(scores, cis)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.1), dpi=130)
    _apply_dark_style(fig, [ax1, ax2])

    ax1.errorbar(iters, scores, yerr=[err_low, err_high], fmt="o-", color="#3B82F6", ecolor="#94A3B8", capsize=5, linewidth=2.2, label="Arena Score (%)")
    ax1.axhline(55.0, color="#EF4444", linestyle="--", linewidth=1.5, label="Ngưỡng Gating (55%)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Arena Score (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Arena Gating & Wilson 95% CI", fontweight="bold", fontsize=11)
    ax1.grid(True, linestyle="--", color="#334155", alpha=0.6)
    ax1.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0", fontsize=8.5)

    ax2.plot(iters, [h["train_total_loss"] for h in hist], marker="o", color="#F59E0B", linewidth=2.2, label="Total Loss")
    ax2.plot(iters, [h["train_policy_loss"] for h in hist], marker="s", color="#4ADE80", linewidth=2.0, label="Policy Loss")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss")
    ax2.set_title("Refinement Loss theo Iteration", fontweight="bold", fontsize=11)
    ax2.grid(True, linestyle="--", color="#334155", alpha=0.6)
    ax2.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0", fontsize=8.5)

    fig.suptitle(f"Cấp 8 AlphaZero Self-Play (Champion v{data.get('champion_version', 0)})", color="#F8FAFC", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_level8_selfplay_gating.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_round_robin_7x7(data: Dict[str, Any], out_dir: Path) -> None:
    """Vẽ ma trận đối đầu 7x7 và xếp hạng Elo từ summary_7x7_200.json."""
    lb = data.get("leaderboard", [])
    pairwise = data.get("pairwise_wdl", {})
    if not lb:
        return
    order = [r["agent"] for r in lb]
    n = len(order)
    mat = np.full((n, n), np.nan)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            key = f"{a}__vs__{b}"
            if i != j and key in pairwise:
                w, d, l = pairwise[key]
                tot = w + d + l
                if tot > 0:
                    mat[i, j] = (w + 0.5 * d) * 100.0 / tot

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=130)
    _apply_dark_style(fig, [ax1, ax2])

    im = ax1.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100)
    short = [x.replace("Level", "L").replace("_", " ") for x in order]
    ax1.set_xticks(range(n)); ax1.set_xticklabels(short, rotation=26, ha="right", fontsize=8.5)
    ax1.set_yticks(range(n)); ax1.set_yticklabels(short, fontsize=8.5)
    for i in range(n):
        for j in range(n):
            txt = "—" if np.isnan(mat[i, j]) else f"{mat[i, j]:.1f}%"
            ax1.text(j, i, txt, ha="center", va="center", fontsize=8.0, fontweight="bold", color="#0F172A" if not np.isnan(mat[i, j]) else "#E2E8F0")
    ax1.set_title("Ma trận Round-Robin 7×7 (200 ván/cặp = 4.200 ván)", fontweight="bold", fontsize=11)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    y_pos = np.arange(n)[::-1]
    elos = [r["elo"] for r in lb]
    scrs = [r["score_pct"] for r in lb]
    colors = ["#4ADE80" if e >= 1300 else ("#3B82F6" if e >= 1100 else "#94A3B8") for e in elos]
    bars = ax2.barh(y_pos, elos, color=colors, height=0.58)
    ax2.set_yticks(y_pos); ax2.set_yticklabels(short, fontsize=8.8)
    ax2.axvline(1200, color="#EF4444", linestyle="--", linewidth=1.3, label="Elo cơ sở (1200)")
    ax2.set_xlabel("Elo ước lượng (1.200 ván / tác tử)")
    ax2.set_title("Xếp hạng Elo Round-Robin 7×7", fontweight="bold", fontsize=11)
    ax2.set_xlim(min(elos) - 100, max(elos) + 230)
    ax2.grid(True, axis="x", linestyle="--", color="#334155", alpha=0.6)
    ax2.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0", loc="lower right", fontsize=8.5)
    for b, e, s in zip(bars, elos, scrs):
        ax2.text(e + 12, b.get_y() + b.get_height() / 2, f"{e:.0f} Elo ({s:.1f}%)", va="center", fontsize=8.4, fontweight="bold", color="#E2E8F0")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_round_robin_7x7_200.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_figures_from_logs(output_dir: str = "results/figures/") -> List[str]:
    """Đọc các tệp JSON kết quả đã lưu để tái tạo biểu đồ trong results/figures/."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []

    ab1_path = Path("results/ablation1/ablation1_results.json")
    if ab1_path.exists():
        ab1 = json.loads(ab1_path.read_text(encoding="utf-8"))
        _plot_ablation1(ab1, out_path)
        generated.extend(["ablation1_progressive.png", "ablation1_depth_sensitivity.png"])

    ab2_path = Path("results/ablation2/ablation2_results.json")
    if ab2_path.exists():
        ab2 = json.loads(ab2_path.read_text(encoding="utf-8"))
        _plot_ablation2(ab2, out_path)
        if ab2.get("part_2a_uct_level4") or ab2.get("part_2a_puct_level7", {}).get("curve"):
            generated.append("ablation2_mcts_sensitivity.png")

    rl_path = Path("results/training/krk_learning_curve.json")
    if rl_path.exists():
        rl_data = json.loads(rl_path.read_text(encoding="utf-8"))
        _plot_rl_krk(rl_data, out_path)
        generated.append("rl_krk_learning_curve.png")

    sl_path = Path("results/training/sl_training_metrics.json")
    if sl_path.exists():
        sl_data = json.loads(sl_path.read_text(encoding="utf-8"))
        _plot_sl_training(sl_data, out_path)
        generated.append("fig_level5_training_curves.png")

    sp_path = Path("results/training/selfplay_metrics.json")
    if sp_path.exists():
        sp_data = json.loads(sp_path.read_text(encoding="utf-8"))
        _plot_selfplay_gating(sp_data, out_path)
        generated.append("fig_level8_selfplay_gating.png")

    rr7_path = Path("results/round_robin_200/summary_7x7_200.json")
    if rr7_path.exists():
        rr7_data = json.loads(rr7_path.read_text(encoding="utf-8"))
        _plot_round_robin_7x7(rr7_data, out_path)
        generated.append("fig_round_robin_7x7_200.png")

    arena_path = Path("results/tournament_summary.json")
    if arena_path.exists():
        arena_data = json.loads(arena_path.read_text(encoding="utf-8"))
        if "agent_a" in arena_data:
            _plot_arena_matchup(arena_data, out_path)
            generated.append("arena_matchup.png")
        elif arena_data.get("matchups"):
            _plot_arena_matchup(arena_data["matchups"][0], out_path)
            generated.append("arena_matchup.png")

    return generated


def main() -> None:
    """Điểm vào CLI của kịch bản tái lập thực nghiệm."""
    parser = argparse.ArgumentParser(description="Tái lập thực nghiệm 101_TTNTNC (--from-logs hoặc --full).")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--from-logs", action="store_true", help="Tái tạo bảng & biểu đồ từ log sẵn có (<30s).")
    mode_group.add_argument("--full", action="store_true", help="Chạy lại toàn bộ Ablation 1, Ablation 2 và Arena.")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Đường dẫn cấu hình YAML.")
    parser.add_argument("--output-dir", default="results/figures/", help="Thư mục xuất biểu đồ.")
    parser.add_argument("--workers", type=int, default=1, help="Số tiến trình song song khi chạy --full.")
    args = parser.parse_args()

    if args.full:
        cfg = load_yaml_config(args.config)
        run_ablation_search(epd_path=cfg["paths"]["wac_epd"])
        run_ablation_mcts(config_path=args.config, workers=args.workers)
        agents = [create_agent(aid) for aid in list_full_board_agents()]
        run_round_robin_tournament(agents=agents, config_path=args.config, budget_mode="workload_control", workers=args.workers)

    generated = generate_figures_from_logs(args.output_dir)
    print(f"Đã tái tạo {len(generated)} biểu đồ tại {args.output_dir}: {generated}")


if __name__ == "__main__":
    main()
