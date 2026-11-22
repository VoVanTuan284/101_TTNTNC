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

from chess_ai.agents.registry import create_agent
from chess_ai.arena.tournament import run_round_robin_tournament
from chess_ai.config import load_yaml_config
from chess_ai.experiments.ablation_mcts import run_ablation_mcts
from chess_ai.experiments.ablation_search import run_ablation_search


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
        ax1.set_title("Ablation 1B: Độ nhạy theo độ sâu tìm kiếm (Config_D)", fontweight="bold")
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

    arena_path = Path("results/tournament_summary.json")
    if arena_path.exists():
        arena_data = json.loads(arena_path.read_text(encoding="utf-8"))
        if "agent_a" in arena_data:
            _plot_arena_matchup(arena_data, out_path)
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
    args = parser.parse_args()

    if args.full:
        cfg = load_yaml_config(args.config)
        run_ablation_search(epd_path=cfg["paths"]["wac_epd"])
        run_ablation_mcts(config_path=args.config)
        agents = [create_agent(aid) for aid in ("level1", "level2", "level3", "level4")]
        run_round_robin_tournament(agents=agents, config_path=args.config, budget_mode="workload_control")

    generated = generate_figures_from_logs(args.output_dir)
    print(f"Đã tái tạo {len(generated)} biểu đồ tại {args.output_dir}: {generated}")


if __name__ == "__main__":
    main()
