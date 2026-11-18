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


def _plot_ablation1(data: Dict[str, Any], out_dir: Path) -> None:
    """Vẽ biểu đồ Ablation 1: Số nút duyệt vs Tactical Accuracy và Độ nhạy độ sâu."""
    part_1a = data.get("part_1a_progressive_configs", [])
    if part_1a:
        names = [r["name"] for r in part_1a]
        nodes = [r["avg_nodes"] for r in part_1a]
        accs = [r["tactical_accuracy"] * 100.0 for r in part_1a]

        fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=150)
        bars = ax1.bar(names, nodes, color="#4C78A8", alpha=0.85, label="Avg Nodes")
        ax1.set_ylabel("Số nút duyệt trung bình (Nodes)", color="#4C78A8")
        ax1.set_title("Ablation 1A: Hiệu quả Alpha-Beta, Ordering, TT & Quiescence (WAC-50)")
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        ax2 = ax1.twinx()
        ax2.plot(names, accs, color="#E45756", marker="o", linewidth=2, label="Tactical Accuracy (%)")
        ax2.set_ylabel("Tactical Accuracy (%)", color="#E45756")
        ax2.set_ylim(0, 100)

        for b, v in zip(bars, nodes):
            ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / "ablation1_progressive.png")
        plt.close(fig)

    part_1b = data.get("part_1b_depth_sensitivity", [])
    if part_1b:
        depths = [f"d={r['depth']}" for r in part_1b]
        accs = [r["tactical_accuracy"] * 100.0 for r in part_1b]
        p95 = [r["p95_latency_ms"] for r in part_1b]

        fig, ax1 = plt.subplots(figsize=(7, 4), dpi=150)
        ax1.plot(depths, accs, marker="s", color="#54A24B", linewidth=2, label="Accuracy (%)")
        ax1.set_ylabel("Tactical Accuracy (%)", color="#54A24B")
        ax1.set_ylim(0, 100)
        ax1.set_title("Ablation 1B: Độ nhạy theo độ sâu tìm kiếm (Config_D)")
        ax1.grid(True, linestyle="--", alpha=0.4)

        ax2 = ax1.twinx()
        ax2.plot(depths, p95, marker="^", linestyle="--", color="#F58518", linewidth=2, label="p95 Latency (ms)")
        ax2.set_ylabel("Độ trễ p95 (ms)", color="#F58518")

        fig.tight_layout()
        fig.savefig(out_dir / "ablation1_depth_sensitivity.png")
        plt.close(fig)


def _plot_ablation2(data: Dict[str, Any], out_dir: Path) -> None:
    """Vẽ đường cong hiệu năng MCTS theo N_sim (Ablation 2A)."""
    uct_curve = data.get("part_2a_uct_level4", [])
    puct_curve = data.get("part_2a_puct_level7", {}).get("curve", [])
    if not uct_curve and not puct_curve:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    if uct_curve:
        xs = [r["simulations"] for r in uct_curve]
        ys = [r["score"] * 100.0 for r in uct_curve]
        ax.plot(xs, ys, marker="o", linewidth=2, color="#4C78A8", label="Cấp 4: Vanilla MCTS (UCT)")
    if puct_curve:
        xs = [r["simulations"] for r in puct_curve]
        ys = [r["score"] * 100.0 for r in puct_curve]
        ax.plot(xs, ys, marker="s", linewidth=2, color="#E45756", label="Cấp 7: Guided MCTS (PUCT)")

    ax.set_xlabel("Ngân sách mô phỏng (N_sim)")
    ax.set_ylabel("Điểm số quy đổi S (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Ablation 2A: Đường cong hiệu năng MCTS theo ngân sách mô phỏng")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ablation2_mcts_sensitivity.png")
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
