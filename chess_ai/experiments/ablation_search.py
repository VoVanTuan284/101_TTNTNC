"""
chess_ai/experiments/ablation_search.py
Ablation 1: Đánh giá 5 cấu hình tìm kiếm cổ điển lũy tiến (Config_A -> Config_D)
và khảo sát độ nhạy theo độ sâu d in {2, 3, 4} trên bộ 50 thế cờ WAC (resources/benchmark_positions.epd).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import chess

from chess_ai.arena.metrics import compute_latency_stats, wilson_confidence_interval
from chess_ai.config import load_yaml_config
from chess_ai.search.alphabeta import search_alphabeta
from chess_ai.search.transposition import TranspositionTable
from chess_ai.utils.epd import EPDRecord, load_epd_file


@dataclass(frozen=True)
class AblationConfigSpec:
    """Đặc tả một cấu hình tìm kiếm trong Ablation 1."""

    name: str
    description: str
    depth: int
    use_alpha_beta: bool
    use_move_ordering: bool
    use_tt: bool
    use_quiescence: bool


def _evaluate_config(
    spec: AblationConfigSpec,
    records: List[EPDRecord],
) -> Dict[str, Any]:
    """Chạy một cấu hình tìm kiếm trên toàn bộ danh sách thế cờ EPD và tổng hợp số liệu."""
    solved_count = 0
    nodes_list: List[float] = []
    latencies_ms: List[float] = []
    tt_hits_list: List[float] = []
    details: List[Dict[str, Any]] = []

    tt = TranspositionTable(max_size_mb=16) if spec.use_tt else None

    for rec in records:
        if tt is not None:
            tt.clear()
        board = chess.Board(rec.fen)
        move, stats = search_alphabeta(
            board=board,
            max_depth=spec.depth,
            use_tt=spec.use_tt,
            use_move_ordering=spec.use_move_ordering,
            use_quiescence=spec.use_quiescence,
            use_alpha_beta=spec.use_alpha_beta,
            tt=tt,
        )
        is_solved = move in rec.best_moves
        if is_solved:
            solved_count += 1

        nodes = float(stats["nodes"])
        time_ms = float(stats["time_sec"]) * 1000.0
        tt_hits = float(stats.get("tt_hits", 0.0))

        nodes_list.append(nodes)
        latencies_ms.append(time_ms)
        tt_hits_list.append(tt_hits)
        details.append(
            {
                "id": rec.record_id,
                "found_move": move.uci(),
                "expected_moves": [m.uci() for m in rec.best_moves],
                "solved": is_solved,
                "nodes": int(nodes),
                "time_ms": round(time_ms, 3),
                "tt_hits": int(tt_hits),
            }
        )

    total = len(records)
    accuracy = solved_count / total if total > 0 else 0.0
    ci_low, ci_high = wilson_confidence_interval(accuracy, total) if total > 0 else (0.0, 0.0)
    lat_stats = compute_latency_stats(latencies_ms)
    total_nodes = sum(nodes_list)

    return {
        "name": spec.name,
        "description": spec.description,
        "depth": spec.depth,
        "flags": {
            "alpha_beta": spec.use_alpha_beta,
            "move_ordering": spec.use_move_ordering,
            "transposition_table": spec.use_tt,
            "quiescence": spec.use_quiescence,
        },
        "total_positions": total,
        "solved_count": solved_count,
        "tactical_accuracy": round(accuracy, 4),
        "wilson_ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "total_nodes": int(total_nodes),
        "avg_nodes": round(total_nodes / total, 2) if total > 0 else 0.0,
        "avg_tt_hits": round(sum(tt_hits_list) / total, 2) if total > 0 else 0.0,
        "avg_latency_ms": round(lat_stats["avg_ms"], 3),
        "p95_latency_ms": round(lat_stats["p95_ms"], 3),
        "positions": details,
    }


def run_ablation_search(
    epd_path: str = "resources/benchmark_positions.epd",
    output_path: str = "results/ablation1/ablation1_results.json",
    max_positions: Optional[int] = None,
    base_depth: int = 3,
    skip_minimax_if_large: bool = False,
) -> Dict[str, Any]:
    """Chạy thực nghiệm Ablation 1 và lưu kết quả JSON."""
    records = load_epd_file(epd_path)
    if max_positions is not None:
        records = records[:max_positions]

    search_cfg = load_yaml_config("search_params.yaml")
    ablation_depths: List[int] = [int(d) for d in search_cfg["alpha_beta"].get("ablation_depths", [2, 3, 4])]

    progressive_specs = [
        AblationConfigSpec("Config_A", "Minimax thuần (Negamax không cắt tỉa)", base_depth, False, False, False, False),
        AblationConfigSpec("Config_B", "Alpha-Beta Pruning", base_depth, True, False, False, False),
        AblationConfigSpec("Config_C1", "Alpha-Beta + Move Ordering (MVV-LVA, Killer, History)", base_depth, True, True, False, False),
        AblationConfigSpec("Config_C2", "Alpha-Beta + Move Ordering + Zobrist TT (ID)", base_depth, True, True, True, False),
        AblationConfigSpec("Config_D", "Cấp 3 đầy đủ (Config_C2 + Quiescence Search)", base_depth, True, True, True, True),
    ]

    part_1a: List[Dict[str, Any]] = []
    baseline_nodes: Optional[float] = None

    for spec in progressive_specs:
        if skip_minimax_if_large and spec.name == "Config_A":
            continue
        res = _evaluate_config(spec, records)
        if spec.name == "Config_A":
            baseline_nodes = res["avg_nodes"]
        if baseline_nodes and baseline_nodes > 0:
            reduction = (1.0 - res["avg_nodes"] / baseline_nodes) * 100.0
            res["node_reduction_vs_minimax_pct"] = round(reduction, 2)
        else:
            res["node_reduction_vs_minimax_pct"] = 0.0
        part_1a.append(res)

    part_1b: List[Dict[str, Any]] = []
    for d in ablation_depths:
        spec = AblationConfigSpec(
            name=f"Depth_{d}",
            description=f"Config_D (Cấp 3 đầy đủ) ở độ sâu d={d}",
            depth=d,
            use_alpha_beta=True,
            use_move_ordering=True,
            use_tt=True,
            use_quiescence=True,
        )
        part_1b.append(_evaluate_config(spec, records))

    payload: Dict[str, Any] = {
        "experiment": "Ablation_1_Classical_Search",
        "epd_path": epd_path,
        "num_positions": len(records),
        "part_1a_progressive_configs": part_1a,
        "part_1b_depth_sensitivity": part_1b,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm Ablation 1 (Tìm kiếm cổ điển trên WAC-50).")
    parser.add_argument("--epd", default="resources/benchmark_positions.epd", help="Đường dẫn tệp EPD benchmark.")
    parser.add_argument("--output", default="results/ablation1/ablation1_results.json", help="Đường dẫn tệp JSON đầu ra.")
    parser.add_argument("--max-positions", type=int, default=None, help="Giới hạn số thế cờ chạy thử nhanh.")
    parser.add_argument("--base-depth", type=int, default=3, help="Độ sâu cơ sở cho Phần 1A (mặc định: 3).")
    args = parser.parse_args()

    res = run_ablation_search(
        epd_path=args.epd,
        output_path=args.output,
        max_positions=args.max_positions,
        base_depth=args.base_depth,
    )
    print(f"Ablation 1 hoàn tất ({res['num_positions']} thế cờ). Lưu tại: {args.output}")
    print("\n--- Phần 1A: 5 Cấu hình lũy tiến ---")
    for row in res["part_1a_progressive_configs"]:
        print(
            f"{row['name']:10s} | Acc: {row['tactical_accuracy']*100:5.1f}% ({row['solved_count']}/{row['total_positions']}) "
            f"| Avg Nodes: {row['avg_nodes']:9.1f} | Giảm: {row['node_reduction_vs_minimax_pct']:6.2f}% "
            f"| Avg ms: {row['avg_latency_ms']:7.2f}"
        )
    print("\n--- Phần 1B: Độ nhạy độ sâu ---")
    for row in res["part_1b_depth_sensitivity"]:
        print(
            f"{row['name']:10s} | Acc: {row['tactical_accuracy']*100:5.1f}% ({row['solved_count']}/{row['total_positions']}) "
            f"| Avg Nodes: {row['avg_nodes']:9.1f} | p95 ms: {row['p95_latency_ms']:7.2f}"
        )


if __name__ == "__main__":
    main()
