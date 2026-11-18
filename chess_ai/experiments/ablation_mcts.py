"""
chess_ai/experiments/ablation_mcts.py
Ablation 2:
- Phần A: Khảo sát độ nhạy ngân sách mô phỏng N_sim in {50, 100, 200, 400, 800, 1000}
  cho cả Cấp 4 (Vanilla MCTS UCT) và Cấp 7 (Guided MCTS PUCT).
- Phần B: Bóc tách đóng góp giữa Policy-only (Cấp 5), Value-only MCTS và Full Guided MCTS (Cấp 7).
Phụ trách: TV3 phối hợp TV2
"""

import argparse
from functools import partial
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from chess_ai.agents.base_agent import BaseAgent
from chess_ai.agents.level2_heuristic import Level2HeuristicAgent
from chess_ai.agents.level4_mcts import Level4MCTSAgent
from chess_ai.agents.registry import create_agent
from chess_ai.arena.tournament import run_paired_matchup
from chess_ai.config import load_yaml_config
from chess_ai.core.chess_env import ChessEnv
from chess_ai.utils.epd import load_epd_file


def _make_level4(simulations: int) -> BaseAgent:
    return Level4MCTSAgent(simulations=simulations)


def _make_level2() -> BaseAgent:
    return Level2HeuristicAgent()


def _make_level7(simulations: int, checkpoint_path: Optional[str]) -> BaseAgent:
    return create_agent("level7", checkpoint_path=checkpoint_path, simulations=simulations, eval_mode=True)


def _is_agent_implemented(agent: BaseAgent) -> bool:
    """Kiểm tra nhanh xem tác tử đã được cài đặt thực tế (không ném NotImplementedError) hay chưa."""
    try:
        agent.select_move(ChessEnv(), max_workload=2)
        return True
    except NotImplementedError:
        return False
    except Exception:
        return False


def run_ablation_mcts(
    output_path: str = "results/ablation2/ablation2_results.json",
    config_path: str = "configs/default_config.yaml",
    sim_budgets: Optional[List[int]] = None,
    num_openings_2a: int = 25,
    num_openings_2b: int = 50,
    checkpoint_path: Optional[str] = "checkpoints/best_model.pt",
    workers: int = 1,
    include_puct: bool = True,
) -> Dict[str, Any]:
    """Chạy thực nghiệm Ablation 2 và lưu kết quả JSON."""
    cfg = load_yaml_config(config_path)
    search_cfg = load_yaml_config("search_params.yaml")
    openings_all = load_epd_file(cfg["paths"]["openings_epd"])
    seeds: List[int] = [int(s) for s in cfg.get("agent_seeds", [42])]
    max_plies = int(cfg["arena"].get("max_plies", 400))
    bootstrap_resamples = int(cfg["arena"].get("bootstrap_resamples", 10000))

    if sim_budgets is None:
        sim_budgets = [int(n) for n in search_cfg["vanilla_mcts"].get("ablation_simulations", [50, 100, 200, 400, 800, 1000])]

    openings_2a = openings_all[:num_openings_2a]
    openings_2b = openings_all[:num_openings_2b]

    # --- Phần 2A: Độ nhạy số lượt mô phỏng cho Cấp 4 (UCT) ---
    uct_curve: List[Dict[str, Any]] = []
    opponent = _make_level2()
    for n_sim in sim_budgets:
        agent_l4 = _make_level4(n_sim)
        res = run_paired_matchup(
            agent_a=agent_l4,
            agent_b=opponent,
            openings=openings_2a,
            seeds=seeds,
            budget_mode="workload_control",
            workload_a=n_sim,
            workload_b=None,
            max_plies=max_plies,
            bootstrap_resamples=bootstrap_resamples,
            workers=workers,
            factory_a=partial(_make_level4, n_sim),
            factory_b=_make_level2,
        )
        res["simulations"] = n_sim
        uct_curve.append(res)

    # --- Phần 2A: Độ nhạy số lượt mô phỏng cho Cấp 7 (PUCT) ---
    puct_section: Dict[str, Any] = {"status": "skipped", "curve": []}
    if include_puct:
        try:
            probe_l7 = _make_level7(sim_budgets[0], checkpoint_path)
            if _is_agent_implemented(probe_l7):
                puct_curve: List[Dict[str, Any]] = []
                for n_sim in sim_budgets:
                    agent_l7 = _make_level7(n_sim, checkpoint_path)
                    res = run_paired_matchup(
                        agent_a=agent_l7,
                        agent_b=opponent,
                        openings=openings_2a,
                        seeds=seeds,
                        budget_mode="workload_control",
                        workload_a=n_sim,
                        workload_b=None,
                        max_plies=max_plies,
                        bootstrap_resamples=bootstrap_resamples,
                        workers=workers,
                        factory_a=partial(_make_level7, n_sim, checkpoint_path),
                        factory_b=_make_level2,
                    )
                    res["simulations"] = n_sim
                    puct_curve.append(res)
                puct_section = {"status": "completed", "curve": puct_curve}
            else:
                puct_section = {
                    "status": "pending_tv2_level7",
                    "reason": "Level7GuidedMCTSAgent hoặc checkpoint chưa sẵn sàng.",
                    "curve": [],
                }
        except Exception as exc:
            puct_section = {
                "status": "pending_tv2_level7",
                "reason": str(exc),
                "curve": [],
            }

    # --- Phần 2B: Bóc tách Policy-only (Cấp 5) vs Value-only MCTS vs Full Guided MCTS (Cấp 7) ---
    part_2b: Dict[str, Any] = {"status": "pending_tv2_level5_level7", "matchups": []}
    if include_puct:
        try:
            agent_l5 = create_agent("level5", checkpoint_path=checkpoint_path)
            agent_l7 = create_agent("level7", checkpoint_path=checkpoint_path, simulations=400, eval_mode=True)
            if _is_agent_implemented(agent_l5) and _is_agent_implemented(agent_l7):
                m_l7_vs_l5 = run_paired_matchup(
                    agent_a=agent_l7,
                    agent_b=agent_l5,
                    openings=openings_2b,
                    seeds=seeds,
                    budget_mode="workload_control",
                    workload_a=400,
                    workload_b=None,
                    max_plies=max_plies,
                    bootstrap_resamples=bootstrap_resamples,
                )
                part_2b = {"status": "completed", "matchups": [m_l7_vs_l5]}
        except Exception as exc:
            part_2b = {"status": "pending_tv2_level5_level7", "reason": str(exc), "matchups": []}

    payload: Dict[str, Any] = {
        "experiment": "Ablation_2_MCTS_Sensitivity_And_Guidance",
        "sim_budgets": sim_budgets,
        "games_per_budget_2a": len(openings_2a) * 2,
        "opponent_2a": opponent.name,
        "part_2a_uct_level4": uct_curve,
        "part_2a_puct_level7": puct_section,
        "part_2b_guidance_ablation": part_2b,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm Ablation 2 (Độ nhạy N_sim MCTS & Dẫn hướng nơ-ron).")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Đường dẫn cấu hình YAML.")
    parser.add_argument("--output", default="results/ablation2/ablation2_results.json", help="Đường dẫn tệp JSON đầu ra.")
    parser.add_argument("--budgets", nargs="+", type=int, default=None, help="Danh sách N_sim (ví dụ: 50 100 200 400 800 1000).")
    parser.add_argument("--num-openings-2a", type=int, default=25, help="Số khai cuộc cho Phần 2A (mặc định 25 = 50 ván/mức).")
    parser.add_argument("--num-openings-2b", type=int, default=50, help="Số khai cuộc cho Phần 2B (mặc định 50 = 100 ván/cặp).")
    parser.add_argument("--workers", type=int, default=1, help="Số tiến trình multiprocessing.")
    parser.add_argument("--no-puct", action="store_true", help="Bỏ qua phần Cấp 5/7 nếu TV2 chưa hoàn thiện.")
    args = parser.parse_args()

    res = run_ablation_mcts(
        output_path=args.output,
        config_path=args.config,
        sim_budgets=args.budgets,
        num_openings_2a=args.num_openings_2a,
        num_openings_2b=args.num_openings_2b,
        workers=args.workers,
        include_puct=not args.no_puct,
    )
    print(f"Ablation 2 hoàn tất. Lưu tại: {args.output}")
    print("\n--- Phần 2A: Cấp 4 (UCT) vs Level2_Heuristic ---")
    for row in res["part_2a_uct_level4"]:
        print(
            f"N_sim={row['simulations']:4d} | S={row['score']:.3f} "
            f"(W/D/L={row['wins']}/{row['draws']}/{row['losses']}) "
            f"| Wilson95={row['wilson_ci_95']} | L4 avg ms={row['agent_a_avg_latency_ms']:.1f}"
        )
    print(f"\nPhần 2A (PUCT Cấp 7): {res['part_2a_puct_level7']['status']}")
    print(f"Phần 2B (Policy vs Value vs PUCT): {res['part_2b_guidance_ablation']['status']}")


if __name__ == "__main__":
    main()
