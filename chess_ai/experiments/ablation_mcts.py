"""
chess_ai/experiments/ablation_mcts.py
Ablation 2:
- Phần A: Khảo sát độ nhạy ngân sách mô phỏng N_sim in {50, 100, 200, 400, 800, 1000}
  cho cả Cấp 4 (Vanilla MCTS UCT) và Cấp 7 (Guided MCTS PUCT).
- Phần B: Bóc tách đóng góp giữa Policy-only (Cấp 5), Value-only MCTS và Full Guided MCTS (Cấp 7).
Phụ trách: TV3 phối hợp TV2
"""


def run_ablation_mcts(output_path: str = "results/ablation2/ablation2_results.json") -> None:
    """Chạy thực nghiệm Ablation 2 và lưu kết quả JSON."""
    raise NotImplementedError


if __name__ == "__main__":
    pass

