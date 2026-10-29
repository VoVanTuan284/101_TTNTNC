"""
chess_ai/experiments/ablation_search.py
Ablation 1: Đánh giá 5 cấu hình tìm kiếm cổ điển lũy tiến (Config_A -> Config_D)
và khảo sát độ nhạy theo độ sâu d in {2, 3, 4} trên bộ 50 thế cờ WAC (resources/benchmark_positions.epd).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""


def run_ablation_search(
    epd_path: str = "resources/benchmark_positions.epd",
    output_path: str = "results/ablation1/ablation1_results.json",
) -> None:
    """Chạy thực nghiệm Ablation 1 và lưu kết quả JSON."""
    raise NotImplementedError


if __name__ == "__main__":
    pass

