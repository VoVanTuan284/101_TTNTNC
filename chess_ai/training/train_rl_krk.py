"""
chess_ai/training/train_rl_krk.py
Kịch bản huấn luyện tác tử Học tăng cường Cấp 6 (Tabular Q-Learning & Policy Gradient)
trên bài toán cờ tàn chuẩn FIDE KRK (Vua + Xe đấu Vua), lưu bảng Q ra checkpoints/l6_krk_qtable.pkl
và xuất dữ liệu đường cong học tập ra results/training/krk_learning_curve.json.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""


def train_krk_rl(
    config_path: str = "configs/search_params.yaml",
    output_qtable: str = "checkpoints/l6_krk_qtable.pkl",
    output_metrics: str = "results/training/krk_learning_curve.json",
) -> None:
    """Thực thi huấn luyện RL trên cờ tàn KRK và lưu bảng Q + Learning Curve."""
    raise NotImplementedError


def main() -> None:
    train_krk_rl()


if __name__ == "__main__":
    main()

