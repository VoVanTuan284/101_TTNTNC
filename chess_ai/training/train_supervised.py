"""
chess_ai/training/train_supervised.py
Kịch bản huấn luyện có giám sát mạng Dual-Head ResNet (Cấp 5)
tích hợp Automatic Mixed Precision (AMP FP16) và CheckpointManager hỗ trợ --resume.
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""


def train_supervised(config_path: str = "configs/model_resnet.yaml", resume_path: str = "") -> None:
    """Thực thi vòng lặp huấn luyện, đánh giá Validation/Test và lưu checkpoint chuẩn."""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO (TV2): Parse CLI arguments (--config, --data_dir, --checkpoint_dir, --resume)
    pass

