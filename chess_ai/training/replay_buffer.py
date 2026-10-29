"""
chess_ai/training/replay_buffer.py
Bộ đệm kinh nghiệm vòng (Circular Experience Replay Buffer) lưu trữ các bộ ba
(state_tensor [18, 8, 8], mcts_policy [4096], outcome_z) sinh ra từ quá trình tự chơi (Self-play).
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""

from typing import Tuple
import numpy as np


class SelfPlayReplayBuffer:
    """Replay Buffer phục vụ huấn luyện lặp AlphaZero-inspired (Cấp 8)."""

    def __init__(self, capacity: int = 50000) -> None:
        self.capacity = capacity

    def push_game_samples(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Thêm toàn bộ mẫu từ một ván tự chơi vào bộ đệm."""
        raise NotImplementedError

    def sample_batch(self, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lấy mẫu ngẫu nhiên một batch (states, policies, values) để cập nhật mạng."""
        raise NotImplementedError

