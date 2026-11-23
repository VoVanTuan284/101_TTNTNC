"""
chess_ai/training/replay_buffer.py
Bộ đệm kinh nghiệm vòng (Circular Experience Replay Buffer) lưu trữ các bộ ba
(state_tensor [18, 8, 8], mcts_policy [4096], outcome_z) sinh ra từ quá trình tự chơi (Self-play).
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np


class SelfPlayReplayBuffer:
    """Replay Buffer phục vụ huấn luyện lặp AlphaZero-inspired (Cấp 8)."""

    def __init__(self, capacity: int = 50000, seed: Optional[int] = 42) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity phải > 0, nhận được: {capacity}")
        self.capacity = int(capacity)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self._states = np.zeros((self.capacity, 18, 8, 8), dtype=np.float32)
        self._policies = np.zeros((self.capacity, 4096), dtype=np.float32)
        self._values = np.zeros((self.capacity,), dtype=np.float32)
        self._size: int = 0
        self._write_pos: int = 0
        self.total_pushed: int = 0

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        """Xóa toàn bộ mẫu trong bộ đệm."""
        self._size = 0
        self._write_pos = 0
        self.total_pushed = 0

    def push_game_samples(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Thêm toàn bộ mẫu từ một ván tự chơi vào bộ đệm theo cơ chế vòng FIFO."""
        s_arr = np.asarray(states, dtype=np.float32)
        p_arr = np.asarray(policies, dtype=np.float32)
        v_arr = np.asarray(values, dtype=np.float32).reshape(-1)

        if s_arr.ndim == 3 and s_arr.shape == (18, 8, 8):
            s_arr = s_arr[np.newaxis, ...]
        if p_arr.ndim == 1 and p_arr.shape == (4096,):
            p_arr = p_arr[np.newaxis, ...]

        n = int(s_arr.shape[0])
        if n == 0:
            return
        if s_arr.shape[1:] != (18, 8, 8):
            raise ValueError(f"states phải có shape (N, 18, 8, 8), nhận được: {s_arr.shape}")
        if p_arr.shape != (n, 4096):
            raise ValueError(f"policies phải có shape ({n}, 4096), nhận được: {p_arr.shape}")
        if v_arr.shape != (n,):
            raise ValueError(f"values phải có shape ({n},), nhận được: {v_arr.shape}")

        for i in range(n):
            idx = self._write_pos
            self._states[idx] = s_arr[i]
            self._policies[idx] = p_arr[i]
            self._values[idx] = v_arr[i]
            self._write_pos = (self._write_pos + 1) % self.capacity
            if self._size < self.capacity:
                self._size += 1
            self.total_pushed += 1

    def sample_batch(self, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lấy mẫu ngẫu nhiên một batch (states, policies, values) để cập nhật mạng."""
        if self._size == 0:
            raise ValueError("Không thể sample_batch từ SelfPlayReplayBuffer đang rỗng.")
        b_size = max(1, int(batch_size))
        replace = self._size < b_size
        indices = self._rng.choice(self._size, size=b_size, replace=replace)
        return (
            self._states[indices].copy(),
            self._policies[indices].copy(),
            self._values[indices].copy(),
        )

    def state_dict(self) -> Dict[str, Any]:
        """Xuất trạng thái bộ đệm để lưu kèm checkpoint huấn luyện tiếp nối (--resume)."""
        return {
            "capacity": self.capacity,
            "size": self._size,
            "write_pos": self._write_pos,
            "total_pushed": self.total_pushed,
            "states": self._states[: self._size].copy(),
            "policies": self._policies[: self._size].copy(),
            "values": self._values[: self._size].copy(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Khôi phục dữ liệu bộ đệm từ state_dict đã lưu."""
        saved_states = np.asarray(state["states"], dtype=np.float32)
        saved_policies = np.asarray(state["policies"], dtype=np.float32)
        saved_values = np.asarray(state["values"], dtype=np.float32).reshape(-1)
        n = min(int(saved_states.shape[0]), self.capacity)
        self.clear()
        if n > 0:
            self._states[:n] = saved_states[:n]
            self._policies[:n] = saved_policies[:n]
            self._values[:n] = saved_values[:n]
            self._size = n
            self._write_pos = int(state.get("write_pos", n % self.capacity)) % self.capacity
            self.total_pushed = int(state.get("total_pushed", n))
