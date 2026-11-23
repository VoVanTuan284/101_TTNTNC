"""
chess_ai/etl/memmap_dataset.py
Lớp PyTorch Dataset nạp dữ liệu trực tiếp từ tệp nhị phân np.memmap
({train,val,test}_{states,actions,values}.dat), đảm bảo RAM tiêu thụ < 1.5GB.
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""

import json
from pathlib import Path
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def mirror_action_index(action_idx: int) -> int:
    """Lật gương trái-phải chỉ số hành động phẳng u * 64 + v (cột a<->h: sq ^ 7)."""
    u, v = divmod(int(action_idx), 64)
    return (u ^ 7) * 64 + (v ^ 7)


class ChessMemmapDataset(Dataset):
    """Dataset ánh xạ bộ nhớ cho các tập train, val, test."""

    def __init__(
        self,
        data_dir: str = "data/processed",
        split: str = "train",
        augment_mirror: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        # Lật gương trái-phải (a<->h) với xác suất 0.5 khi không còn quyền nhập thành.
        # Khi hết quyền nhập thành, luật cờ đối xứng gương hoàn toàn nên nhãn vẫn đúng.
        self.augment_mirror = bool(augment_mirror)

        meta_path = self.data_dir / "dataset_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Không tìm thấy metadata tại {meta_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.length = int(meta["splits"][split]["samples"])

        if self.length > 0:
            self.states = np.memmap(
                self.data_dir / f"{split}_states.dat",
                dtype=np.uint8,
                mode="r",
                shape=(self.length, 18, 8, 8),
            )
            self.actions = np.memmap(
                self.data_dir / f"{split}_actions.dat",
                dtype=np.int16,
                mode="r",
                shape=(self.length,),
            )
            self.values = np.memmap(
                self.data_dir / f"{split}_values.dat",
                dtype=np.float32,
                mode="r",
                shape=(self.length,),
            )
        else:
            self.states = np.empty((0, 18, 8, 8), dtype=np.uint8)
            self.actions = np.empty((0,), dtype=np.int16)
            self.values = np.empty((0,), dtype=np.float32)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trả về (state_float32 [18, 8, 8], action_int64, value_float32)."""
        state_np = np.array(self.states[idx], dtype=np.float32, copy=True)
        action_idx = int(self.actions[idx])
        if self.augment_mirror and not state_np[12:16].any() and random.random() < 0.5:
            state_np = np.ascontiguousarray(state_np[:, :, ::-1])
            action_idx = mirror_action_index(action_idx)
        state_np[17] /= 100.0
        state_t = torch.from_numpy(state_np)
        action_t = torch.tensor(action_idx, dtype=torch.long)
        value_t = torch.tensor(float(self.values[idx]), dtype=torch.float32)
        return state_t, action_t, value_t
