"""
chess_ai/etl/memmap_dataset.py
Lớp PyTorch Dataset nạp dữ liệu trực tiếp từ tệp nhị phân np.memmap
({train,val,test}_{states,actions,values}.dat), đảm bảo RAM tiêu thụ < 1.5GB.
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""

from typing import Tuple
import torch
from torch.utils.data import Dataset


class ChessMemmapDataset(Dataset):
    """Dataset ánh xạ bộ nhớ cho các tập train, val, test."""

    def __init__(self, data_dir: str, split: str = "train") -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        # TODO (TV2): Mở các tệp np.memmap theo metadata trong dataset_metadata.json
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trả về (state_float32 [18, 8, 8], action_int64, value_float32)."""
        raise NotImplementedError

