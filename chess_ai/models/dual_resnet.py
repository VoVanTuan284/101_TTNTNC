"""
chess_ai/models/dual_resnet.py
Kiến trúc mạng nơ-ron đa nhiệm Dual-Head ResNet (Shared Trunk 6x64,
Policy Head 16 filters -> 4096, Value Head 1 filter -> 64 -> 1 -> Tanh)
nhận đầu vào Tensor chuẩn hóa góc nhìn bên đi shape (B, 18, 8, 8).
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""

from typing import Tuple
import torch
import torch.nn as nn


class DualHeadResNet(nn.Module):
    """Mạng Dual-Head ResNet dẫn hướng cho Cấp 5, Cấp 7 và Cấp 8."""

    def __init__(
        self,
        in_channels: int = 18,
        num_res_blocks: int = 6,
        filters: int = 64,
        policy_conv_filters: int = 16,
        value_conv_filters: int = 1,
        value_hidden_units: int = 64,
        action_space: int = 4096,
    ) -> None:
        super().__init__()
        # TODO (TV2): Khởi tạo lớp Conv đầu vào, 6 Residual Blocks, Policy Head và Value Head
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trả về tuple (policy_logits [B, 4096], value [B, 1] trong [-1, 1])."""
        raise NotImplementedError

