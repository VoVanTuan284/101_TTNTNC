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
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Khối Residual Block 2 lớp Conv 3x3 + BatchNorm + ReLU + Skip Connection."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual, inplace=True)
        return out


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
        self.in_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.filters = filters
        self.policy_conv_filters = policy_conv_filters
        self.value_conv_filters = value_conv_filters
        self.value_hidden_units = value_hidden_units
        self.action_space = action_space

        # 1. Stem: Conv 3x3 (18 -> 64) + BN + ReLU
        self.stem_conv = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(filters)

        # 2. Shared Trunk: 6 Residual Blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(filters) for _ in range(num_res_blocks)])

        # 3. Policy Head: Conv 1x1 (64 -> 16) -> BN -> ReLU -> Flatten (16*8*8 = 1024) -> Linear(1024 -> 4096)
        self.policy_conv = nn.Conv2d(filters, policy_conv_filters, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_conv_filters)
        self.policy_fc = nn.Linear(policy_conv_filters * 8 * 8, action_space)

        # 4. Value Head: Conv 1x1 (64 -> 1) -> BN -> ReLU -> Flatten (64) -> Linear(64 -> 64) -> ReLU -> Linear(64 -> 1) -> Tanh
        self.value_conv = nn.Conv2d(filters, value_conv_filters, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_conv_filters)
        self.value_fc1 = nn.Linear(value_conv_filters * 8 * 8, value_hidden_units)
        self.value_fc2 = nn.Linear(value_hidden_units, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trả về tuple (policy_logits [B, 4096], value [B, 1] trong [-1, 1])."""
        out = F.relu(self.stem_bn(self.stem_conv(x)), inplace=True)
        for block in self.res_blocks:
            out = block(out)

        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(out)), inplace=True)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value Head
        v = F.relu(self.value_bn(self.value_conv(out)), inplace=True)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v), inplace=True)
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

