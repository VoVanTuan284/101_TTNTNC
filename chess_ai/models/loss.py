"""
chess_ai/models/loss.py
Hàm mất mát đa nhiệm kết hợp Cross-Entropy (Policy) và MSE (Value).
Điều chuẩn L2 được thực hiện trực tiếp qua decoupled weight_decay của AdamW.
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


class DualHeadLoss(nn.Module):
    """Hàm mất mát kết hợp cho Dual-Head ResNet."""

    def __init__(self, policy_weight: float = 1.0, value_weight: float = 1.0) -> None:
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight

    def forward(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        target_actions: torch.Tensor,
        target_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Tính tổng mất mát và trả về (total_loss, loss_components_dict)."""
        raise NotImplementedError

