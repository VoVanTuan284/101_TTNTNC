"""
chess_ai/models/loss.py
Hàm mất mát đa nhiệm kết hợp Cross-Entropy (Policy) và MSE (Value).
Điều chuẩn L2 được thực hiện trực tiếp qua decoupled weight_decay của AdamW.
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualHeadLoss(nn.Module):
    """Hàm mất mát kết hợp cho Dual-Head ResNet."""

    def __init__(self, policy_weight: float = 1.0, value_weight: float = 1.0) -> None:
        super().__init__()
        self.policy_weight = float(policy_weight)
        self.value_weight = float(value_weight)

    def forward(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        target_actions: torch.Tensor,
        target_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Tính tổng mất mát và trả về (total_loss, loss_components_dict).
        - target_actions:1D Tensor (B,) chứa chỉ số hành động [0, 4095], hoặc 2D Tensor (B, 4096) chứa phân phối xác suất pi (dùng cho AlphaZero Self-Play).
        - target_values: Tensor (B,) hoặc (B, 1) trong [-1, 1].
        """
        if target_actions.dim() == 1:
            policy_loss = F.cross_entropy(policy_logits, target_actions.long())
        else:
            log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_loss = -(target_actions * log_probs).sum(dim=-1).mean()

        v_pred_flat = value_pred.view(-1)
        v_target_flat = target_values.view(-1).to(dtype=v_pred_flat.dtype)
        value_loss = F.mse_loss(v_pred_flat, v_target_flat)

        total_loss = self.policy_weight * policy_loss + self.value_weight * value_loss
        components: Dict[str, float] = {
            "total_loss": float(total_loss.detach().item()),
            "policy_loss": float(policy_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
        }
        return total_loss, components

