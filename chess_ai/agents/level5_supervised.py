"""
chess_ai/agents/level5_supervised.py
Cấp 5 [BẮT BUỘC]: Tác tử mạng nơ-ron học có giám sát Dual-Head ResNet
trên Tensor (18, 8, 8) góc nhìn bên đi, ra quyết định có áp Legal Action Masking.
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""

from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple
import chess
import numpy as np
import torch

from chess_ai.agents.base_agent import BaseAgent
from chess_ai.config import load_yaml_config
from chess_ai.core.board_tensor import (
    action_index_to_move,
    board_to_tensor,
    get_legal_action_mask,
)
from chess_ai.core.chess_env import ChessEnv
from chess_ai.models.dual_resnet import DualHeadResNet


def build_resnet_from_config(model_cfg: Optional[Dict[str, Any]] = None) -> DualHeadResNet:
    """Khởi tạo mạng DualHeadResNet theo cấu hình từ configs/model_resnet.yaml."""
    if model_cfg is None:
        model_cfg = load_yaml_config("model_resnet.yaml")
    net_cfg = model_cfg.get("network", model_cfg)
    p_head = net_cfg.get("policy_head", {})
    v_head = net_cfg.get("value_head", {})
    return DualHeadResNet(
        in_channels=18,
        num_res_blocks=int(net_cfg.get("num_res_blocks", 6)),
        filters=int(net_cfg.get("filters", 64)),
        policy_conv_filters=int(p_head.get("conv_filters", 16)),
        value_conv_filters=int(v_head.get("conv_filters", 1)),
        value_hidden_units=int(v_head.get("hidden_units", 64)),
        action_space=int(p_head.get("linear_out", 4096)),
    )


class Level5SupervisedAgent(BaseAgent):
    """Tác tử Cấp 5: Supervised Dual-Head ResNet."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = "checkpoints/best_model.pt",
        device: Optional[str] = None,
        allow_untrained_fallback: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__(name="Level5_Supervised")
        self.checkpoint_path = checkpoint_path or "checkpoints/best_model.pt"
        self.allow_untrained_fallback = allow_untrained_fallback
        self.seed = seed
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[DualHeadResNet] = None
        self.checkpoint_loaded: bool = False
        self.checkpoint_meta: Dict[str, Any] = {}
        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        """Nạp trọng số từ checkpoint_path nếu tệp tồn tại, ngược lại khởi tạo mạng mặc định ở chế độ eval()."""
        if self.model is not None:
            return
        ckpt_file = Path(self.checkpoint_path) if self.checkpoint_path else None
        if ckpt_file is not None and ckpt_file.exists():
            payload = torch.load(ckpt_file, map_location=self.device, weights_only=False)
            model_cfg = payload.get("model_config") if isinstance(payload, dict) else None
            self.model = build_resnet_from_config(model_cfg)
            state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
            self.model.load_state_dict(state_dict)
            self.checkpoint_loaded = True
            if isinstance(payload, dict):
                self.checkpoint_meta = {
                    k: payload.get(k)
                    for k in ("epoch", "global_step", "val_policy_acc", "val_value_mse", "best_val_loss")
                    if k in payload
                }
        else:
            self.model = build_resnet_from_config()
            self.checkpoint_loaded = False

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_policy_and_value(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        """
        Dự đoán phân phối xác suất hợp lệ P(s, a) shape (4096,) và giá trị thế cờ V(s) trong [-1, 1].
        Áp dụng Legal Action Masking (gán -inf cho mọi hành động không hợp lệ trước khi Softmax).
        """
        self._ensure_model_loaded()
        assert self.model is not None

        tensor_np = board_to_tensor(board)
        x = torch.from_numpy(tensor_np).unsqueeze(0).to(self.device)
        logits_t, value_t = self.model(x)
        logits = logits_t[0].detach().cpu().numpy()
        value = float(value_t[0, 0].detach().cpu().item())

        mask = get_legal_action_mask(board)
        masked_logits = np.where(mask > 0.5, logits, -1e9)
        max_logit = float(np.max(masked_logits))
        exp_scores = np.where(mask > 0.5, np.exp(masked_logits - max_logit), 0.0)
        sum_exp = float(np.sum(exp_scores))
        if sum_exp > 0.0:
            probs = (exp_scores / sum_exp).astype(np.float32)
        else:
            mask_sum = float(np.sum(mask))
            probs = (mask / max(1.0, mask_sum)).astype(np.float32)

        return probs, value

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Chọn nước đi hợp lệ có xác suất Policy Head cao nhất."""
        t0 = time.perf_counter()
        board = env.board
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError(f"Không có nước đi hợp lệ tại thế cờ: {board.fen()}")

        if len(legal_moves) == 1:
            self.last_search_stats = {
                "nodes": 1.0,
                "simulations": 0.0,
                "depth": 1.0,
                "eval": 0.0,
                "time_sec": time.perf_counter() - t0,
            }
            return legal_moves[0]

        probs, value = self.predict_policy_and_value(board)
        best_action_idx = int(np.argmax(probs))
        move = action_index_to_move(best_action_idx, board)

        elapsed = time.perf_counter() - t0
        self.last_search_stats = {
            "nodes": 1.0,
            "simulations": 0.0,
            "depth": 1.0,
            "eval": float(value),
            "top_prob": float(probs[best_action_idx]),
            "time_sec": float(elapsed),
        }
        return move

