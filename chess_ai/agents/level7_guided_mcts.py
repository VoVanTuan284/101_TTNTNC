"""
chess_ai/agents/level7_guided_mcts.py
Cấp 7 [MỞ RỘNG]: Tác tử AlphaGo-inspired Guided MCTS (PUCT)
tích hợp Policy Prior P(s, a) và Value Head V(s) từ mạng Dual-Head ResNet,
FPU Reduction, phát hiện lặp nước 3 lần, Zobrist Eval Cache và Batched Virtual Loss.
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""

from collections import OrderedDict
from pathlib import Path
import time
from typing import Any, Dict, Optional
import chess
import numpy as np
import torch

from chess_ai.agents.base_agent import BaseAgent
from chess_ai.agents.level5_supervised import build_resnet_from_config
from chess_ai.config import load_yaml_config
from chess_ai.core.chess_env import ChessEnv
from chess_ai.core.zobrist import compute_zobrist_hash
from chess_ai.models.dual_resnet import DualHeadResNet
from chess_ai.search.mcts_node import MCTSNode
from chess_ai.search.mcts_puct import EvalCache, run_puct_search


_PUCT_CFG = load_yaml_config("search_params.yaml")["guided_mcts"]


class Level7GuidedMCTSAgent(BaseAgent):
    """Tác tử Cấp 7: Guided MCTS (ResNet Prior + Value + PUCT)."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = "checkpoints/best_model.pt",
        simulations: int = int(_PUCT_CFG.get("default_simulations", 400)),
        c_puct: float = float(_PUCT_CFG.get("c_puct", 1.414)),
        eval_mode: bool = True,
        subtree_reuse: bool = bool(_PUCT_CFG.get("subtree_reuse", True)),
        allow_untrained_fallback: bool = True,
        device: Optional[str] = None,
        seed: Optional[int] = 42,
        fpu_reduction: float = float(_PUCT_CFG.get("fpu_reduction", 0.2)),
        eval_cache_size: int = int(_PUCT_CFG.get("eval_cache_size", 8192)),
        batch_size: Optional[int] = None,
        uniform_prior: bool = False,
    ) -> None:
        super().__init__(name="Level7_ValueOnlyMCTS" if uniform_prior else "Level7_GuidedMCTS")
        self.checkpoint_path = checkpoint_path or "checkpoints/best_model.pt"
        self.simulations = simulations
        self.c_puct = c_puct
        self.eval_mode = eval_mode
        self.subtree_reuse = subtree_reuse
        self.allow_untrained_fallback = allow_untrained_fallback
        self.seed = seed
        self.fpu_reduction = fpu_reduction
        self.eval_cache_size = eval_cache_size
        self.uniform_prior = uniform_prior
        self._rng = np.random.default_rng(seed)
        self._root: Optional[MCTSNode] = None
        self._eval_cache: EvalCache = OrderedDict()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if batch_size is None:
            self.batch_size = 8 if self.device.type == "cuda" else 1
        else:
            self.batch_size = max(1, int(batch_size))

        self.model: Optional[DualHeadResNet] = None
        self.checkpoint_loaded: bool = False
        self.checkpoint_meta: Dict[str, Any] = {}
        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        """Nạp trọng số DualHeadResNet từ checkpoint_path (ưu tiên best_model.pt của Level 5)."""
        if self.model is not None:
            return

        candidates = [self.checkpoint_path] if self.checkpoint_path else []
        if self.allow_untrained_fallback:
            for fallback in ("checkpoints/best_model.pt", "checkpoints/l5_supervised_best.pt", "checkpoints/latest.pt"):
                if fallback not in candidates:
                    candidates.append(fallback)

        loaded_file: Optional[Path] = None
        for cand in candidates:
            p = Path(cand)
            if p.exists():
                loaded_file = p
                break

        if loaded_file is not None:
            payload = torch.load(loaded_file, map_location=self.device, weights_only=False)
            model_cfg = payload.get("model_config") if isinstance(payload, dict) else None
            self.model = build_resnet_from_config(model_cfg)
            state_dict = (
                payload["model_state_dict"]
                if isinstance(payload, dict) and "model_state_dict" in payload
                else payload
            )
            self.model.load_state_dict(state_dict)
            self.checkpoint_loaded = True
            self.checkpoint_path = str(loaded_file)
            if isinstance(payload, dict):
                self.checkpoint_meta = {
                    k: payload.get(k)
                    for k in ("epoch", "global_step", "val_policy_acc", "val_value_mse", "best_val_loss")
                    if k in payload
                }
        else:
            if not self.allow_untrained_fallback:
                raise FileNotFoundError(
                    f"Không tìm thấy checkpoint cho Level7GuidedMCTSAgent tại: {self.checkpoint_path}"
                )
            if self.seed is not None:
                torch.manual_seed(self.seed)
            self.model = build_resnet_from_config()
            self.checkpoint_loaded = False

        self.model.to(self.device)
        self.model.eval()

    def reset(self) -> None:
        """Xóa cây MCTS lưu trữ, xóa NN eval cache và khởi tạo lại RNG khi sang ván mới."""
        super().reset()
        self._root = None
        self._eval_cache.clear()
        self._rng = np.random.default_rng(self.seed)

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Thực thi tìm kiếm PUCT dẫn hướng bởi mạng Dual-Head ResNet."""
        t0 = time.perf_counter()
        self._ensure_model_loaded()
        assert self.model is not None

        legal_moves = list(env.board.legal_moves)
        if not legal_moves:
            raise ValueError(f"Không có nước đi hợp lệ tại thế cờ: {env.board.fen()}")

        # Forced move shortcut: chỉ có 1 nước hợp lệ -> trả về ngay, tiết kiệm thời gian
        if len(legal_moves) == 1:
            forced_move = legal_moves[0]
            if self.subtree_reuse and self._root is not None:
                target_key = compute_zobrist_hash(env.board)
                reusable = self._root.find_reusable_root(target_key)
                if reusable is not None and forced_move in reusable.children:
                    child = reusable.children[forced_move]
                    child.parent = None
                    self._root = child
                else:
                    self._root = None
            else:
                self._root = None

            self.last_search_stats = {
                "nodes": 1.0,
                "simulations": 0.0,
                "depth": 1.0,
                "eval": 0.0,
                "root_value": 0.0,
                "cache_hits": 0.0,
                "time_sec": time.perf_counter() - t0,
            }
            return forced_move

        if max_workload is not None:
            sims = int(max_workload)
        elif time_limit is not None:
            sims = 100_000
        else:
            sims = self.simulations

        reusable_root: Optional[MCTSNode] = None
        if self.subtree_reuse and self._root is not None:
            target_key = compute_zobrist_hash(env.board)
            reusable_root = self._root.find_reusable_root(target_key)

        selfplay_cfg = _PUCT_CFG.get("selfplay_mode", {})
        dir_alpha = float(selfplay_cfg.get("dirichlet_alpha", 0.3))
        dir_eps = float(selfplay_cfg.get("dirichlet_epsilon", 0.25))

        best_move, stats, new_root = run_puct_search(
            board=env.board,
            model=self.model,
            simulations=sims,
            time_limit=time_limit,
            c_puct=self.c_puct,
            eval_mode=self.eval_mode,
            dirichlet_alpha=dir_alpha,
            dirichlet_epsilon=dir_eps,
            root=reusable_root,
            device=self.device,
            rng=self._rng,
            fpu_reduction=self.fpu_reduction,
            eval_cache=self._eval_cache if self.eval_cache_size > 0 else None,
            max_cache_size=self.eval_cache_size,
            batch_size=self.batch_size,
            uniform_prior=self.uniform_prior,
        )
        self._root = new_root if self.subtree_reuse else None
        self.last_search_stats = {
            "nodes": float(stats["nodes"]),
            "simulations": float(stats["simulations"]),
            "depth": float(stats["depth"]),
            "eval": float(stats["eval"]),
            "root_value": float(stats.get("root_value", 0.0)),
            "cache_hits": float(stats.get("cache_hits", 0.0)),
            "time_sec": float(stats["time_sec"]),
        }
        return best_move
