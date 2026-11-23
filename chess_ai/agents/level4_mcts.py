"""
chess_ai/agents/level4_mcts.py
Cấp 4 [BẮT BUỘC]: Tác tử Vanilla MCTS (UCT) với mô phỏng cắt ngưỡng (max_rollout_depth = 60).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

import random
from typing import Optional
import chess
from chess_ai.agents.base_agent import BaseAgent
from chess_ai.config import load_yaml_config
from chess_ai.core.chess_env import ChessEnv
from chess_ai.core.zobrist import compute_zobrist_hash
from chess_ai.search.mcts_node import MCTSNode
from chess_ai.search.mcts_uct import run_uct_search


_MCTS_CFG = load_yaml_config("search_params.yaml")["vanilla_mcts"]


class Level4MCTSAgent(BaseAgent):
    """Tác tử Cấp 4: Vanilla MCTS (UCT)."""

    def __init__(
        self,
        simulations: int = int(_MCTS_CFG.get("default_simulations", 400)),
        c_uct: float = float(_MCTS_CFG.get("c_uct", 1.414)),
        max_rollout_depth: int = int(_MCTS_CFG.get("max_rollout_depth", 60)),
        subtree_reuse: bool = bool(_MCTS_CFG.get("subtree_reuse", True)),
        seed: Optional[int] = 42,
        rollout_mode: str = "see_tactical",
    ) -> None:
        if rollout_mode not in ("see_tactical", "uniform"):
            raise ValueError(f"rollout_mode không hợp lệ: '{rollout_mode}'. Hợp lệ: 'see_tactical', 'uniform'.")
        super().__init__(name="Level4_MCTS_UCT")
        self.simulations = simulations
        self.c_uct = c_uct
        self.max_rollout_depth = max_rollout_depth
        self.subtree_reuse = subtree_reuse
        self.seed = seed
        self.rollout_mode = rollout_mode
        self._rng = random.Random(seed)
        self._root: Optional[MCTSNode] = None

    def reset(self) -> None:
        """Xóa cây MCTS lưu trữ và đặt lại bộ sinh số ngẫu nhiên khi sang ván mới."""
        super().reset()
        self._root = None
        self._rng = random.Random(self.seed)

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Chọn nước đi bằng thuật toán UCT."""
        if max_workload is not None:
            sims = max_workload
        elif time_limit is not None:
            sims = 100_000
        else:
            sims = self.simulations

        reusable_root: Optional[MCTSNode] = None
        if self.subtree_reuse and self._root is not None:
            target_key = compute_zobrist_hash(env.board)
            reusable_root = self._root.find_reusable_root(target_key)

        best_move, stats, new_root = run_uct_search(
            board=env.board,
            simulations=sims,
            time_limit=time_limit,
            c_uct=self.c_uct,
            max_rollout_depth=self.max_rollout_depth,
            root=reusable_root,
            rng=self._rng,
            rollout_mode=self.rollout_mode,
        )
        self._root = new_root if self.subtree_reuse else None
        self.last_search_stats = stats
        return best_move
