"""
chess_ai/agents/level4_mcts.py
Cấp 4 [BẮT BUỘC]: Tác tử Vanilla MCTS (UCT) với mô phỏng cắt ngưỡng (max_rollout_depth = 60).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from typing import Optional
import chess
from chess_ai.agents.base_agent import BaseAgent
from chess_ai.core.chess_env import ChessEnv


class Level4MCTSAgent(BaseAgent):
    """Tác tử Cấp 4: Vanilla MCTS (UCT)."""

    def __init__(self, simulations: int = 400, c_uct: float = 1.414) -> None:
        super().__init__(name="Level4_MCTS_UCT")
        self.simulations = simulations
        self.c_uct = c_uct

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Chọn nước đi bằng thuật toán UCT."""
        raise NotImplementedError

