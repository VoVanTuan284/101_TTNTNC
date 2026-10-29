"""
chess_ai/agents/level1_random.py
Cấp 1 [BẮT BUỘC]: Tác tử chọn nước đi ngẫu nhiên đồng đều (Uniform Random Baseline).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from typing import Optional
import chess
from chess_ai.agents.base_agent import BaseAgent
from chess_ai.core.chess_env import ChessEnv


class Level1RandomAgent(BaseAgent):
    """Tác tử Cấp 1: Uniform Random."""

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(name="Level1_Random")
        self.seed = seed

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Lấy mẫu ngẫu nhiên đồng đều một nước đi từ env.legal_actions()."""
        raise NotImplementedError

