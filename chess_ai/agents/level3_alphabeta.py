"""
chess_ai/agents/level3_alphabeta.py
Cấp 3 [BẮT BUỘC]: Tác tử Iterative Deepening Minimax / Alpha-Beta
tích hợp Move Ordering, Zobrist Transposition Table và Quiescence Search.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from typing import Optional
import chess
from chess_ai.agents.base_agent import BaseAgent
from chess_ai.core.chess_env import ChessEnv


class Level3AlphaBetaAgent(BaseAgent):
    """Tác tử Cấp 3: Iterative Deepening Alpha-Beta + Ordering + TT + Quiescence."""

    def __init__(self, default_depth: int = 4, max_nodes: int = 10000) -> None:
        super().__init__(name="Level3_AlphaBeta")
        self.default_depth = default_depth
        self.max_nodes = max_nodes

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Tìm nước đi tối ưu bằng Iterative Deepening Alpha-Beta."""
        raise NotImplementedError

