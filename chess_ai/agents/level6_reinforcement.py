"""
chess_ai/agents/level6_reinforcement.py
Cấp 6 [MỞ RỘNG]: Tác tử Học tăng cường (Tabular Q-Learning & Policy Gradient REINFORCE)
trên không gian trạng thái cờ tàn chuẩn FIDE KRK (Vua + Xe đấu Vua).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from typing import Optional
import chess
from chess_ai.agents.base_agent import BaseAgent
from chess_ai.core.chess_env import ChessEnv


class Level6ReinforcementAgent(BaseAgent):
    """Tác tử Cấp 6: Reinforcement Learning trên cờ tàn KRK."""

    def __init__(self, q_table_path: Optional[str] = None) -> None:
        super().__init__(name="Level6_RL_KRK")
        self.q_table_path = q_table_path

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Chọn nước đi từ chính sách RL đã học."""
        raise NotImplementedError

