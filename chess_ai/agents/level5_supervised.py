"""
chess_ai/agents/level5_supervised.py
Cấp 5 [BẮT BUỘC]: Tác tử mạng nơ-ron học có giám sát Dual-Head ResNet
trên Tensor (18, 8, 8) góc nhìn bên đi, ra quyết định có áp Legal Action Masking.
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""

from typing import Optional
import chess
from chess_ai.agents.base_agent import BaseAgent
from chess_ai.core.chess_env import ChessEnv


class Level5SupervisedAgent(BaseAgent):
    """Tác tử Cấp 5: Supervised Dual-Head ResNet."""

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        super().__init__(name="Level5_Supervised")
        self.checkpoint_path = checkpoint_path

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Chọn nước đi hợp lệ có xác suất Policy Head cao nhất."""
        raise NotImplementedError

