"""
chess_ai/agents/level2_heuristic.py
Cấp 2 [BẮT BUỘC]: Tác tử Heuristic dựa trên luật phân cấp 3 tầng (Think 1-3):
- Think 1: Chiếu hết ngay trong 1 nước (Mate-in-1).
- Think 2: Phòng thủ nước chiếu hết của đối thủ & Bắt quân có lợi theo MVV-LVA.
- Think 3: Tối ưu hóa hàm lượng giá tĩnh (Material + PST Michniewski + King Safety + Mobility).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from typing import Optional
import chess
from chess_ai.agents.base_agent import BaseAgent
from chess_ai.core.chess_env import ChessEnv


class Level2HeuristicAgent(BaseAgent):
    """Tác tử Cấp 2: Rule-based / Heuristic 3 tầng."""

    def __init__(self) -> None:
        super().__init__(name="Level2_Heuristic")

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Chọn nước đi theo thứ tự ưu tiên luật Think 1 -> Think 2 -> Think 3."""
        raise NotImplementedError

