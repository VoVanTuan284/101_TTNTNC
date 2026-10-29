"""
chess_ai/agents/base_agent.py
Lớp cơ sở trừu tượng (Abstract Base Class) cho toàn bộ 8 cấp độ tác tử AI.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import chess
from chess_ai.core.chess_env import ChessEnv


class BaseAgent(ABC):
    """Hợp đồng giao diện chung cho mọi tác tử cờ vua."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.last_search_stats: Dict[str, Any] = {}

    @abstractmethod
    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Chọn nước đi hợp lệ tốt nhất trong ngân sách time_limit hoặc max_workload."""
        raise NotImplementedError

    def reset(self) -> None:
        """Tái thiết lập trạng thái đệm nội tại khi bắt đầu ván đấu mới."""
        self.last_search_stats.clear()

