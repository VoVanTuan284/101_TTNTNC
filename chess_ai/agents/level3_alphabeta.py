"""
chess_ai/agents/level3_alphabeta.py
Cấp 3 [BẮT BUỘC]: Tác tử Iterative Deepening Minimax / Alpha-Beta
tích hợp Move Ordering, Zobrist Transposition Table và Quiescence Search.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from typing import Optional
import chess
from chess_ai.agents.base_agent import BaseAgent
from chess_ai.config import load_yaml_config
from chess_ai.core.chess_env import ChessEnv
from chess_ai.search.alphabeta import search_alphabeta
from chess_ai.search.transposition import TranspositionTable


_AB_CFG = load_yaml_config("search_params.yaml")["alpha_beta"]


class Level3AlphaBetaAgent(BaseAgent):
    """Tác tử Cấp 3: Iterative Deepening Alpha-Beta + Ordering + TT + Quiescence."""

    def __init__(
        self,
        default_depth: int = int(_AB_CFG.get("default_depth", 4)),
        max_nodes: int = int(_AB_CFG.get("max_nodes", 10000)),
        tt_size_mb: int = int(_AB_CFG.get("tt_size_mb", 64)),
        use_quiescence: bool = bool(_AB_CFG.get("quiescence", {}).get("enabled", True)),
    ) -> None:
        super().__init__(name="Level3_AlphaBeta")
        self.default_depth = default_depth
        self.max_nodes = max_nodes
        self.use_quiescence = use_quiescence
        self.tt = TranspositionTable(max_size_mb=tt_size_mb)

    def reset(self) -> None:
        """Xóa Transposition Table và thống kê khi bắt đầu ván mới."""
        super().reset()
        self.tt.clear()

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Tìm nước đi tối ưu bằng Iterative Deepening Alpha-Beta."""
        # Khi có ngân sách nút hoặc thời gian, mở trần độ sâu (20) để ID tận dụng tối đa
        if max_workload is not None or time_limit is not None:
            search_depth = 20
        else:
            search_depth = self.default_depth

        best_move, stats = search_alphabeta(
            board=env.board,
            max_depth=search_depth,
            time_limit=time_limit,
            max_nodes=max_workload,
            use_tt=True,
            use_move_ordering=True,
            use_quiescence=self.use_quiescence,
            use_alpha_beta=True,
            tt=self.tt,
        )
        self.last_search_stats = stats
        return best_move
