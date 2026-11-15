"""
chess_ai/agents/level1_random.py
Cấp 1 [BẮT BUỘC]: Tác tử chọn nước đi ngẫu nhiên đồng đều (Uniform Random Baseline).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

import random
import time
from typing import Optional
import chess
from chess_ai.agents.base_agent import BaseAgent
from chess_ai.core.chess_env import ChessEnv


class Level1RandomAgent(BaseAgent):
    """Tác tử Cấp 1: Uniform Random."""

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(name="Level1_Random")
        self.seed = seed
        self._rng = random.Random(seed)

    def reset(self) -> None:
        super().reset()
        self._rng = random.Random(self.seed)

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Lấy mẫu ngẫu nhiên đồng đều một nước đi từ env.legal_actions()."""
        t0 = time.perf_counter()
        legal_moves = env.legal_actions()
        if not legal_moves:
            raise ValueError(f"Không có nước đi hợp lệ tại thế cờ: {env.board.fen()}")

        move = self._rng.choice(legal_moves)
        self.last_search_stats = {
            "nodes": 1,
            "simulations": 0,
            "depth": 0,
            "eval": 0.0,
            "time_sec": time.perf_counter() - t0,
        }
        return move
