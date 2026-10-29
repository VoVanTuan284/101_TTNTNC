"""
chess_ai/search/mcts_uct.py
Động cơ Vanilla MCTS sử dụng công thức UCB1 (UCT) cho Tác tử Cấp 4,
tích hợp mô phỏng cắt ngưỡng (max_rollout_depth = 60) và định giá tĩnh tanh.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from typing import Dict, Optional, Tuple
import chess


def search_uct(
    board: chess.Board,
    simulations: int = 400,
    time_limit: Optional[float] = None,
    c_uct: float = 1.414,
    max_rollout_depth: int = 60,
) -> Tuple[chess.Move, Dict[str, float]]:
    """Thực thi tìm kiếm Vanilla MCTS (UCT) và trả về (best_move, stats)."""
    raise NotImplementedError

