"""
chess_ai/search/mcts_node.py
Cấu trúc nút cây MCTS cơ sở dùng chung cho cả Vanilla MCTS (UCT - Cấp 4)
và Guided MCTS (PUCT - Cấp 7 & Cấp 8), hỗ trợ tái sử dụng cây con (Subtree Reuse).
Phụ trách: TV1 phối hợp TV2
"""

from typing import Dict, Optional
import chess


class MCTSNode:
    """Nút trong cây tìm kiếm Monte Carlo."""

    def __init__(
        self,
        parent: Optional["MCTSNode"] = None,
        prior_prob: float = 1.0,
    ) -> None:
        self.parent = parent
        self.children: Dict[chess.Move, "MCTSNode"] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior_prob

    @property
    def q_value(self) -> float:
        """Giá trị kỳ vọng trung bình Q(s, a)."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def is_expanded(self) -> bool:
        """Kiểm tra nút đã được mở rộng hay chưa."""
        return len(self.children) > 0

