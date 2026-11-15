"""
chess_ai/search/mcts_node.py
Cấu trúc nút cây MCTS cơ sở dùng chung cho cả Vanilla MCTS (UCT - Cấp 4)
và Guided MCTS (PUCT - Cấp 7 & Cấp 8), hỗ trợ tái sử dụng cây con (Subtree Reuse).
Phụ trách: TV1 phối hợp TV2
"""

from typing import Dict, Iterable, Optional
import chess


class MCTSNode:
    """Nút trong cây tìm kiếm Monte Carlo."""

    def __init__(
        self,
        parent: Optional["MCTSNode"] = None,
        prior_prob: float = 1.0,
        zobrist_key: Optional[int] = None,
    ) -> None:
        self.parent = parent
        self.children: Dict[chess.Move, "MCTSNode"] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior_prob
        self.zobrist_key: Optional[int] = zobrist_key

    @property
    def q_value(self) -> float:
        """Giá trị kỳ vọng trung bình Q(s, a) theo góc nhìn của nút cha."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def is_expanded(self) -> bool:
        """Kiểm tra nút đã được mở rộng hay chưa."""
        return len(self.children) > 0

    def is_leaf(self) -> bool:
        """Kiểm tra nút hiện tại có phải nút lá hay không."""
        return len(self.children) == 0

    def expand(
        self,
        actions: Iterable[chess.Move],
        priors: Optional[Dict[chess.Move, float]] = None,
    ) -> None:
        """Mở rộng các nút con từ danh sách nước đi hợp lệ và xác suất tiên nghiệm (nếu có)."""
        for move in actions:
            if move not in self.children:
                p = priors.get(move, 1.0) if priors is not None else 1.0
                self.children[move] = MCTSNode(parent=self, prior_prob=p)

    def find_reusable_root(self, target_zobrist: int) -> Optional["MCTSNode"]:
        """
        Tìm nút con hoặc nút cháu khớp khóa Zobrist 64-bit để tái sử dụng cây con (Subtree Reuse).
        Nếu tìm thấy, tách nút đó thành gốc độc lập (parent = None).
        """
        if self.zobrist_key == target_zobrist:
            self.parent = None
            return self
        for child in self.children.values():
            if child.zobrist_key == target_zobrist:
                child.parent = None
                return child
            for grandchild in child.children.values():
                if grandchild.zobrist_key == target_zobrist:
                    grandchild.parent = None
                    return grandchild
        return None
