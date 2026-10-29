"""
chess_ai/arena/match.py
Hàm điều phối một ván đấu đơn lẻ giữa hai tác tử (Single-Game Runner),
dùng chung cho:
1. Giải đấu Round-Robin (chess_ai/arena/tournament.py).
2. Thực nghiệm Ablation 2 (chess_ai/experiments/ablation_mcts.py).
3. Đánh giá Checkpoint Arena Gating của Cấp 8 (chess_ai/training/selfplay.py).
4. Chế độ Máy đấu Máy trên giao diện Pygame (chess_ai/ui/desktop_pygame.py).
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

from typing import Any, Dict, Optional
import chess
from chess_ai.agents.base_agent import BaseAgent


def play_single_match(
    white_agent: BaseAgent,
    black_agent: BaseAgent,
    initial_fen: str = chess.STARTING_FEN,
    time_limit_sec: Optional[float] = None,
    max_workload: Optional[int] = None,
    max_plies: int = 400,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Điều phối một ván đấu hoàn chỉnh từ initial_fen đến khi kết thúc hoặc chạm max_plies.
    Trả về dictionary chứa kết quả, danh sách nước đi, độ trễ từng nước và thống kê tìm kiếm.
    """
    raise NotImplementedError

