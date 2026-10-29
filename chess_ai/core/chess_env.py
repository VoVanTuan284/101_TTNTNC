"""
chess_ai/core/chess_env.py
Lớp bọc môi trường Cờ vua chuẩn FIDE 8x8 dựa trên python-chess Bitboard.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from typing import List, Optional, Tuple
import chess
from chess_ai.core.game_env import GameEnv


class ChessEnv(GameEnv):
    """Hiện thực hóa GameEnv cho Cờ vua FIDE 8x8."""

    def __init__(self, fen: Optional[str] = None) -> None:
        """Khởi tạo bàn cờ từ FEN hoặc trạng thái xuất phát mặc định."""
        raise NotImplementedError

    def reset(self, fen: Optional[str] = None) -> chess.Board:
        """Đặt lại bàn cờ."""
        raise NotImplementedError

    def current_player(self) -> int:
        """Trả về +1 nếu lượt Trắng, -1 nếu lượt Đen."""
        raise NotImplementedError

    def legal_actions(self) -> List[chess.Move]:
        """Danh sách nước đi hợp lệ."""
        raise NotImplementedError

    def step(self, action: chess.Move) -> Tuple[chess.Board, float, bool, dict]:
        """Thực thi nước đi và trả về (board, reward, done, info)."""
        raise NotImplementedError

    def is_terminal(self) -> bool:
        """Kiểm tra kết thúc ván cờ đầy đủ (bao gồm lặp thế cờ 3 lần và 50 nước)."""
        raise NotImplementedError

    def is_terminal_fast(self) -> bool:
        """Kiểm tra kết thúc nhanh tại các nút trong của cây tìm kiếm."""
        raise NotImplementedError

    def result(self) -> Optional[float]:
        """Kết quả ván đấu theo góc nhìn Trắng (+1.0, 0.0, -1.0 hoặc None)."""
        raise NotImplementedError

    def render(self, mode: str = "unicode") -> Optional[str]:
        """Biểu diễn chuỗi bàn cờ."""
        raise NotImplementedError

    def clone(self) -> "ChessEnv":
        """Sao chép độc lập môi trường kèm ngăn xếp lịch sử (stack=True)."""
        raise NotImplementedError

    def undo(self) -> None:
        """Hoàn tác nước đi cuối."""
        raise NotImplementedError

