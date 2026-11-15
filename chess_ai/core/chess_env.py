"""
chess_ai/core/chess_env.py
Lớp bọc môi trường Cờ vua chuẩn FIDE 8x8 dựa trên python-chess Bitboard.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from typing import Any, Dict, List, Optional, Tuple
import chess
from chess_ai.core.game_env import GameEnv


class ChessEnv(GameEnv):
    """Hiện thực hóa GameEnv cho Cờ vua FIDE 8x8."""

    def __init__(self, fen: Optional[str] = None) -> None:
        """Khởi tạo bàn cờ từ FEN hoặc trạng thái xuất phát mặc định."""
        self.initial_fen = fen
        self.board = chess.Board(fen) if fen else chess.Board()

    def reset(self, fen: Optional[str] = None) -> chess.Board:
        """Đặt lại bàn cờ."""
        target_fen = fen if fen is not None else self.initial_fen
        self.board = chess.Board(target_fen) if target_fen else chess.Board()
        return self.board

    def current_player(self) -> int:
        """Trả về +1 nếu lượt Trắng, -1 nếu lượt Đen."""
        return 1 if self.board.turn == chess.WHITE else -1

    def legal_actions(self) -> List[chess.Move]:
        """Danh sách nước đi hợp lệ."""
        return list(self.board.legal_moves)

    def step(self, action: chess.Move) -> Tuple[chess.Board, float, bool, Dict[str, Any]]:
        """Thực thi nước đi và trả về (board, reward, done, info)."""
        if action not in self.board.legal_moves:
            raise ValueError(f"Nước cờ bất hợp lệ: {action} tại thế cờ {self.board.fen()}")

        moving_color = self.board.turn
        self.board.push(action)
        done = self.is_terminal()
        res = self.result()

        # Phần thưởng tính theo góc nhìn tương đối của người chơi vừa thực hiện nước đi
        reward = 0.0
        if done and res is not None:
            if res == 1.0:
                reward = 1.0 if moving_color == chess.WHITE else -1.0
            elif res == -1.0:
                reward = 1.0 if moving_color == chess.BLACK else -1.0

        info: Dict[str, Any] = {
            "turn": self.board.turn,
            "is_check": self.board.is_check(),
            "fen": self.board.fen(),
            "halfmove_clock": self.board.halfmove_clock,
            "move_count": len(self.board.move_stack),
        }
        return self.board, reward, done, info

    def is_terminal(self) -> bool:
        """Kiểm tra kết thúc ván cờ đầy đủ (bao gồm lặp thế cờ 3 lần và 50 nước)."""
        return self.board.is_game_over(claim_draw=True)

    def is_terminal_fast(self) -> bool:
        """
        Kiểm tra kết thúc nhanh tại các nút trong của cây tìm kiếm (không quét lại toàn bộ
        move_stack cho threefold repetition; lặp thế cờ trong cây kiểm tra qua Zobrist path).
        """
        return (
            self.board.is_checkmate()
            or self.board.is_stalemate()
            or self.board.is_insufficient_material()
            or self.board.is_seventyfive_moves()
            or self.board.halfmove_clock >= 100
        )

    def result(self) -> Optional[float]:
        """Kết quả ván đấu theo góc nhìn Trắng (+1.0, 0.0, -1.0 hoặc None)."""
        if not self.is_terminal():
            return None
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == chess.WHITE else -1.0

    def render(self, mode: str = "unicode") -> Optional[str]:
        """Biểu diễn chuỗi bàn cờ."""
        if mode == "unicode":
            return self.board.unicode()
        if mode == "fen":
            return self.board.fen()
        return str(self.board)

    def clone(self) -> "ChessEnv":
        """Sao chép độc lập môi trường kèm ngăn xếp lịch sử (stack=True)."""
        new_env = ChessEnv.__new__(ChessEnv)
        new_env.initial_fen = self.initial_fen
        new_env.board = self.board.copy(stack=True)
        return new_env

    def undo(self) -> None:
        """Hoàn tác nước đi cuối."""
        if len(self.board.move_stack) > 0:
            self.board.pop()
