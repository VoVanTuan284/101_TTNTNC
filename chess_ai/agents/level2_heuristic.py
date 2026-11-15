"""
chess_ai/agents/level2_heuristic.py
Cấp 2 [BẮT BUỘC]: Tác tử Heuristic dựa trên luật phân cấp 3 tầng (Think 1-3):
- Think 1: Chiếu hết ngay trong 1 nước (Mate-in-1).
- Think 2: Phòng thủ nước chiếu hết của đối thủ & Bắt quân có lợi theo MVV-LVA.
- Think 3: Tối ưu hóa hàm lượng giá tĩnh (Material + PST Michniewski + King Safety + Mobility).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

import random
import time
from typing import Optional
import chess
from chess_ai.agents.base_agent import BaseAgent
from chess_ai.core.chess_env import ChessEnv
from chess_ai.core.evaluation import MATE_VALUE, PIECE_VALUES, evaluate_board


class Level2HeuristicAgent(BaseAgent):
    """Tác tử Cấp 2: Rule-based / Heuristic 3 tầng."""

    def __init__(self, seed: Optional[int] = 42, repetition_penalty: float = 150.0) -> None:
        super().__init__(name="Level2_Heuristic")
        self.seed = seed
        self.repetition_penalty = repetition_penalty
        self._rng = random.Random(seed)

    def reset(self) -> None:
        super().reset()
        self._rng = random.Random(self.seed)

    @staticmethod
    def _opponent_has_mate_in_one(board: chess.Board) -> bool:
        """Kiểm tra nhanh xem đối thủ có nước chiếu hết ngay ở lượt kế tiếp không."""
        for opp_move in board.legal_moves:
            if not board.gives_check(opp_move):
                continue
            board.push(opp_move)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return True
        return False

    @staticmethod
    def _max_opponent_capture_gain(board: chess.Board) -> float:
        """
        Ước lượng mức lỗ vật chất lớn nhất nếu đối thủ thực hiện một nước bắt quân ngay sau đó
        (tính gần đúng Static Exchange: nếu ô đích có quân ta bảo vệ thì trừ giá trị quân tấn công).
        """
        our_color = not board.turn
        best_gain = 0.0
        for cap in board.generate_legal_captures():
            if board.is_en_passant(cap):
                victim_val = PIECE_VALUES[chess.PAWN]
            else:
                victim = board.piece_at(cap.to_square)
                victim_val = PIECE_VALUES[victim.piece_type] if victim else 0

            attacker = board.piece_at(cap.from_square)
            attacker_val = PIECE_VALUES[attacker.piece_type] if attacker else 0

            if board.is_attacked_by(our_color, cap.to_square):
                net_gain = victim_val - attacker_val
            else:
                net_gain = victim_val

            if net_gain > best_gain:
                best_gain = float(net_gain)
        return best_gain

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Chọn nước đi theo thứ tự ưu tiên luật Think 1 -> Think 2 -> Think 3."""
        t0 = time.perf_counter()
        board = env.board
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError(f"Không có nước đi hợp lệ tại thế cờ: {board.fen()}")

        # Think 1: Thắng ngay lập tức nếu có nước chiếu hết trong 1 nước (Mate-in-1)
        for move in legal_moves:
            if not board.gives_check(move):
                continue
            board.push(move)
            if board.is_checkmate():
                board.pop()
                self.last_search_stats = {
                    "nodes": len(legal_moves),
                    "simulations": 0,
                    "depth": 1,
                    "eval": MATE_VALUE,
                    "time_sec": time.perf_counter() - t0,
                }
                return move
            board.pop()

        best_move = legal_moves[0]
        best_score = -float("inf")
        nodes = 0

        # Xáo nhẹ thứ tự để phá vỡ thế hòa điểm tất định một cách có kiểm soát
        candidates = list(legal_moves)
        self._rng.shuffle(candidates)

        for move in candidates:
            nodes += 1
            board.push(move)

            # Think 3: Điểm lượng giá tĩnh sau 1 nước đi (đảo dấu vì lượt đã chuyển sang đối thủ)
            static_eval = -evaluate_board(board)

            # Think 2a: Tránh tự sát vào thế bị đối thủ chiếu hết ở nước kế tiếp
            if self._opponent_has_mate_in_one(board):
                score = -MATE_VALUE + 100.0
            else:
                # Think 2b: Phạt nếu để treo quân hoặc đổi quân lỗ theo MVV-LVA / SEE gần đúng
                opp_capture_threat = self._max_opponent_capture_gain(board)
                score = static_eval - 0.85 * opp_capture_threat

                # Tránh hòa lặp thế cờ vô nghĩa khi đang có ưu thế chất
                if static_eval > 50.0 and board.is_repetition(2):
                    score -= self.repetition_penalty

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if time_limit is not None and (time.perf_counter() - t0) >= time_limit:
                break

        self.last_search_stats = {
            "nodes": nodes,
            "simulations": 0,
            "depth": 1,
            "eval": float(best_score),
            "time_sec": time.perf_counter() - t0,
        }
        return best_move
