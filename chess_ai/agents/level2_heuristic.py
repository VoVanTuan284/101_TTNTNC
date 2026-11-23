"""
chess_ai/agents/level2_heuristic.py
Cấp 2 [BẮT BUỘC]: Tác tử Heuristic 1-ply với hệ thống luật phân tầng tường minh 6 Tiers:
- Tier 1 (`RULE_T1_MATE_IN_1`): Chiếu hết ngay trong 1 nước.
- Tier 2 (`RULE_T2_SAFE_PROMOTION` / `RULE_T2_WINNING_CAPTURE_SEE`): Phong Hậu an toàn hoặc bắt quân lời ròng theo SEE.
- Tier 3 (`RULE_T3_SAFE_CHECK` / `RULE_T3_POSITIONAL`): Kiểm soát vị trí + 4 ô trung tâm + chiếu an toàn.
- Tier 4 (`RULE_T4_AVOID_REPETITION_2/3`): Nước gây lặp thế cờ khi không thua thiệt (chỉ đi nếu hết Tier 1-3).
- Tier 5 (`RULE_T5_AVOID_STALEMATE`): Nước gây hòa Pat / thiếu lực lượng khi đang thắng.
- Tier 6 (`RULE_T6_AVOID_MATE_IN_1`): Nước để đối thủ chiếu hết ngay ở lượt kế tiếp.
Ghi nhận đầy đủ luật kích hoạt và xung đột luật vào `last_search_stats`.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

import random
import time
from typing import Any, Dict, List, Optional, Tuple
import chess

from chess_ai.agents.base_agent import BaseAgent
from chess_ai.config import load_yaml_config
from chess_ai.core.chess_env import ChessEnv
from chess_ai.core.evaluation import MATE_VALUE, evaluate_board, see_capture


_L2_CFG: Dict[str, Any] = load_yaml_config("search_params.yaml").get("heuristic_level2", {})
_CENTER_SQUARES: Tuple[int, ...] = (chess.D4, chess.D5, chess.E4, chess.E5)


def center_control_score(board: chess.Board, color: chess.Color) -> int:
    """Đo mức kiểm soát 4 ô trung tâm (d4, d5, e4, e5) của `color` so với đối thủ."""
    opp = not color
    score = 0
    for sq in _CENTER_SQUARES:
        our_att = bin(board.attackers_mask(color, sq)).count("1")
        opp_att = bin(board.attackers_mask(opp, sq)).count("1")
        score += our_att - opp_att
        piece = board.piece_at(sq)
        if piece is not None:
            score += 2 if piece.color == color else -2
    return score


class Level2HeuristicAgent(BaseAgent):
    """Tác tử Cấp 2: Rule-based / Heuristic 3 tầng với thứ tự ưu tiên & giải quyết xung đột luật."""

    def __init__(
        self,
        seed: Optional[int] = 42,
        repetition_penalty: Optional[float] = None,
        repetition_eval_threshold: Optional[float] = None,
        stalemate_penalty: Optional[float] = None,
        safe_check_bonus: Optional[float] = None,
        center_control_weight: Optional[float] = None,
        see_hanging_weight: Optional[float] = None,
    ) -> None:
        super().__init__(name="Level2_Heuristic")
        self.seed = seed
        self.repetition_penalty = float(
            repetition_penalty
            if repetition_penalty is not None
            else _L2_CFG.get("repetition_penalty", 180.0)
        )
        self.repetition_eval_threshold = float(
            repetition_eval_threshold
            if repetition_eval_threshold is not None
            else _L2_CFG.get("repetition_eval_threshold", 0.0)
        )
        self.stalemate_penalty = float(
            stalemate_penalty
            if stalemate_penalty is not None
            else _L2_CFG.get("stalemate_penalty", 5000.0)
        )
        self.safe_check_bonus = float(
            safe_check_bonus
            if safe_check_bonus is not None
            else _L2_CFG.get("safe_check_bonus", 30.0)
        )
        self.center_control_weight = float(
            center_control_weight
            if center_control_weight is not None
            else _L2_CFG.get("center_control_weight", 8.0)
        )
        self.see_hanging_weight = float(
            see_hanging_weight
            if see_hanging_weight is not None
            else _L2_CFG.get("see_hanging_weight", 0.90)
        )
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
    def _max_opponent_see_gain(board: chess.Board) -> float:
        """Ước lượng mức lời vật chất lớn nhất đối thủ có thể đạt được bằng SEE ở lượt kế tiếp."""
        best_gain = 0.0
        for cap in board.generate_legal_captures():
            gain = float(see_capture(board, cap))
            if gain > best_gain:
                best_gain = gain
        return best_gain

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Chọn nước đi theo thứ tự ưu tiên 3 tầng (Think 1 -> Think 2 -> Think 3) & ghi nhận xung đột luật."""
        t0 = time.perf_counter()
        board = env.board
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError(f"Không có nước đi hợp lệ tại thế cờ: {board.fen()}")

        our_color = board.turn
        root_eval = evaluate_board(board)

        # Think 1 (Tier 1 - Cao nhất): Thắng ngay lập tức nếu có Mate-in-1
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
                    "selected_tier": 1,
                    "selected_rule": "RULE_T1_MATE_IN_1",
                    "conflicts_detected": 0,
                    "conflict_examples": [],
                }
                return move
            board.pop()

        best_move = legal_moves[0]
        best_tier = 99
        best_rule = "RULE_T3_POSITIONAL"
        best_score = -float("inf")
        nodes = 0
        conflicts_detected = 0
        conflict_examples: List[str] = []

        candidates = list(legal_moves)
        self._rng.shuffle(candidates)

        for move in candidates:
            nodes += 1
            is_capture = board.is_capture(move)
            is_check = board.gives_check(move)
            move_see = see_capture(board, move) if is_capture else 0

            board.push(move)

            static_eval = -evaluate_board(board)
            center_bonus = self.center_control_weight * center_control_score(board, our_color)
            positional_score = static_eval + center_bonus

            opp_mate = self._opponent_has_mate_in_one(board)
            is_stalemate_or_insuff = board.is_stalemate() or board.is_insufficient_material()
            is_rep3 = board.is_repetition(3)
            is_rep2 = board.is_repetition(2)
            opp_see_threat = 0.0 if opp_mate else self._max_opponent_see_gain(board)

            board.pop()

            # Xác định tầng ưu tiên (tier nhỏ hơn = ưu tiên cao hơn) và phát hiện xung đột luật
            positive_signals: List[str] = []
            negative_signals: List[str] = []

            if move.promotion == chess.QUEEN and opp_see_threat < 800.0:
                positive_signals.append("SAFE_PROMOTION")
            if is_capture and move_see > 0:
                positive_signals.append("WINNING_CAPTURE")
            elif is_capture and move_see < 0:
                negative_signals.append("LOSING_CAPTURE")
            if is_check and (not is_capture or move_see >= 0) and opp_see_threat < 100.0:
                positive_signals.append("SAFE_CHECK")
            if positional_score > root_eval + 15.0:
                positive_signals.append("IMPROVES_POSITION")

            if opp_mate:
                negative_signals.append("ALLOWS_OPP_MATE_IN_1")
            if is_stalemate_or_insuff and root_eval >= 100.0:
                negative_signals.append("CAUSES_STALEMATE_WHILE_WINNING")
            if (is_rep2 or is_rep3) and static_eval >= self.repetition_eval_threshold:
                negative_signals.append("CAUSES_REPETITION")
            if opp_see_threat >= 100.0:
                negative_signals.append("LEAVES_PIECE_HANGING")

            if positive_signals and negative_signals:
                conflicts_detected += 1
                if len(conflict_examples) < 5:
                    conflict_examples.append(
                        f"{move.uci()}: {'+'.join(positive_signals)} vs {'+'.join(negative_signals)}"
                    )

            # Phân tầng quyết định 6 Tiers rạch ròi (Tier 1 xử lý ở trên -> Tier 2..6 tại đây)
            base_score = positional_score - self.see_hanging_weight * opp_see_threat
            if opp_mate:
                tier = 6
                rule = "RULE_T6_AVOID_MATE_IN_1"
                score = -MATE_VALUE + 100.0
            elif is_stalemate_or_insuff and root_eval >= 100.0:
                tier = 5
                rule = "RULE_T5_AVOID_STALEMATE"
                score = -self.stalemate_penalty
            elif static_eval >= self.repetition_eval_threshold and (is_rep3 or is_rep2):
                tier = 4
                if is_rep3:
                    rule = "RULE_T4_AVOID_REPETITION_3"
                    score = base_score - 2.0 * self.repetition_penalty
                else:
                    rule = "RULE_T4_AVOID_REPETITION_2"
                    score = base_score - self.repetition_penalty
            elif move.promotion == chess.QUEEN and opp_see_threat < 800.0:
                tier = 2
                rule = "RULE_T2_SAFE_PROMOTION"
                score = base_score
            elif is_capture and move_see > 0 and opp_see_threat < move_see:
                tier = 2
                rule = "RULE_T2_WINNING_CAPTURE_SEE"
                score = base_score
            else:
                tier = 3
                if is_check and opp_see_threat < 100.0:
                    score = base_score + self.safe_check_bonus
                    rule = "RULE_T3_SAFE_CHECK"
                else:
                    score = base_score
                    rule = "RULE_T3_POSITIONAL"

            # So sánh theo cặp (tier ưu tiên hơn, cùng tier thì điểm score cao hơn)
            if (tier < best_tier) or (tier == best_tier and score > best_score):
                best_tier = tier
                best_score = score
                best_move = move
                best_rule = rule

            if time_limit is not None and (time.perf_counter() - t0) >= time_limit:
                break

        self.last_search_stats = {
            "nodes": nodes,
            "simulations": 0,
            "depth": 1,
            "eval": float(best_score),
            "time_sec": time.perf_counter() - t0,
            "selected_tier": best_tier,
            "selected_rule": best_rule,
            "conflicts_detected": conflicts_detected,
            "conflict_examples": conflict_examples,
        }
        return best_move
