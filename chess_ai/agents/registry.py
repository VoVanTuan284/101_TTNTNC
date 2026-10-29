"""
chess_ai/agents/registry.py
Registry khởi tạo tác tử theo định danh ('level1' -> 'level8') dùng chung cho
giao diện Pygame, động cơ giải đấu Arena và các kịch bản thực nghiệm Ablation.
Đánh dấu rõ phạm vi bàn cờ hỗ trợ (8x8 đầy đủ hay cờ tàn KRK chuyên biệt).
Phụ trách: TV3 phối hợp TV1 & TV2
"""

from typing import Any, Dict, List
from chess_ai.agents.base_agent import BaseAgent


AGENT_SPEC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "level1": {"class_name": "Level1RandomAgent", "full_board_supported": True},
    "level2": {"class_name": "Level2HeuristicAgent", "full_board_supported": True},
    "level3": {"class_name": "Level3AlphaBetaAgent", "full_board_supported": True},
    "level4": {"class_name": "Level4MCTSAgent", "full_board_supported": True},
    "level5": {"class_name": "Level5SupervisedAgent", "full_board_supported": True},
    "level6": {"class_name": "Level6ReinforcementAgent", "full_board_supported": False, "subgame": "KRK"},
    "level7": {"class_name": "Level7GuidedMCTSAgent", "full_board_supported": True},
    "level8": {"class_name": "Level8AlphaZeroAgent", "full_board_supported": True},
}


def create_agent(agent_id: str, **kwargs: Any) -> BaseAgent:
    """Khởi tạo thực thể BaseAgent theo mã cấp độ ('level1' .. 'level8')."""
    raise NotImplementedError


def list_full_board_agents() -> List[str]:
    """Danh sách các cấp tác tử hỗ trợ thi đấu giải 8x8 chuẩn."""
    return [k for k, v in AGENT_SPEC_REGISTRY.items() if v.get("full_board_supported", True)]

