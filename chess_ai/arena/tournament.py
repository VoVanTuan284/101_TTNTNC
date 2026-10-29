"""
chess_ai/arena/tournament.py
Động cơ điều phối giải đấu Round-Robin 100 ván/cặp (50 thế cờ khai cuộc cân bằng x 2 màu)
hỗ trợ hai chế độ ngân sách: Workload-Control và Time-Control.
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

from typing import List
from chess_ai.agents.base_agent import BaseAgent


def run_round_robin_tournament(
    agents: List[BaseAgent],
    config_path: str = "configs/default_config.yaml",
    budget_mode: str = "workload_control",
) -> None:
    """Vận hành giải đấu đối kháng giữa danh sách tác tử và lưu vết ra results/round_robin/."""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO (TV3): Parse CLI arguments và khởi chạy giải đấu
    pass

