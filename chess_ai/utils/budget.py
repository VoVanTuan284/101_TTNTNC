"""
chess_ai/utils/budget.py
Bộ quản lý ngân sách tính toán hợp tác (Cooperative Time & Workload Budget Controller)
dùng chung cho Alpha-Beta (Cấp 3), Vanilla MCTS (Cấp 4) và Guided MCTS (Cấp 7/8).
Phụ trách: TV1 phối hợp TV3
"""

import time
from typing import Optional


class SearchBudgetController:
    """Kiểm tra giới hạn thời gian suy nghĩ (Time-Control) và số bước duyệt (Workload-Control)."""

    def __init__(
        self,
        time_limit_sec: Optional[float] = None,
        max_workload: Optional[int] = None,
        safety_margin_sec: float = 0.02,
    ) -> None:
        self.time_limit_sec = time_limit_sec
        self.max_workload = max_workload
        self.safety_margin_sec = safety_margin_sec
        self.start_time: float = time.perf_counter()
        self.steps_completed: int = 0

    def reset(self) -> None:
        """Khởi động lại bộ đếm khi bắt đầu tìm kiếm nước đi mới."""
        self.start_time = time.perf_counter()
        self.steps_completed = 0

    def elapsed_sec(self) -> float:
        """Thời gian đã trôi qua (giây)."""
        return time.perf_counter() - self.start_time

    def should_stop(self) -> bool:
        """Trả về True nếu đã chạm ngưỡng max_workload hoặc time_limit_sec."""
        if self.max_workload is not None and self.steps_completed >= self.max_workload:
            return True
        if self.time_limit_sec is not None:
            if self.elapsed_sec() >= max(0.001, self.time_limit_sec - self.safety_margin_sec):
                return True
        return False

