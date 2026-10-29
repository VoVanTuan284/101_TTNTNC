"""
chess_ai/core/game_env.py
Hợp đồng giao diện trừu tượng (Abstract Base Class) cho môi trường trò chơi đối kháng.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple


class GameEnv(ABC):
    """Abstract Base Class xác lập hợp đồng API thống nhất cho môi trường game đối kháng."""

    @abstractmethod
    def reset(self) -> Any:
        """Khởi tạo lại trạng thái ván đấu ban đầu."""
        raise NotImplementedError

    @abstractmethod
    def current_player(self) -> int:
        """Trả về người chơi đến lượt: +1 (Trắng / Tiên) hoặc -1 (Đen / Hậu)."""
        raise NotImplementedError

    @abstractmethod
    def legal_actions(self) -> List[Any]:
        """Trả về danh sách toàn bộ hành động hợp lệ tại trạng thái hiện tại."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        """Thực thi một hành động hợp lệ. Trả về (state, reward, done, info)."""
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self) -> bool:
        """Kiểm tra trạng thái hiện tại đã kết thúc ván đấu hay chưa."""
        raise NotImplementedError

    @abstractmethod
    def result(self) -> Optional[float]:
        """Trả về kết quả theo góc nhìn Trắng (+1.0, 0.0, -1.0) hoặc None nếu chưa kết thúc."""
        raise NotImplementedError

    @abstractmethod
    def render(self, mode: str = "unicode") -> Optional[str]:
        """Hiển thị trạng thái bàn cờ."""
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> "GameEnv":
        """Tạo bản sao độc lập của môi trường (bảo toàn lịch sử nước đi)."""
        raise NotImplementedError

    @abstractmethod
    def undo(self) -> None:
        """Hoàn tác nước đi gần nhất."""
        raise NotImplementedError

