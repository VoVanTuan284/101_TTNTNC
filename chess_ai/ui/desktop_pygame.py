"""Giao diện Pygame tối giản cho người đấu AI và xem lại ván đấu.

Module này không đọc ``ChessEnv.board`` và không chứa luật cờ. Mọi trạng thái
được lấy từ ``render``, ``legal_actions``, ``clone`` và mọi thay đổi đi qua
``step`` của môi trường.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Any, Optional

from chess_ai.agents.registry import AGENT_SPEC_REGISTRY, create_agent
from chess_ai.core.chess_env import ChessEnv


@dataclass(frozen=True)
class MoveRecord:
    """Một nước đi và trạng thái sau nước đi, dùng cho lịch sử/replay."""

    action: Any
    san: str
    snapshot: ChessEnv
    elapsed: float


class ChessPygameApp:
    """Ứng dụng desktop; Pygame chỉ được import khi thực sự khởi chạy."""

    BOARD_SIZE = 640
    PANEL_WIDTH = 360
    WINDOW_SIZE = (BOARD_SIZE + PANEL_WIDTH, BOARD_SIZE)
    LIGHT = (238, 238, 210)
    DARK = (118, 150, 86)
    PANEL = (30, 34, 42)
    TEXT = (235, 239, 244)
    MUTED = (165, 174, 188)
    ACCENT = (86, 166, 255)
    HIGHLIGHT = (246, 210, 82)

    def __init__(
        self,
        player_color: str = "white",
        agent_id: str = "level1",
        time_limit: Optional[float] = 1.0,
        max_workload: Optional[int] = None,
    ) -> None:
        self.player_color = player_color
        self.agent_id = agent_id
        self.time_limit = time_limit
        self.max_workload = max_workload
        self.env = ChessEnv()
        self.agent = create_agent(agent_id)
        self.records: list[MoveRecord] = []
        self.replay_index = 0
        self.selected_square: Optional[int] = None
        self.message = "Lượt Trắng"
        self.ai_future: Optional[Future[Any]] = None
        self.ai_started = 0.0
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.flipped = False

    def run(self) -> None:
        """Vòng lặp Pygame chính."""
        import pygame

        pygame.init()
        screen = pygame.display.set_mode(self.WINDOW_SIZE)
        pygame.display.set_caption("Chess AI - Người vs AI")
        clock = pygame.time.Clock()
        running = True
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._handle_click(event.pos, pygame)
                    elif event.type == pygame.KEYDOWN:
                        running = self._handle_key(event.key, pygame) and running
                self._poll_ai()
                self._draw(screen, pygame)
                pygame.display.flip()
                clock.tick(60)
        finally:
            self.executor.shutdown(wait=False, cancel_futures=True)
            pygame.quit()

    def _new_game(self) -> None:
        if self.ai_future is not None and not self.ai_future.done():
            self.message = "Hãy chờ AI hoàn tất nước đang tính"
            return
        self.ai_future = None
        self.env = ChessEnv()
        self.agent = create_agent(self.agent_id)
        self.records.clear()
        self.replay_index = 0
        self.selected_square = None
        self.message = "Lượt Trắng"

    def _undo(self) -> None:
        if self.ai_future is not None and not self.ai_future.done():
            return
        count = 2 if self.records and self._is_ai_turn_after_last_move() else 1
        del self.records[max(0, len(self.records) - count):]
        self._restore_index(len(self.records))
        self.message = "Đã đi lại một nước"

    def _is_ai_turn_after_last_move(self) -> bool:
        return len(self.records) > 0 and self._side_to_move() != self.player_color

    def _restore_index(self, index: int) -> None:
        self.replay_index = max(0, min(index, len(self.records)))
        self.env = (
            ChessEnv()
            if self.replay_index == 0
            else self.records[self.replay_index - 1].snapshot.clone()
        )
        self.selected_square = None

    def _side_to_move(self) -> str:
        fen = self.env.render("fen")
        if not isinstance(fen, str):
            raise RuntimeError("ChessEnv không trả về FEN dạng chuỗi")
        return "white" if fen.split()[1] == "w" else "black"

    def _handle_key(self, key: int, pygame: Any) -> bool:
        if key == pygame.K_n:
            self._new_game()
        elif key == pygame.K_f:
            self.flipped = not self.flipped
        elif key == pygame.K_u:
            self._undo()
        elif key == pygame.K_HOME:
            self._restore_index(0)
        elif key == pygame.K_END:
            self._restore_index(len(self.records))
        elif key == pygame.K_LEFT:
            self._restore_index(self.replay_index - 1)
        elif key == pygame.K_RIGHT:
            self._restore_index(self.replay_index + 1)
        elif key == pygame.K_ESCAPE:
            self.selected_square = None
        elif key == pygame.K_p:
            self.player_color = "black" if self.player_color == "white" else "white"
            self._new_game()
        elif key == pygame.K_a:
            self._cycle_agent()
        elif key == pygame.K_b:
            self._toggle_budget()
        return True

    def _handle_click(self, position: tuple[int, int], pygame: Any) -> None:
        x, y = position
        if x >= self.BOARD_SIZE:
            self._handle_panel_click(y)
            return
        if self.ai_future is not None or self.env.is_terminal():
            return
        if self.replay_index != len(self.records):
            self._restore_index(len(self.records))
        square = self._square_at(x, y)
        if self._side_to_move() != self.player_color:
            return
        if self.selected_square is None:
            legal = self._legal_from(square)
            if legal:
                self.selected_square = square
            return
        action = self._find_action(self.selected_square, square)
        if action is None:
            self.selected_square = square if self._legal_from(square) else None
            return
        self._play_action(action)

    def _handle_panel_click(self, y: int) -> None:
        if y < 48:
            self._new_game()
        elif y < 96:
            self.flipped = not self.flipped
        elif y < 144:
            self._undo()
        elif y < 192:
            self._restore_index(0)
        elif y < 240:
            self._restore_index(self.replay_index - 1)
        elif y < 288:
            self._restore_index(self.replay_index + 1)
        elif y < 336:
            self._restore_index(len(self.records))
        elif y < 384:
            self.player_color = "black" if self.player_color == "white" else "white"
            self._new_game()
        elif y < 432:
            self._cycle_agent()
        elif y < 480:
            self._toggle_budget()

    def _cycle_agent(self) -> None:
        agent_ids = sorted(AGENT_SPEC_REGISTRY)
        next_index = (agent_ids.index(self.agent_id) + 1) % len(agent_ids)
        self.agent_id = agent_ids[next_index]
        self._new_game()

    def _toggle_budget(self) -> None:
        if self.time_limit is not None:
            self.time_limit = None
            self.max_workload = 200
        else:
            self.time_limit = 1.0
            self.max_workload = None
        self._new_game()

    def _legal_from(self, square: int) -> list[Any]:
        return [
            move for move in self.env.legal_actions()
            if move.from_square == square
        ]

    def _find_action(self, source: int, target: int) -> Optional[Any]:
        for move in self.env.legal_actions():
            if move.from_square == source and move.to_square == target:
                return move
        return None

    def _play_action(self, action: Any) -> None:
        started = time.perf_counter()
        san = self.env.san(action)
        self.env.step(action)
        self.records = self.records[:self.replay_index]
        self.records.append(MoveRecord(action, san, self.env.clone(), time.perf_counter() - started))
        self.replay_index = len(self.records)
        self.selected_square = None
        self.message = self._status_text()

    def _start_ai(self) -> None:
        if self.env.is_terminal() or self._side_to_move() == self.player_color:
            return
        snapshot = self.env.clone()
        self.ai_started = time.perf_counter()
        self.message = f"{self.agent.name} đang suy nghĩ..."
        self.ai_future = self.executor.submit(
            self.agent.select_move,
            snapshot,
            self.time_limit,
            self.max_workload,
        )

    def _poll_ai(self) -> None:
        if self.ai_future is None:
            self._start_ai()
            return
        if not self.ai_future.done():
            return
        future, self.ai_future = self.ai_future, None
        try:
            self._play_action(future.result())
            self.records[-1] = MoveRecord(
                self.records[-1].action,
                self.records[-1].san,
                self.records[-1].snapshot,
                time.perf_counter() - self.ai_started,
            )
        except Exception as exc:
            self.message = f"Lỗi AI: {exc}"

    def _status_text(self) -> str:
        if self.env.is_terminal():
            result = self.env.result()
            winner = "Trắng thắng" if result == 1.0 else "Đen thắng" if result == -1.0 else "Hòa"
            reason = self.env.termination_reason() or "Kết thúc ván"
            return f"{winner} - {reason}"
        return "Lượt Trắng" if self._side_to_move() == "white" else "Lượt Đen"

    def _square_at(self, x: int, y: int) -> int:
        col, row = x // (self.BOARD_SIZE // 8), y // (self.BOARD_SIZE // 8)
        if self.flipped:
            col, row = 7 - col, 7 - row
        return (7 - row) * 8 + col

    def _screen_square(self, square: int) -> tuple[int, int]:
        file_index, rank_index = square % 8, square // 8
        if self.flipped:
            file_index, rank_index = 7 - file_index, 7 - rank_index
        return file_index, 7 - rank_index

    def _pieces(self) -> dict[int, str]:
        fen = self.env.render("fen")
        if not isinstance(fen, str):
            raise RuntimeError("ChessEnv không trả về FEN dạng chuỗi")
        pieces: dict[int, str] = {}
        for rank, row in enumerate(fen.split()[0].split("/")):
            file_index = 0
            for char in row:
                if char.isdigit():
                    file_index += int(char)
                else:
                    pieces[(7 - rank) * 8 + file_index] = char
                    file_index += 1
        return pieces

    def _draw(self, screen: Any, pygame: Any) -> None:
        screen.fill(self.PANEL)
        tile = self.BOARD_SIZE // 8
        pieces = self._pieces()
        legal_targets = {move.to_square for move in self._legal_from(self.selected_square)}
        for square in range(64):
            x, y = self._screen_square(square)
            color = self.LIGHT if (x + y) % 2 == 0 else self.DARK
            rect = pygame.Rect(x * tile, y * tile, tile, tile)
            pygame.draw.rect(screen, color, rect)
            if square == self.selected_square:
                pygame.draw.rect(screen, self.HIGHLIGHT, rect, 5)
            elif square in legal_targets:
                pygame.draw.circle(screen, self.HIGHLIGHT, rect.center, 10)
            if square in pieces:
                font = pygame.font.SysFont("segoeuisymbol", 52)
                glyph = {
                    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
                    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
                }[pieces[square]]
                text = font.render(glyph, True, (25, 25, 25) if pieces[square].islower() else (250, 250, 250))
                screen.blit(text, text.get_rect(center=rect.center))
        self._draw_panel(screen, pygame)

    def _draw_panel(self, screen: Any, pygame: Any) -> None:
        font = pygame.font.SysFont("segoeui", 18)
        small = pygame.font.SysFont("segoeui", 15)
        x = self.BOARD_SIZE + 18
        buttons = [
            "Ván mới [N]", "Đảo bàn [F]", "Undo [U]", "Replay: First [Home]",
            "Prev [←]", "Next [→]", "Last [End]", "Đổi phe [P]",
            "Đổi AI [A]", "Đổi ngân sách [B]",
        ]
        for index, label in enumerate(buttons):
            rect = pygame.Rect(self.BOARD_SIZE + 10, 10 + index * 48, self.PANEL_WIDTH - 20, 38)
            pygame.draw.rect(screen, (48, 55, 67), rect, border_radius=6)
            screen.blit(font.render(label, True, self.TEXT), (x, rect.y + 8))
        y = 515
        lines = [
            f"Người: {'Trắng' if self.player_color == 'white' else 'Đen'}",
            f"AI: {self.agent.name} ({self.agent_id})",
            f"Ngân sách: {self.time_limit}s" if self.time_limit is not None else f"Mô phỏng: {self.max_workload}",
            f"Trạng thái: {self.message}",
            f"Replay: {self.replay_index}/{len(self.records)}",
            "",
            "Lịch sử SAN:",
        ]
        for line in lines:
            screen.blit(small.render(line, True, self.TEXT if line else self.MUTED), (x, y))
            y += 22
        history = []
        for i in range(0, len(self.records), 2):
            item = f"{i // 2 + 1}. {self.records[i].san}"
            if i + 1 < len(self.records):
                item += f" {self.records[i + 1].san}"
            history.append(item)
        for line in history[-10:]:
            screen.blit(small.render(line, True, self.MUTED), (x, y))
            y += 20


def launch_desktop_app() -> None:
    """Khởi chạy cửa sổ ứng dụng cờ vua: ``python -m chess_ai.ui.desktop_pygame``."""
    ChessPygameApp().run()


if __name__ == "__main__":
    launch_desktop_app()

