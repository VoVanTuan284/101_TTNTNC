"""Desktop Pygame interface for chess play, analysis and replay."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any, Optional

from chess_ai.agents.registry import (
    AGENT_SPEC_REGISTRY,
    create_agent,
    list_full_board_agents,
)
from chess_ai.core.chess_env import ChessEnv


@dataclass(frozen=True)
class MoveRecord:
    action: Any
    san: str
    snapshot: ChessEnv
    elapsed: float


@dataclass
class _Clock:
    seconds: float
    increment: float = 0.0

    def tick(self, elapsed: float) -> None:
        self.seconds = max(0.0, self.seconds - elapsed)

    def add_increment(self) -> None:
        self.seconds += self.increment


class ChessPygameApp:
    """Desktop Pygame application following ChessEnv and agent contracts."""

    WINDOW_SIZE = (1280, 760)
    TAB_HEIGHT = 44
    LEFT_WIDTH = 270
    ARENA_WIDTH = 670
    RIGHT_WIDTH = 340
    BOARD_SIZE = 544
    LIGHT = (238, 238, 210)
    DARK = (118, 150, 86)
    BG = (15, 19, 28)
    PANEL = (24, 30, 42)
    PANEL_2 = (33, 42, 56)
    PANEL_3 = (42, 54, 72)
    BORDER = (60, 72, 92)
    BORDER_STRONG = (76, 92, 116)
    SURFACE = (28, 35, 48)
    SURFACE_HOVER = (38, 48, 64)
    TEXT = (235, 239, 244)
    MUTED = (165, 174, 188)
    ACCENT = (86, 166, 255)
    GREEN = (74, 222, 128)
    ORANGE = (245, 158, 11)
    HIGHLIGHT = (246, 210, 82)
    LAST_MOVE = (186, 203, 68)

    def __init__(
        self,
        player_color: str = "white",
        agent_id: str = "level1",
        time_limit: Optional[float] = 1.0,
        max_workload: Optional[int] = None,
        piece_set: str = "cburnett",
    ) -> None:
        self.player_color = player_color
        self.agent_id = agent_id
        self.time_limit = time_limit
        self.max_workload = max_workload
        self.piece_set = piece_set

        self.env = ChessEnv()
        self.white_agent = create_agent(agent_id)
        self.black_agent = create_agent(agent_id)
        self.agent = self.black_agent if player_color == "white" else self.white_agent
        self.records: list[MoveRecord] = []
        self.replay_index = 0
        self.selected_square: Optional[int] = None
        self.message = "Lượt Trắng"
        self.ai_future: Optional[Future[Any]] = None
        self.ai_agent: Optional[Any] = None
        self.ai_started = 0.0
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.flipped = False
        self.game_mode = "human_ai"
        self.white_clock = _Clock(600.0)
        self.black_clock = _Clock(600.0)
        self.clock_running = True
        self.last_clock_tick = time.perf_counter()
        self.fullscreen = False
        self.modal_open = False
        self.modal_mode = "human_ai"
        self.modal_human_color = "white"
        self.modal_white_agent = "level1"
        self.modal_black_agent = "level1"
        self.modal_budget = "time"
        self.modal_time = 1.0
        self.modal_workload = 400
        self.modal_base = 600.0
        self.modal_increment = 0.0
        self.active_tab = "play"
        self.promotion_pending: Optional[Any] = None
        self.piece_images: dict[str, Any] = {}
        self.resign_result: Optional[float] = None
        self.draw_offer_pending = False

        self.lab_view = "arena"
        self.lab_running = False
        self.lab_progress = 0.0
        self.lab_message = "Sẵn sàng"
        self.lab_scenario = "arena"
        self.lab_agents = list_full_board_agents()
        self.lab_num_games = 20
        self.lab_budget_mode = "workload"
        self.lab_workers = 1
        self.lab_results: dict[str, Any] = {}
        self.lab_stop_event = threading.Event()

    def run(self) -> None:
        import pygame

        pygame.init()
        screen = pygame.display.set_mode(self.WINDOW_SIZE, pygame.RESIZABLE)
        pygame.display.set_caption("Chess AI - Play & Telemetry")
        canvas = pygame.Surface(self.WINDOW_SIZE)
        clock = pygame.time.Clock()
        self._load_piece_images(pygame)
        running = True
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        position = self._fit_window_position(event.pos, screen, pygame)
                        self._handle_click(position, pygame)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_F11 or (
                            event.key == pygame.K_RETURN
                            and pygame.key.get_mods() & pygame.KMOD_ALT
                        ):
                            screen = self._toggle_fullscreen(pygame)
                        elif event.key == pygame.K_ESCAPE and self.fullscreen:
                            screen = self._leave_fullscreen(pygame)
                        else:
                            running = self._handle_key(event.key, pygame) and running
                    elif hasattr(pygame, "WINDOWMAXIMIZED") and event.type == pygame.WINDOWMAXIMIZED:
                        screen = self._enter_fullscreen(pygame)
                    elif hasattr(pygame, "WINDOWRESTORED") and event.type == pygame.WINDOWRESTORED:
                        if self.fullscreen:
                            screen = self._leave_fullscreen(pygame)
                self._tick_clocks()
                self._poll_ai()
                self._draw(canvas, pygame)
                screen.fill((0, 0, 0))
                fitted = self._fit_surface(canvas, screen)
                screen.blit(fitted, fitted.get_rect(center=screen.get_rect().center))
                pygame.display.flip()
                clock.tick(60)
        finally:
            self.executor.shutdown(wait=False, cancel_futures=True)
            import pygame as pg
            pg.quit()

    def _load_piece_images(self, pygame: Any) -> None:
        piece_dir = (
            Path(__file__).resolve().parent
            / "assets"
            / "piece"
            / self.piece_set
        )
        names = {
            "K": "wK", "Q": "wQ", "R": "wR", "B": "wB", "N": "wN", "P": "wP",
            "k": "bK", "q": "bQ", "r": "bR", "b": "bB", "n": "bN", "p": "bP",
        }
        for symbol, filename in names.items():
            for suffix in (".png", ".svg"):
                path = piece_dir / f"{filename}{suffix}"
                if path.exists():
                    try:
                        image = pygame.image.load(str(path)).convert_alpha()
                        self.piece_images[symbol] = pygame.transform.smoothscale(
                            image, (60, 60)
                        )
                        break
                    except Exception:
                        continue

    def _tick_clocks(self) -> None:
        now = time.perf_counter()
        elapsed = now - self.last_clock_tick
        self.last_clock_tick = now
        if not self.modal_open and self.clock_running and not self.env.is_terminal():
            side = self._side_to_move()
            (self.white_clock if side == "white" else self.black_clock).tick(elapsed)
            if (self.white_clock if side == "white" else self.black_clock).seconds <= 0:
                self.message = f"Het gio - {'Den' if side == 'white' else 'Trang'} thang"
                self.clock_running = False

    def _new_game(self) -> None:
        if self.modal_open:
            return
        if self.ai_future is not None and not self.ai_future.done():
            self.message = "Hãy chờ AI hoàn tất nước đang tính"
            return
        self.modal_open = True

    def _start_configured_game(self) -> None:
        if self.ai_future is not None and not self.ai_future.done():
            return
        self.game_mode = self.modal_mode
        if self.modal_mode == "human_human":
            self.player_color = "both"
            self.white_agent = None
            self.black_agent = None
        elif self.modal_mode == "human_ai":
            self.player_color = self.modal_human_color
            ai_id = (
                self.modal_black_agent
                if self.modal_human_color == "white"
                else self.modal_white_agent
            )
            ai_id = ai_id if ai_id in list_full_board_agents() else "level1"
            ai_agent = create_agent(ai_id)
            self.white_agent = ai_agent if self.player_color == "black" else None
            self.black_agent = ai_agent if self.player_color == "white" else None
            self.agent_id = ai_id
            self.agent = ai_agent
        else:
            self.player_color = "none"
            self.white_agent = create_agent(self.modal_white_agent)
            self.black_agent = create_agent(self.modal_black_agent)
            self.agent = self.white_agent

        if self.modal_budget == "time":
            self.time_limit = self.modal_time
            self.max_workload = None
        else:
            self.time_limit = None
            self.max_workload = self.modal_workload
        if self.modal_base > 0:
            self.white_clock = _Clock(self.modal_base, self.modal_increment)
            self.black_clock = _Clock(self.modal_base, self.modal_increment)
        else:
            self.white_clock = _Clock(0.0, 0.0)
            self.black_clock = _Clock(0.0, 0.0)
        self.env = ChessEnv()
        self.records.clear()
        self.replay_index = 0
        self.selected_square = None
        self.ai_future = None
        self.modal_open = False
        self.clock_running = self.modal_base > 0
        self.last_clock_tick = time.perf_counter()
        self.resign_result = None
        self.draw_offer_pending = False
        self.message = "Lượt Trắng"

    def _undo(self) -> None:
        if self.ai_future is not None and not self.ai_future.done():
            return
        count = 2 if self.game_mode == "human_ai" else 1
        count = min(count, len(self.records))
        if count:
            del self.records[len(self.records) - count:]
        self._restore_index(len(self.records))
        self.message = "Đã hoàn tác"

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

    def _is_human_turn(self) -> bool:
        if self.game_mode == "human_human":
            return True
        if self.game_mode == "human_ai":
            return self._side_to_move() == self.player_color
        return False

    def _enter_fullscreen(self, pygame: Any, native: bool = True) -> Any:
        if native:
            desktop_sizes = pygame.display.get_desktop_sizes()
            size = desktop_sizes[0] if desktop_sizes else self.WINDOW_SIZE
            screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        else:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.fullscreen = True
        pygame.display.set_caption("Chess AI - Play & Telemetry")
        return screen

    def _leave_fullscreen(self, pygame: Any) -> Any:
        screen = pygame.display.set_mode(self.WINDOW_SIZE, pygame.RESIZABLE)
        self.fullscreen = False
        pygame.display.set_caption("Chess AI - Play & Telemetry")
        return screen

    def _toggle_fullscreen(self, pygame: Any) -> Any:
        if self.fullscreen:
            return self._leave_fullscreen(pygame)
        return self._enter_fullscreen(pygame, native=False)

    def _fit_surface(self, canvas: Any, screen: Any) -> Any:
        import pygame

        scale = min(
            screen.get_width() / self.WINDOW_SIZE[0],
            screen.get_height() / self.WINDOW_SIZE[1],
        )
        size = (
            max(1, int(self.WINDOW_SIZE[0] * scale)),
            max(1, int(self.WINDOW_SIZE[1] * scale)),
        )
        return pygame.transform.smoothscale(canvas, size)

    def _fit_window_position(
        self, position: tuple[int, int], screen: Any, pygame: Any
    ) -> tuple[int, int]:
        x, y = position
        scale = min(
            screen.get_width() / self.WINDOW_SIZE[0],
            screen.get_height() / self.WINDOW_SIZE[1],
        )
        fitted_width = int(self.WINDOW_SIZE[0] * scale)
        fitted_height = int(self.WINDOW_SIZE[1] * scale)
        offset_x = (screen.get_width() - fitted_width) // 2
        offset_y = (screen.get_height() - fitted_height) // 2
        return (
            max(0, min(self.WINDOW_SIZE[0] - 1, int((x - offset_x) / scale))),
            max(0, min(self.WINDOW_SIZE[1] - 1, int((y - offset_y) / scale))),
        )

    def _handle_key(self, key: int, pygame: Any) -> bool:
        if self.modal_open:
            if key == pygame.K_ESCAPE:
                self.modal_open = False
            return True
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
        elif key == pygame.K_m:
            self.game_mode = (
                "human_ai" if self.game_mode == "human_human"
                else "human_human" if self.game_mode == "ai_ai"
                else "ai_ai"
            )
        elif key == pygame.K_r:
            self._resign()
        elif key == pygame.K_d:
            self._offer_draw()
        return True

    def _handle_click(self, position: tuple[int, int], pygame: Any) -> None:
        x, y = position
        if y < self.TAB_HEIGHT:
            if 0 <= x < 350:
                self.active_tab = "play"
            elif 350 <= x < 700:
                self.active_tab = "lab"
            return
        if self.modal_open:
            self._handle_modal_click(x, y, pygame)
            return
        if self.active_tab == "lab":
            self._handle_lab_click(x, y)
            return
        if self.promotion_pending is not None:
            self._handle_promotion_click(x, y)
            return
        if x < self.LEFT_WIDTH:
            self._handle_left_click(x, y, pygame)
        elif x < self.LEFT_WIDTH + self.ARENA_WIDTH:
            self._handle_board_click(x - self.LEFT_WIDTH, y - self.TAB_HEIGHT)
        else:
            self._handle_right_click(x - self.LEFT_WIDTH - self.ARENA_WIDTH, y - self.TAB_HEIGHT)

    def _handle_left_click(self, x: int, y: int, pygame: Any = None) -> None:
        local_y = y - self.TAB_HEIGHT
        ids = list_full_board_agents()
        agent_index = next(
            (
                index
                for index in range(len(ids))
                if 78 + index * 28 <= local_y < 102 + index * 28
            ),
            None,
        )
        if agent_index is not None:
            agent_id = ids[agent_index]
            self.agent_id = agent_id
            if self.game_mode == "human_ai":
                ai_id = self.agent_id
                self.agent = create_agent(ai_id)
                if self.player_color == "white":
                    self.black_agent = self.agent
                else:
                    self.white_agent = self.agent
            self.message = f"Đã chọn {self.agent_id}"
        elif 272 <= local_y <= 306:
            self._toggle_budget()
        elif 350 <= local_y <= 382:
            self.flipped = not self.flipped
        elif 390 <= local_y <= 422:
            self._undo()
        elif 430 <= local_y <= 462:
            self._resign()
        elif 470 <= local_y <= 502:
            self._offer_draw()
        elif 584 <= local_y <= 630:
            self._new_game()

    def _handle_board_click(self, x: int, y: int) -> None:
        if y < 48 or y >= 48 + self.BOARD_SIZE:
            return
        board_y = y - 48
        if self.ai_future is not None and not self.ai_future.done():
            return
        if self.env.is_terminal() or not self._is_human_turn():
            return
        if self.replay_index != len(self.records):
            self._restore_index(len(self.records))
        square = self._square_at(x, board_y)
        if self.selected_square is None:
            legal = self._legal_from(square)
            if legal:
                self.selected_square = square
            return
        action = self._find_action(self.selected_square, square)
        if action is None:
            self.selected_square = square if self._legal_from(square) else None
            return
        promotion_moves = [
            move for move in self.env.legal_actions()
            if move.from_square == action.from_square
            and move.to_square == action.to_square
            and move.promotion is not None
        ]
        if promotion_moves:
            self.promotion_pending = promotion_moves
        else:
            self._play_action(action)

    def _handle_promotion_click(self, x: int, y: int) -> None:
        board_x = 270 + (670 - self.BOARD_SIZE) // 2
        board_y = self.TAB_HEIGHT + 48
        choices = [("q", 0), ("r", 1), ("b", 2), ("n", 3)]
        cx = board_x + 150
        cy = board_y + 220
        index = None
        for symbol, offset in choices:
            rect = (cx + offset * 54, cy, 50, 50)
            if rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]:
                index = symbol
                break
        if index is not None:
            import chess
            piece_type = {
                "q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT
            }[index]
            move = next(
                (move for move in self.promotion_pending if move.promotion == piece_type),
                None,
            )
            self.promotion_pending = None
            if move is not None:
                self._play_action(move)

    def _handle_right_click(self, x: int, y: int) -> None:
        replay_y = 631
        if replay_y <= y <= replay_y + 34:
            index = (x - 12) // 79
            if index == 0:
                self._restore_index(0)
            elif index == 1:
                self._restore_index(self.replay_index - 1)
            elif index == 2:
                self._restore_index(self.replay_index + 1)
            elif index == 3:
                self._restore_index(len(self.records))

    def _handle_lab_click(self, x: int, y: int) -> None:
        if 18 <= x <= 322:
            scenarios = [
                ("arena", 100),
                ("ablation1", 138),
                ("ablation2", 176),
                ("logs", 214),
            ]
            for key, top in scenarios:
                if top <= y <= top + 32:
                    self.lab_scenario = key
                    self.lab_view = key
                    break
        if 340 <= x <= 1280 and 44 <= y <= 88:
            for index, key in enumerate(("arena", "ablation1", "ablation2", "logs")):
                if 360 + index * 220 <= x <= 570 + index * 220:
                    self.lab_view = key
                    break
        if 18 <= x <= 163 and 600 <= y <= 640:
            self._start_lab()
        elif 177 <= x <= 322 and 600 <= y <= 640:
            self.lab_stop_event.set()

    def _handle_modal_click(self, x: int, y: int, pygame: Any) -> None:
        mx, my = 380, 140
        w, h = 520, 480
        if not (mx <= x <= mx + w and my <= y <= my + h):
            return
        lx, ly = x - mx, y - my
        if 58 <= ly <= 92:
            if 25 <= lx <= 170:
                self.modal_mode = "human_human"
            elif 180 <= lx <= 325:
                self.modal_mode = "human_ai"
            elif 335 <= lx <= 480:
                self.modal_mode = "ai_ai"
        if 132 <= ly <= 166:
            if 25 <= lx <= 245:
                if self.modal_mode == "human_ai":
                    self.modal_human_color = "white"
                elif self.modal_mode == "ai_ai":
                    ids = list_full_board_agents()
                    index = ids.index(self.modal_white_agent) if self.modal_white_agent in ids else 0
                    self.modal_white_agent = ids[(index + 1) % len(ids)]
            elif 270 <= lx <= 490:
                if self.modal_mode == "human_ai":
                    self.modal_human_color = "black"
                elif self.modal_mode == "ai_ai":
                    ids = list_full_board_agents()
                    index = ids.index(self.modal_black_agent) if self.modal_black_agent in ids else 0
                    self.modal_black_agent = ids[(index + 1) % len(ids)]
            if self.modal_mode == "human_human":
                pass
        if 215 <= ly <= 285:
            col = (lx - 25) // 118
            row = (ly - 215) // 38
            index = row * 4 + col
            options = [(0, 0), (60, 0), (180, 2), (300, 0), (600, 5), (900, 10), (1800, 20)]
            if 0 <= index < len(options):
                self.modal_base, self.modal_increment = options[index]
        if 325 <= ly <= 359:
            if 25 <= lx <= 245:
                self.modal_budget = "time"
                self.modal_time = 1.0 if self.modal_time not in (0.5, 1.0, 2.0) else self.modal_time
            elif 270 <= lx <= 490:
                self.modal_budget = "workload"
        if 400 <= ly <= 440:
            if 40 <= lx <= 190:
                self.modal_open = False
            elif 300 <= lx <= 470:
                self._start_configured_game()

    def _cycle_agent(self) -> None:
        ids = list_full_board_agents()
        next_index = (ids.index(self.agent_id) + 1) % len(ids)
        self.agent_id = ids[next_index]
        self.agent = create_agent(self.agent_id)

    def _toggle_budget(self) -> None:
        if self.time_limit is not None:
            self.time_limit = None
            self.max_workload = 400
        else:
            self.time_limit = 1.0
            self.max_workload = None

    def _legal_from(self, square: int) -> list[Any]:
        return [move for move in self.env.legal_actions() if move.from_square == square]

    def _find_action(self, source: int, target: int) -> Optional[Any]:
        for move in self.env.legal_actions():
            if (
                move.from_square == source
                and move.to_square == target
                and move.promotion is None
            ):
                return move
        return next(
            (
                move for move in self.env.legal_actions()
                if move.from_square == source and move.to_square == target
            ),
            None,
        )

    def _play_action(self, action: Any) -> None:
        started = time.perf_counter()
        san = self.env.san(action)
        side = self._side_to_move()
        self.env.step(action)
        (self.white_clock if side == "white" else self.black_clock).add_increment()
        self.records = self.records[:self.replay_index]
        self.records.append(
            MoveRecord(action, san, self.env.clone(), time.perf_counter() - started)
        )
        self.replay_index = len(self.records)
        self.selected_square = None
        self.message = self._status_text()

    def _agent_for_side(self, side: str) -> Optional[Any]:
        if side == "white":
            return self.white_agent
        return self.black_agent

    def _start_ai(self) -> None:
        if self.env.is_terminal() or self.replay_index != len(self.records):
            return
        if self.ai_future is not None:
            return
        side = self._side_to_move()
        if self.game_mode == "human_ai" and side == self.player_color:
            return
        if self.game_mode == "human_human":
            return
        agent = self._agent_for_side(side)
        if agent is None:
            return
        snapshot = self.env.clone()
        self.ai_started = time.perf_counter()
        self.ai_agent = agent
        self.message = f"{agent.name} đang suy nghĩ..."
        self.ai_future = self.executor.submit(
            agent.select_move,
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
            action = future.result()
            self._play_action(action)
            if self.records:
                last = self.records[-1]
                self.records[-1] = MoveRecord(
                    last.action,
                    last.san,
                    last.snapshot,
                    time.perf_counter() - self.ai_started,
                )
        except Exception as exc:
            self.message = f"Lỗi AI: {exc}"

    def _status_text(self) -> str:
        if self.resign_result is not None:
            winner = "Trắng thắng" if self.resign_result == 1.0 else "Đen thắng" if self.resign_result == -1.0 else "Hòa"
            return f"{winner} - đầu hàng/ngoài cuộc"
        if self.env.is_terminal():
            result = self.env.result()
            winner = "Trắng thắng" if result == 1.0 else "Đen thắng" if result == -1.0 else "Hòa"
            reason = self.env.termination_reason() or "Kết thúc ván"
            return f"{winner} - {reason}"
        return "Lượt Trắng" if self._side_to_move() == "white" else "Lượt Đen"

    def _resign(self) -> None:
        side = self._side_to_move()
        if self.ai_future is not None and not self.ai_future.done():
            self.ai_future.cancel()
        self.resign_result = -1.0 if side == "white" else 1.0
        self.message = "Trắng thắng do Đen đầu hàng" if side == "black" else "Đen thắng do Trắng đầu hàng"
        self.clock_running = False

    def _offer_draw(self) -> None:
        if self.game_mode == "ai_ai":
            return
        if self.game_mode == "human_human":
            if self.draw_offer_pending:
                self.resign_result = 0.0
                self.draw_offer_pending = False
                self.message = "Hai bên đồng ý hòa"
                self.clock_running = False
            else:
                self.draw_offer_pending = True
                self.message = "Đề nghị hòa: bấm [D] lần nữa để đồng ý"
            return
        agent = self.agent
        stats = getattr(agent, "last_search_stats", {}) or {}
        eval_score = float(stats.get("eval", 0.0))
        if eval_score <= 20.0:
            self.message = "AI đồng ý hòa"
            self.clock_running = False
        else:
            self.message = "AI từ chối hòa"

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

    def _format_clock(self, clock: _Clock) -> str:
        if clock.seconds <= 0 and self.modal_base == 0:
            return "∞"
        seconds = int(max(0, clock.seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def _draw(self, screen: Any, pygame: Any) -> None:
        screen.fill(self.BG)
        self._draw_tabs(screen, pygame)
        if self.active_tab == "play":
            self._draw_play_tab(screen, pygame)
        else:
            self._draw_lab_tab(screen, pygame)
        if self.modal_open:
            self._draw_modal(screen, pygame)

    def _draw_tabs(self, screen: Any, pygame: Any) -> None:
        pygame.draw.rect(screen, (11, 15, 23), pygame.Rect(0, 0, 1280, 44))
        pygame.draw.line(screen, self.BORDER, (0, 43), (1280, 43), 1)
        tabs = [
            ("play", "♟  Thi đấu & Phân tích"),
            ("lab", "📊  Nghiên cứu & Thực nghiệm"),
        ]
        for index, (key, label) in enumerate(tabs):
            rect = pygame.Rect(index * 350, 0, 350, 44)
            color = self.SURFACE if self.active_tab == key else (14, 19, 28)
            pygame.draw.rect(screen, color, rect)
            if self.active_tab == key:
                pygame.draw.rect(screen, self.ACCENT, pygame.Rect(rect.x, 41, rect.width, 3))
            font = pygame.font.SysFont("segoeui", 17, bold=True)
            muted = self.ACCENT if self.active_tab == key else self.MUTED
            screen.blit(font.render(label, True, muted), (rect.x + 18, 11))

    def _draw_play_tab(self, screen: Any, pygame: Any) -> None:
        self._draw_config(screen, pygame)
        self._draw_arena(screen, pygame)
        self._draw_telemetry(screen, pygame)

    def _draw_config(self, screen: Any, pygame: Any) -> None:
        rect = pygame.Rect(0, self.TAB_HEIGHT, self.LEFT_WIDTH, 716)
        pygame.draw.rect(screen, (20, 25, 36), rect)
        pygame.draw.line(screen, self.BORDER, (269, 44), (269, 760), 1)
        font = pygame.font.SysFont("segoeui", 15, bold=True)
        small = pygame.font.SysFont("segoeui", 13)
        mono = pygame.font.SysFont("consolas", 12)

        screen.blit(font.render("AGENT SELECTION", True, self.MUTED), (16, 58))
        ids = list_full_board_agents()
        names = {
            "level1": ("Random", "L1"),
            "level2": ("Heuristic", "L2"),
            "level3": ("Alpha-Beta", "L3"),
            "level4": ("MCTS UCT", "L4"),
            "level5": ("ResNet", "L5"),
            "level7": ("PUCT", "L7"),
            "level8": ("AlphaZero", "L8"),
        }
        for index, agent_id in enumerate(ids):
            y = 78 + index * 28
            item = pygame.Rect(12, y, 246, 26)
            selected = agent_id == self.agent_id
            pygame.draw.rect(
                screen,
                (34, 67, 112) if selected else self.SURFACE,
                item,
                border_radius=7,
            )
            if selected:
                pygame.draw.rect(screen, self.ACCENT, item, 2, border_radius=7)
            label, badge = names.get(agent_id, (agent_id, ""))
            screen.blit(small.render(f"{agent_id.replace('level', 'L')}", True,
                                     self.ACCENT if selected else self.MUTED),
                        (20, y + 6))
            screen.blit(small.render(label, True, self.TEXT), (54, y + 6))
            if badge:
                badge_rect = pygame.Rect(216, y + 4, 34, 18)
                pygame.draw.rect(screen, self.PANEL_3, badge_rect, border_radius=4)
                screen.blit(mono.render(badge, True, self.MUTED),
                            badge_rect.move(7, 2))

        screen.blit(font.render("BUDGET & SEARCH", True, self.MUTED), (16, 290))
        budget = f"Time: {self.time_limit:.1f}s" if self.time_limit is not None else f"Nodes/Sim: {self.max_workload}"
        pygame.draw.rect(screen, self.SURFACE, pygame.Rect(12, 316, 246, 38), border_radius=7)
        pygame.draw.rect(screen, self.BORDER, pygame.Rect(12, 316, 246, 38), 1, border_radius=7)
        screen.blit(mono.render(budget, True, self.TEXT), (20, 327))

        screen.blit(font.render("CONTROLS", True, self.MUTED), (16, 370))
        buttons = [
            ("Đảo bàn", "F", 394),
            ("Undo", "U", 434),
            ("Đầu hàng", "R", 474),
            ("Cầu hòa", "D", 514),
        ]
        for label, key, y in buttons:
            rect = pygame.Rect(12, y, 246, 34)
            visible = label != "Cầu hòa" or self.game_mode != "ai_ai"
            if visible:
                pygame.draw.rect(screen, self.SURFACE, rect, border_radius=7)
                pygame.draw.rect(screen, self.BORDER, rect, 1, border_radius=7)
                screen.blit(small.render(label, True, self.TEXT), (20, y + 8))
                key_rect = pygame.Rect(208, y + 6, 42, 22)
                pygame.draw.rect(screen, self.PANEL_3, key_rect, border_radius=5)
                key_text = mono.render(key, True, self.MUTED)
                screen.blit(key_text, key_text.get_rect(center=key_rect.center))
        new_rect = pygame.Rect(12, 628, 246, 48)
        pygame.draw.rect(screen, self.ACCENT, new_rect, border_radius=7)
        pygame.draw.rect(screen, (132, 190, 255), new_rect, 1, border_radius=7)
        screen.blit(font.render("VÁN MỚI", True, (10, 22, 38)), (64, 641))
        key_text = mono.render("[N]", True, (24, 48, 72))
        screen.blit(key_text, (174, 645))

    def _draw_arena(self, screen: Any, pygame: Any) -> None:
        arena_x = self.LEFT_WIDTH
        rect = pygame.Rect(arena_x, self.TAB_HEIGHT, self.Arena_width if hasattr(self, "Arena_width") else self.ARENA_WIDTH, 716)
        pygame.draw.rect(screen, (18, 23, 33), rect)
        board_x = arena_x + (self.ARENA_WIDTH - self.BOARD_SIZE) // 2
        board_y = self.TAB_HEIGHT + 48
        top = "Đen" if not self.flipped else "Trắng"
        bottom = "Trắng" if not self.flipped else "Đen"
        top_clock = self.black_clock if top == "Đen" else self.white_clock
        bottom_clock = self.white_clock if bottom == "Trắng" else self.black_clock
        self._draw_player_bar(screen, pygame, arena_x + 63, self.TAB_HEIGHT + 4, 544, top, top_clock)
        self._draw_board(screen, pygame, board_x, board_y)
        self._draw_player_bar(screen, pygame, arena_x + 63, self.TAB_HEIGHT + 592, 544, bottom, bottom_clock)
        footer = pygame.Rect(arena_x + 63, self.TAB_HEIGHT + 640, 544, 34)
        pygame.draw.rect(screen, (15, 21, 31), footer, border_radius=7)
        pygame.draw.rect(screen, self.BORDER, footer, 1, border_radius=7)
        fen = self.env.render("fen")
        small = pygame.font.SysFont("consolas", 11)
        label = small.render("FEN", True, self.ACCENT)
        screen.blit(label, (footer.x + 10, footer.y + 10))
        fen_text = small.render(str(fen)[:68], True, self.TEXT)
        fen_rect = fen_text.get_rect(midleft=(footer.x + 48, footer.centery))
        screen.blit(fen_text, fen_rect)

    def _draw_player_bar(self, screen: Any, pygame: Any, x: int, y: int, width: int, side: str, clock: _Clock) -> None:
        rect = pygame.Rect(x, y, width, 40)
        is_turn = (
            (side == "Trắng" and self._side_to_move() == "white")
            or (side == "Đen" and self._side_to_move() == "black")
        )
        pygame.draw.rect(screen, (30, 48, 72) if is_turn else self.SURFACE, rect, border_radius=8)
        if is_turn:
            pygame.draw.rect(screen, self.ACCENT, rect, 2, border_radius=8)
        font = pygame.font.SysFont("segoeui", 15, bold=True)
        mono = pygame.font.SysFont("consolas", 17, bold=True)
        name = "Người chơi" if (
            (side == "Trắng" and self.player_color == "white")
            or (side == "Đen" and self.player_color == "black")
        ) else (
            getattr(self._agent_for_side("white" if side == "Trắng" else "black"), "name", "AI")
        )
        role = "♙" if side == "Trắng" else "♟"
        screen.blit(font.render(f"{role}  {side}  ·  {name}", True, self.TEXT), (x + 12, y + 9))
        clock_text = self._format_clock(clock)
        clock_surface = mono.render(clock_text, True, self.ORANGE if is_turn else self.TEXT)
        clock_box = pygame.Rect(x + width - 94, y + 6, 82, 28)
        pygame.draw.rect(screen, (10, 15, 23), clock_box, border_radius=6)
        screen.blit(clock_surface, clock_surface.get_rect(center=clock_box.center))

    def _draw_board(self, screen: Any, pygame: Any, board_x: int, board_y: int) -> None:
        tile = self.BOARD_SIZE // 8
        board_rect = pygame.Rect(board_x - 3, board_y - 3, self.BOARD_SIZE + 6, self.BOARD_SIZE + 6)
        pygame.draw.rect(screen, (9, 12, 18), board_rect, border_radius=4)
        pygame.draw.rect(screen, (82, 94, 112), board_rect, 2, border_radius=4)
        pieces = self._pieces()
        legal_targets = {move.to_square for move in self._legal_from(self.selected_square)} if self.selected_square is not None else set()
        last_squares = set()
        if self.records:
            last = self.records[self.replay_index - 1] if self.replay_index > 0 else None
            if last is not None:
                last_squares = {last.action.from_square, last.action.to_square}
        for square in range(64):
            sx, sy = self._screen_square(square)
            rect = pygame.Rect(board_x + sx * tile, board_y + sy * tile, tile, tile)
            color = self.LIGHT if (sx + sy) % 2 == 0 else self.DARK
            pygame.draw.rect(screen, color, rect)
            if square in last_squares:
                overlay = pygame.Surface((tile, tile), pygame.SRCALPHA)
                overlay.fill((*self.LAST_MOVE, 150))
                screen.blit(overlay, rect.topleft)
            if square == self.selected_square:
                pygame.draw.rect(screen, self.HIGHLIGHT, rect, 5)
            if square in legal_targets:
                pygame.draw.circle(screen, self.HIGHLIGHT, rect.center, 9)
            if square in pieces:
                symbol = pieces[square]
                if symbol in self.piece_images:
                    image = self.piece_images[symbol]
                    screen.blit(image, image.get_rect(center=rect.center))
                else:
                    font = pygame.font.SysFont("segoeuisymbol", 52)
                    glyph = {
                        "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
                        "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
                    }[symbol]
                    text_rect = font.render(glyph, True, (245, 248, 250)).get_rect(center=rect.center)
                    if symbol.isupper():
                        outline = (55, 65, 78)
                        for offset_x, offset_y in (
                            (-2, 0), (2, 0), (0, -2), (0, 2),
                            (-1, -1), (1, -1), (-1, 1), (1, 1),
                        ):
                            outline_text = font.render(glyph, True, outline)
                            screen.blit(
                                outline_text,
                                outline_text.get_rect(
                                    center=(text_rect.centerx + offset_x, text_rect.centery + offset_y)
                                ),
                            )
                        text = font.render(glyph, True, (250, 252, 255))
                    else:
                        text = font.render(glyph, True, (28, 34, 43))
                    screen.blit(text, text_rect)
        small = pygame.font.SysFont("segoeui", 12, bold=True)
        files = "abcdefgh" if not self.flipped else "hgfedcba"
        ranks = "12345678" if self.flipped else "87654321"
        for i, file_name in enumerate(files):
            label = small.render(file_name, True, self.TEXT)
            label_rect = label.get_rect(center=(board_x + i * tile + tile // 2, board_y + self.BOARD_SIZE - 11))
            screen.blit(label, label_rect)
        for i, rank_name in enumerate(ranks):
            label = small.render(rank_name, True, self.TEXT)
            label_rect = label.get_rect(center=(board_x + 10, board_y + i * tile + 12))
            screen.blit(label, label_rect)

        if self.promotion_pending is not None:
            cx, cy = board_x + 150, board_y + 220
            pygame.draw.rect(screen, (20, 24, 32), pygame.Rect(cx - 10, cy - 10, 236, 70), border_radius=8)
            for index, (symbol, glyph) in enumerate((("q", "♕"), ("r", "♖"), ("b", "♗"), ("n", "♘"))):
                rect = pygame.Rect(cx + index * 54, cy, 50, 50)
                pygame.draw.rect(screen, self.PANEL_2, rect, border_radius=5)
                screen.blit(pygame.font.SysFont("segoeui", 34).render(glyph, True, self.TEXT), rect.move(8, 2))

    def _draw_telemetry(self, screen: Any, pygame: Any) -> None:
        rect = pygame.Rect(self.LEFT_WIDTH + self.ARENA_WIDTH, self.TAB_HEIGHT, self.RIGHT_WIDTH, 716)
        pygame.draw.rect(screen, (20, 25, 36), rect)
        pygame.draw.line(screen, self.BORDER, (940, 44), (940, 760), 1)
        font = pygame.font.SysFont("segoeui", 14, bold=True)
        small = pygame.font.SysFont("segoeui", 12)
        mono = pygame.font.SysFont("consolas", 12)

        screen.blit(font.render("EVALUATION & WIN PROBABILITY", True, self.MUTED), (960, 58))
        stats = getattr(self.ai_agent or self.agent, "last_search_stats", {}) or {}
        evaluation = float(stats.get("eval", 0.0))
        eval_value = max(-1000.0, min(1000.0, evaluation))
        win_probability = 50.0 + 50.0 * (evaluation / (abs(evaluation) + 200.0))
        summary_card = pygame.Rect(952, 82, 316, 132)
        pygame.draw.rect(screen, self.SURFACE, summary_card, border_radius=9)
        pygame.draw.rect(screen, self.BORDER, summary_card, 1, border_radius=9)
        pygame.draw.rect(screen, (12, 17, 25), pygame.Rect(968, 94, 18, 106), border_radius=5)
        fill = int(118 * ((eval_value + 1000.0) / 2000.0))
        bar_y = 200 - fill
        bar_color = self.GREEN if evaluation >= 0 else (248, 113, 113)
        pygame.draw.rect(screen, bar_color, pygame.Rect(968, bar_y, 18, fill), border_radius=5)
        screen.blit(mono.render(f"{evaluation:+.2f}", True, self.TEXT), (998, 108))
        screen.blit(small.render("EVALUATION", True, self.MUTED), (998, 132))
        gauge = pygame.Rect(1138, 94, 112, 112)
        pygame.draw.circle(screen, (12, 17, 25), gauge.center, 48)
        pygame.draw.circle(screen, self.BORDER_STRONG, gauge.center, 48, 7)
        arc_angle = int(360 * max(0.0, min(1.0, win_probability / 100.0)))
        for angle in range(-90, -90 + arc_angle, 8):
            end = angle + 8
            pygame.draw.arc(screen, self.ACCENT, gauge.inflate(-12, -12),
                            angle * 3.14159 / 180, end * 3.14159 / 180, 7)
        probability_text = mono.render(f"{win_probability:.1f}%", True, self.TEXT)
        screen.blit(probability_text, probability_text.get_rect(center=(gauge.centerx, gauge.centery - 4)))
        chance_label = small.render("WIN", True, self.MUTED)
        screen.blit(chance_label, chance_label.get_rect(center=(gauge.centerx, gauge.centery + 18)))

        y = 230
        screen.blit(font.render("AI SEARCH METRICS", True, self.MUTED), (960, y))
        metrics = [
            ("NPS / SPS", f"{stats.get('nodes', 0):.0f} / {stats.get('simulations', 0):.0f}"),
            ("Search Depth", f"{stats.get('depth', 0):.0f}"),
            ("TT Hits", f"{stats.get('tt_hits', 0):.0f}"),
            ("MCTS Sims", f"{stats.get('simulations', 0):.0f}"),
            ("Latency ms", f"{stats.get('time_sec', 0) * 1000:.0f}"),
        ]
        y = 258
        for label, value in metrics:
            metric_rect = pygame.Rect(952, y, 316, 32)
            pygame.draw.rect(screen, self.SURFACE, metric_rect, border_radius=7)
            pygame.draw.rect(screen, self.BORDER, metric_rect, 1, border_radius=7)
            pygame.draw.rect(screen, self.ACCENT, pygame.Rect(952, y + 7, 3, 18), border_radius=2)
            screen.blit(small.render(label, True, self.MUTED), (964, y + 8))
            value_surface = mono.render(str(value), True, self.TEXT)
            screen.blit(value_surface, (1256 - value_surface.get_width(), y + 8))
            y += 38

        y = 445
        screen.blit(font.render("MOVE HISTORY (SAN)", True, self.MUTED), (960, y))
        y = 475
        history = []
        for i in range(0, len(self.records), 2):
            item = f"{i // 2 + 1}. {self.records[i].san}"
            if i + 1 < len(self.records):
                item += f"  {self.records[i + 1].san}"
            history.append(item)
        pygame.draw.rect(screen, (15, 21, 31), pygame.Rect(952, 468, 316, 184), border_radius=9)
        pygame.draw.rect(screen, self.BORDER, pygame.Rect(952, 468, 316, 184), 1, border_radius=9)
        y = 480
        for index, line in enumerate(history[-7:]):
            row = pygame.Rect(960, y - 5, 300, 22)
            if index % 2 == 0:
                pygame.draw.rect(screen, self.SURFACE, row, border_radius=5)
            screen.blit(mono.render(line, True, self.TEXT), (968, y))
            y += 23

        replay_y = 675
        labels = ["|<", "<", ">", ">|"]
        for index, label in enumerate(labels):
            rect = pygame.Rect(952 + index * 79, replay_y, 70, 34)
            pygame.draw.rect(screen, self.SURFACE, rect, border_radius=7)
            pygame.draw.rect(screen, self.BORDER, rect, 1, border_radius=7)
            label_surface = mono.render(label, True, self.TEXT)
            screen.blit(label_surface, label_surface.get_rect(center=rect.center))

    def _draw_modal(self, screen: Any, pygame: Any) -> None:
        overlay = pygame.Surface(self.WINDOW_SIZE, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))
        mx, my, w, h = 380, 140, 520, 480
        rect = pygame.Rect(mx, my, w, h)
        pygame.draw.rect(screen, self.PANEL, rect, border_radius=10)
        pygame.draw.rect(screen, self.BORDER, rect, 2, border_radius=10)
        font = pygame.font.SysFont("segoeui", 20)
        small = pygame.font.SysFont("segoeui", 14)
        screen.blit(font.render("THIẾT LẬP VÁN ĐẤU MỚI", True, self.TEXT), (mx + 105, my + 18))

        modes = [("human_human", "Người vs Người"), ("human_ai", "Người vs Máy"), ("ai_ai", "Máy vs Máy")]
        for index, (key, label) in enumerate(modes):
            button = pygame.Rect(mx + 25 + index * 155, my + 58, 145, 34)
            pygame.draw.rect(screen, self.ACCENT if self.modal_mode == key else self.PANEL_2, button, border_radius=5)
            screen.blit(small.render(label, True, self.TEXT), (button.x + 8, button.y + 8))

        screen.blit(small.render("Bên TRẮNG", True, self.MUTED), (mx + 25, my + 112))
        screen.blit(small.render("Bên ĐEN", True, self.MUTED), (mx + 270, my + 112))
        pygame.draw.rect(screen, self.PANEL_2, pygame.Rect(mx + 25, my + 132, 220, 34), border_radius=5)
        pygame.draw.rect(screen, self.PANEL_2, pygame.Rect(mx + 270, my + 132, 220, 34), border_radius=5)
        white_label = "Người" if self.modal_mode == "human_human" or (self.modal_mode == "human_ai" and self.modal_human_color == "white") else self.modal_white_agent
        black_label = "Người" if self.modal_mode == "human_human" or (self.modal_mode == "human_ai" and self.modal_human_color == "black") else self.modal_black_agent
        screen.blit(small.render(white_label, True, self.TEXT), (mx + 35, my + 140))
        screen.blit(small.render(black_label, True, self.TEXT), (mx + 280, my + 140))

        screen.blit(small.render("Đồng hồ (base + increment)", True, self.MUTED), (mx + 25, my + 190))
        options = [(0, 0), (60, 0), (180, 2), (300, 0), (600, 5), (900, 10), (1800, 20)]
        for index, (base, inc) in enumerate(options):
            col, row = index % 4, index // 4
            button = pygame.Rect(mx + 25 + col * 118, my + 215 + row * 38, 108, 32)
            selected = self.modal_base == base and self.modal_increment == inc
            pygame.draw.rect(screen, self.ACCENT if selected else self.PANEL_2, button, border_radius=5)
            label = "∞" if base == 0 else f"{base // 60}+{inc}"
            screen.blit(small.render(label, True, self.TEXT), (button.x + 35, button.y + 7))

        screen.blit(small.render("Ngân sách AI", True, self.MUTED), (mx + 25, my + 300))
        pygame.draw.rect(screen, self.ACCENT if self.modal_budget == "time" else self.PANEL_2, pygame.Rect(mx + 25, my + 325, 220, 34), border_radius=5)
        pygame.draw.rect(screen, self.ACCENT if self.modal_budget == "workload" else self.PANEL_2, pygame.Rect(mx + 270, my + 325, 220, 34), border_radius=5)
        screen.blit(small.render(f"Time {self.modal_time:.1f}s", True, self.TEXT), (mx + 35, my + 333))
        screen.blit(small.render(f"Workload {self.modal_workload}", True, self.TEXT), (mx + 280, my + 333))

        cancel = pygame.Rect(mx + 40, my + 400, 150, 42)
        start = pygame.Rect(mx + 300, my + 400, 170, 42)
        pygame.draw.rect(screen, self.PANEL_2, cancel, border_radius=6)
        pygame.draw.rect(screen, self.ACCENT, start, border_radius=6)
        screen.blit(font.render("Hủy", True, self.TEXT), (cancel.x + 50, cancel.y + 7))
        screen.blit(font.render("Bắt đầu", True, (10, 22, 38)), (start.x + 35, start.y + 7))

    def _draw_lab_tab(self, screen: Any, pygame: Any) -> None:
        pygame.draw.rect(screen, (16, 21, 30), pygame.Rect(0, 44, 1280, 716))
        pygame.draw.rect(screen, (20, 25, 36), pygame.Rect(0, 44, 340, 716))
        pygame.draw.line(screen, self.BORDER, (339, 44), (339, 760), 1)
        font = pygame.font.SysFont("segoeui", 16, bold=True)
        small = pygame.font.SysFont("segoeui", 13)
        mono = pygame.font.SysFont("consolas", 13)
        screen.blit(font.render("EXPERIMENT RUNNER", True, self.MUTED), (18, 64))
        scenarios = [
            ("arena", "Đấu số lượng lớn (Arena)"),
            ("ablation1", "Ablation 1 (WAC-50)"),
            ("ablation2", "Ablation 2 (MCTS N_sim)"),
            ("logs", "Nạp từ Log"),
        ]
        for index, (key, label) in enumerate(scenarios):
            y = 100 + index * 38
            selected = self.lab_scenario == key
            card = pygame.Rect(18, y, 304, 32)
            pygame.draw.rect(screen, (34, 67, 112) if selected else self.SURFACE,
                             card, border_radius=7)
            pygame.draw.rect(screen, self.ACCENT if selected else self.BORDER,
                             card, 2 if selected else 1, border_radius=7)
            screen.blit(small.render(label, True, self.TEXT), (28, y + 7))
            if selected:
                pygame.draw.rect(screen, self.ACCENT, pygame.Rect(306, y + 8, 5, 16),
                                 border_radius=2)

        config_card = pygame.Rect(18, 258, 304, 82)
        pygame.draw.rect(screen, self.SURFACE, config_card, border_radius=9)
        pygame.draw.rect(screen, self.BORDER, config_card, 1, border_radius=9)
        screen.blit(small.render("CẤU HÌNH", True, self.MUTED), (30, 268))
        screen.blit(mono.render(f"Số ván/cặp  {self.lab_num_games}", True, self.TEXT), (30, 292))
        screen.blit(mono.render(f"Workers      {self.lab_workers}", True, self.TEXT), (30, 314))
        progress_width = int(304 * self.lab_progress)
        pygame.draw.rect(screen, (10, 15, 23), pygame.Rect(18, 356, 304, 18), border_radius=9)
        if progress_width:
                pygame.draw.rect(screen, self.GREEN, pygame.Rect(18, 356, progress_width, 18), border_radius=9)
        screen.blit(mono.render(f"{int(self.lab_progress * 100):3d}%  {self.lab_message}", True, self.TEXT), (18, 384))

        pygame.draw.rect(screen, self.ACCENT, pygame.Rect(18, 600, 145, 42), border_radius=7)
        pygame.draw.rect(screen, (180, 74, 74), pygame.Rect(177, 600, 145, 42), border_radius=7)
        screen.blit(small.render("Chạy thực nghiệm", True, (10, 22, 38)), (29, 614))
        screen.blit(small.render("Dừng", True, self.TEXT), (226, 614))

        pygame.draw.rect(screen, (20, 25, 36), pygame.Rect(340, 44, 940, 44))
        views = ["arena", "ablation1", "ablation2", "logs"]
        for index, key in enumerate(views):
            rect = pygame.Rect(360 + index * 220, 50, 210, 32)
            selected = self.lab_view == key
            pygame.draw.rect(screen, (34, 67, 112) if selected else self.SURFACE,
                             rect, border_radius=7)
            pygame.draw.rect(screen, self.ACCENT if selected else self.BORDER,
                             rect, 1, border_radius=7)
            screen.blit(small.render(key.upper(), True,
                                     self.ACCENT if selected else self.MUTED),
                        (rect.x + 12, rect.y + 7))

        chart_rect = pygame.Rect(360, 108, 900, 400)
        pygame.draw.rect(screen, (12, 17, 25), chart_rect, border_radius=10)
        pygame.draw.rect(screen, self.BORDER, chart_rect, 1, border_radius=10)
        screen.blit(font.render("BIỂU ĐỒ / BẢNG SỐ LIỆU", True, self.MUTED), (380, 126))
        if self.lab_results:
            text = json.dumps(self.lab_results, ensure_ascii=False)[:900]
            y = 164
            for line in text.splitlines() or [text]:
                screen.blit(mono.render(line[:120], True, self.TEXT), (380, y))
                y += 22
        else:
            icon_rect = pygame.Rect(720, 180, 180, 180)
            pygame.draw.circle(screen, self.SURFACE, icon_rect.center, 58)
            pygame.draw.circle(screen, self.BORDER, icon_rect.center, 58, 2)
            pygame.draw.rect(screen, self.ACCENT, pygame.Rect(690, 292, 42, 52),
                             border_radius=5)
            pygame.draw.rect(screen, self.GREEN, pygame.Rect(744, 264, 42, 80),
                             border_radius=5)
            pygame.draw.rect(screen, (248, 181, 73), pygame.Rect(798, 236, 42, 108),
                             border_radius=5)
            empty = small.render("Chưa có dữ liệu thực nghiệm", True, self.TEXT)
            screen.blit(empty, empty.get_rect(center=(810, 380)))
            hint = small.render("Chọn kịch bản, cấu hình rồi nhấn Chạy thực nghiệm", True, self.MUTED)
            screen.blit(hint, hint.get_rect(center=(810, 406)))

    def _start_lab(self) -> None:
        if self.lab_running:
            return
        self.lab_running = True
        self.lab_stop_event.clear()
        self.lab_progress = 0.0
        thread = threading.Thread(target=self._run_lab_worker, daemon=True)
        thread.start()

    def _run_lab_worker(self) -> None:
        try:
            if self.lab_scenario == "ablation1":
                from chess_ai.experiments.ablation_search import run_ablation_search
                self.lab_results = run_ablation_search()
            elif self.lab_scenario == "ablation2":
                from chess_ai.experiments.ablation_mcts import run_ablation_mcts
                self.lab_results = run_ablation_mcts()
            elif self.lab_scenario == "arena":
                from chess_ai.arena.tournament import run_paired_matchup
                agent_a = create_agent(self.lab_agents[0])
                agent_b = create_agent(self.lab_agents[1] if len(self.lab_agents) > 1 else self.lab_agents[0])
                from chess_ai.utils.epd import load_epd_file
                openings = load_epd_file("resources/openings_50.epd")[:min(25, self.lab_num_games)]
                self.lab_results = run_paired_matchup(
                    agent_a,
                    agent_b,
                    openings,
                    seeds=[42],
                    budget_mode="workload_control",
                    workload_a=self.max_workload or 400,
                    workload_b=self.max_workload or 400,
                    workers=self.lab_workers,
                )
            else:
                paths = [
                    "results/tournament_summary.json",
                    "results/ablation1/ablation1_results.json",
                    "results/ablation2/ablation2_results.json",
                ]
                combined = {}
                for path in paths:
                    file_path = Path(path)
                    if file_path.exists():
                        combined[path] = json.loads(file_path.read_text(encoding="utf-8"))
                self.lab_results = combined
            self.lab_progress = 1.0
            self.lab_message = "Hoàn tất"
        except Exception as exc:
            self.lab_message = f"Lỗi: {exc}"
        finally:
            self.lab_running = False


def launch_desktop_app() -> None:
    ChessPygameApp().run()


if __name__ == "__main__":
    launch_desktop_app()
