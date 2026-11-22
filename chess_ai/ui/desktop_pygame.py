"""Desktop Pygame interface for chess play, analysis and replay."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import pygame
from pathlib import Path
import threading
import time
from typing import Any, Callable, Optional

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

    BTN_COLS = 2
    BTN_W = 165
    BTN_H = 34
    BTN_GAP_X = 10
    BTN_GAP_Y = 6
    BTN_START_X = BOARD_SIZE + 10
    BTN_START_Y = 10

    TIME_STEPS = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
    WORKLOAD_STEPS = [100, 200, 400, 800, 1200, 1600, 2500, 4000, 6000]

    def __init__(
        self,
        player_color: str = "white",
        agent_id: str = "level3",
        white_agent_id: str = "level2",
        mode: str = "human_vs_ai",
        time_limit: Optional[float] = 1.0,
        max_workload: Optional[int] = None,
        piece_set: str = "cburnett",
    ) -> None:
        self.player_color = player_color
        self.agent_id = agent_id
        self.white_agent_id = white_agent_id
        self.mode = mode  # "human_vs_ai" hoặc "ai_vs_ai"
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
        mode_map = {
            "human_vs_ai": "human_ai",
            "ai_vs_ai": "ai_ai",
            "human_vs_human": "human_human",
        }
        self.game_mode = mode_map.get(mode, mode)
        self.white_clock = _Clock(600.0)
        self.black_clock = _Clock(600.0)
        self.clock_running = True
        self.last_clock_tick = time.perf_counter()
        self.fullscreen = False
        self.modal_open = False
        self.modal_mode = self.game_mode
        self.modal_human_color = "white"
        if self.game_mode == "human_human":
            self.modal_white_side = "human"
            self.modal_black_side = "human"
        elif self.game_mode == "ai_ai":
            self.modal_white_side = white_agent_id if white_agent_id in list_full_board_agents() else "level2"
            self.modal_black_side = agent_id if agent_id in list_full_board_agents() else "level3"
        else:
            if player_color == "white":
                self.modal_white_side = "human"
                self.modal_black_side = agent_id if agent_id in list_full_board_agents() else "level3"
            else:
                self.modal_white_side = agent_id if agent_id in list_full_board_agents() else "level3"
                self.modal_black_side = "human"
        self.modal_white_agent = self.modal_white_side if self.modal_white_side != "human" else "level2"
        self.modal_black_agent = self.modal_black_side if self.modal_black_side != "human" else "level3"
        self.modal_budget = "time" if time_limit is not None else "workload"
        self.modal_time = float(time_limit if time_limit is not None else 1.0)
        self.modal_workload = int(max_workload if max_workload is not None else 400)
        self.modal_base = 600.0
        self.modal_increment = 0.0
        self.active_tab = "play"
        self.promotion_pending: Optional[Any] = None
        self.piece_images: dict[str, Any] = {}
        self.resign_result: Optional[float] = None
        self.resign_reason: Optional[str] = None
        self.draw_offer_pending = False
        self.game_over_dismissed = False
        self.game_over_shown_at = 0.0
        self.dragging_slider: Optional[str] = None
        self.paused = False
        self.modal_dropdown_open: Optional[str] = None
        self.modal_dropdown_scroll = 0

        self.lab_view = "arena"
        self.lab_running = False
        self.lab_progress = 0.0
        self.lab_message = "Sẵn sàng"
        self.lab_scenario = "arena"
        self.lab_agents = list_full_board_agents()
        self.lab_agent_a = "level3" if "level3" in self.lab_agents else self.lab_agents[0]
        self.lab_agent_b = "level2" if "level2" in self.lab_agents else self.lab_agents[-1]
        self.lab_dropdown_open: Optional[str] = None
        self.lab_dropdown_scroll = 0
        self.lab_num_games = 20
        self.lab_budget_mode = "workload"
        self.lab_time_limit = 1.0
        self.lab_workload = 400
        self.lab_workers = 1
        self.lab_ab1_positions = 10
        self.lab_ab1_depth = 3
        self.lab_ab1_skip_minimax = True
        self.lab_ab2_openings = 5
        self.lab_ab2_Preset = "fast"
        self.lab_rl_episodes = 500
        self.lab_rl_reinforce = 200
        self.lab_results: dict[str, Any] = {}
        self.lab_logs: list[str] = [
            "[Ready] Chọn kịch bản thí nghiệm, thiết lập tham số bên trái và bấm 'Chạy thực nghiệm'.",
            "[Tip] Bấm 'Nạp kết quả có sẵn' để xem ngay dữ liệu & biểu đồ từ thư mục results/.",
        ]
        self.lab_figure_cache: dict[str, Any] = {}
        self.lab_selected_figure_idx = 0
        self.lab_figures_dir = Path("results/ui_sandbox/figures")
        self.lab_stop_event = threading.Event()
        self._load_existing_lab_results()

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
                    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                        self.dragging_slider = None
                    elif event.type == pygame.MOUSEWHEEL and self.modal_open and self.modal_dropdown_open is not None:
                        max_scroll = max(0, (1 + len(list_full_board_agents())) - 5)
                        self.modal_dropdown_scroll = max(0, min(max_scroll, self.modal_dropdown_scroll - event.y))
                    elif event.type == pygame.MOUSEWHEEL and self.active_tab == "lab" and self.lab_dropdown_open is not None:
                        max_scroll = max(0, len(self.lab_agents) - 5)
                        self.lab_dropdown_scroll = max(0, min(max_scroll, self.lab_dropdown_scroll - event.y))
                    elif event.type == pygame.MOUSEMOTION and self.dragging_slider is not None:
                        if pygame.mouse.get_pressed()[0]:
                            position = self._fit_window_position(event.pos, screen, pygame)
                            self._update_slider_from_mouse(position[0])
                        else:
                            self.dragging_slider = None
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

    def _is_game_over(self) -> bool:
        return self.env.is_terminal() or self.resign_result is not None

    def _trigger_game_over_popup(self) -> None:
        self.game_over_dismissed = False
        self.game_over_shown_at = time.perf_counter()

    def _tick_clocks(self) -> None:
        now = time.perf_counter()
        elapsed = now - self.last_clock_tick
        self.last_clock_tick = now
        if not self.modal_open and not self.paused and self.clock_running and not self._is_game_over():
            side = self._side_to_move()
            (self.white_clock if side == "white" else self.black_clock).tick(elapsed)
            if (self.white_clock if side == "white" else self.black_clock).seconds <= 0:
                self.resign_result = -1.0 if side == "white" else 1.0
                self.resign_reason = "Hết thời gian (Flag)"
                self.message = f"Hết giờ - {'Đen' if side == 'white' else 'Trắng'} thắng"
                self.clock_running = False
                self._trigger_game_over_popup()

    def _new_game(self) -> None:
        if self.modal_open:
            return
        if self.ai_future is not None and not self.ai_future.done():
            self.message = "Hãy chờ AI hoàn tất nước đang tính"
            return
        self.modal_budget = "time" if self.time_limit is not None else "workload"
        if self.time_limit is not None:
            self.modal_time = self.time_limit
        if self.max_workload is not None:
            self.modal_workload = self.max_workload
        self.modal_dropdown_open = None
        self.modal_dropdown_scroll = 0
        self.modal_open = True

    def _start_configured_game(self) -> None:
        if self.ai_future is not None and not self.ai_future.done():
            return
        w_side = self.modal_white_side
        b_side = self.modal_black_side
        valid_ids = list_full_board_agents()
        if w_side == "human" and b_side == "human":
            self.game_mode = "human_human"
            self.mode = "human_vs_human"
            self.player_color = "both"
            self.white_agent = None
            self.black_agent = None
        elif w_side == "human" and b_side != "human":
            self.game_mode = "human_ai"
            self.mode = "human_vs_ai"
            self.player_color = "white"
            ai_id = b_side if b_side in valid_ids else "level3"
            self.agent_id = ai_id
            self.black_agent = create_agent(ai_id)
            self.white_agent = None
            self.agent = self.black_agent
        elif w_side != "human" and b_side == "human":
            self.game_mode = "human_ai"
            self.mode = "human_vs_ai"
            self.player_color = "black"
            ai_id = w_side if w_side in valid_ids else "level3"
            self.agent_id = ai_id
            self.white_agent = create_agent(ai_id)
            self.black_agent = None
            self.agent = self.white_agent
        else:
            self.game_mode = "ai_ai"
            self.mode = "ai_vs_ai"
            self.player_color = "none"
            w_id = w_side if w_side in valid_ids else "level2"
            b_id = b_side if b_side in valid_ids else "level3"
            self.white_agent_id = w_id
            self.agent_id = b_id
            self.white_agent = create_agent(w_id)
            self.black_agent = create_agent(b_id)
            self.agent = self.white_agent
        self.modal_mode = self.game_mode

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
        self.resign_reason = None
        self.draw_offer_pending = False
        self.game_over_dismissed = False
        self.paused = False
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
                if self.modal_dropdown_open is not None:
                    self.modal_dropdown_open = None
                else:
                    self.modal_open = False
            return True
        if key == pygame.K_n:
            self._new_game()
        elif key == pygame.K_f:
            self.flipped = not self.flipped
        elif key == pygame.K_u:
            self._undo()
        elif key == pygame.K_m:
            self._toggle_mode()
        elif key == pygame.K_HOME:
            self._restore_index(0)
        elif key == pygame.K_END:
            self._restore_index(len(self.records))
        elif key == pygame.K_LEFT:
            self._restore_index(self.replay_index - 1)
        elif key == pygame.K_RIGHT:
            self._restore_index(self.replay_index + 1)
        elif key == pygame.K_ESCAPE:
            if self.promotion_pending is not None:
                self.promotion_pending = None
            elif self._is_game_over() and not self.game_over_dismissed:
                self.game_over_dismissed = True
            else:
                self.selected_square = None
        elif key == pygame.K_p:
            self.player_color = "black" if self.player_color == "white" else "white"
            self._new_game()
        elif key == pygame.K_a:
            self._cycle_agent()
        elif key == pygame.K_b:
            self._toggle_budget()
        elif key == pygame.K_r:
            self._resign()
        elif key == pygame.K_d:
            self._offer_draw()
        elif key == pygame.K_SPACE:
            self._toggle_pause()
        return True

    def _toggle_pause(self) -> None:
        if self._is_game_over():
            return
        self.paused = not self.paused
        self.last_clock_tick = time.perf_counter()
        self.message = "Đã tạm dừng ván đấu" if self.paused else self._status_text()

    def _toggle_flip(self) -> None:
        self.flipped = not self.flipped

    def _toggle_mode(self) -> None:
        if self.game_mode == "human_human":
            self.game_mode = "human_ai"
            self.mode = "human_vs_ai"
        elif self.game_mode == "human_ai":
            self.game_mode = "ai_ai"
            self.mode = "ai_vs_ai"
        else:
            self.game_mode = "human_human"
            self.mode = "human_vs_human"

    def _handle_click(self, position: tuple[int, int], pygame_mod: Any = None) -> None:
        x, y = position
        if y < self.TAB_HEIGHT:
            if 0 <= x < 350:
                self.active_tab = "play"
            elif 350 <= x < 700:
                self.active_tab = "lab"
            return
        if self.modal_open:
            self._handle_modal_click(x, y, pygame_mod)
            return
        if self.active_tab == "lab":
            self._handle_lab_click(x, y)
            return
        if self._is_game_over() and not self.game_over_dismissed:
            if self._handle_game_over_click(x, y):
                return
        if self.promotion_pending is not None:
            self._handle_promotion_click(x, y)
            return
        if x < self.LEFT_WIDTH:
            self._handle_left_click(x, y, pygame_mod)
        elif x < self.LEFT_WIDTH + self.ARENA_WIDTH:
            self._handle_board_click(x - self.LEFT_WIDTH, y - self.TAB_HEIGHT)
        else:
            self._handle_right_click(x - self.LEFT_WIDTH - self.ARENA_WIDTH, y - self.TAB_HEIGHT)

    def _handle_game_over_click(self, x: int, y: int) -> bool:
        board_x = self.LEFT_WIDTH + (self.ARENA_WIDTH - self.BOARD_SIZE) // 2
        board_y = self.TAB_HEIGHT + 48
        if not (board_x <= x <= board_x + self.BOARD_SIZE and board_y <= y <= board_y + self.BOARD_SIZE):
            return False
        cx = board_x + self.BOARD_SIZE // 2
        cy = board_y + self.BOARD_SIZE // 2
        card_x, card_y = cx - 210, cy - 125
        review_rect = (card_x + 28, card_y + 182, 172, 42)
        new_game_rect = (card_x + 220, card_y + 182, 172, 42)
        if review_rect[0] <= x <= review_rect[0] + review_rect[2] and review_rect[1] <= y <= review_rect[1] + review_rect[3]:
            self.game_over_dismissed = True
            return True
        if new_game_rect[0] <= x <= new_game_rect[0] + new_game_rect[2] and new_game_rect[1] <= y <= new_game_rect[1] + new_game_rect[3]:
            self.game_over_dismissed = True
            self._new_game()
            return True
        return True

    def _update_slider_from_mouse(self, mouse_x: int) -> None:
        if self.dragging_slider == "left_time":
            ratio = max(0.0, min(1.0, (mouse_x - 24) / 222.0))
            idx = round(ratio * (len(self.TIME_STEPS) - 1))
            self.time_limit = self.TIME_STEPS[idx]
            self.max_workload = None
            self.modal_budget = "time"
            self.modal_time = self.time_limit
        elif self.dragging_slider == "left_workload":
            ratio = max(0.0, min(1.0, (mouse_x - 24) / 222.0))
            idx = round(ratio * (len(self.WORKLOAD_STEPS) - 1))
            self.max_workload = self.WORKLOAD_STEPS[idx]
            self.time_limit = None
            self.modal_budget = "workload"
            self.modal_workload = self.max_workload
        elif self.dragging_slider == "modal_slider":
            mx = 340
            track_x = mx + 170
            track_w = 395
            ratio = max(0.0, min(1.0, (mouse_x - track_x) / float(track_w)))
            if self.modal_budget == "time":
                idx = round(ratio * (len(self.TIME_STEPS) - 1))
                self.modal_time = self.TIME_STEPS[idx]
            else:
                idx = round(ratio * (len(self.WORKLOAD_STEPS) - 1))
                self.modal_workload = self.WORKLOAD_STEPS[idx]
        elif self.dragging_slider == "lab_slider":
            ratio = max(0.0, min(1.0, (mouse_x - 30) / 280.0))
            if self.lab_budget_mode == "time":
                idx = round(ratio * (len(self.TIME_STEPS) - 1))
                self.lab_time_limit = self.TIME_STEPS[idx]
            else:
                idx = round(ratio * (len(self.WORKLOAD_STEPS) - 1))
                self.lab_workload = self.WORKLOAD_STEPS[idx]

    def _handle_left_click(self, x: int, y: int, pygame: Any = None) -> None:
        ids = list_full_board_agents()
        agent_index = next(
            (
                index
                for index in range(len(ids))
                if 76 + index * 27 <= y < 101 + index * 27
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
                    self.modal_black_side = ai_id
                else:
                    self.white_agent = self.agent
                    self.modal_white_side = ai_id
            elif self.game_mode == "ai_ai":
                self.black_agent = create_agent(agent_id)
                self.modal_black_side = agent_id
            self.message = f"Đã chọn {self.agent_id}"
        elif 296 <= y <= 336:
            self.dragging_slider = "left_time"
            self._update_slider_from_mouse(x)
        elif 344 <= y <= 384:
            self.dragging_slider = "left_workload"
            self._update_slider_from_mouse(x)
        elif 420 <= y <= 452:
            self.flipped = not self.flipped
        elif 458 <= y <= 490:
            self._undo()
        elif 496 <= y <= 528:
            self._resign()
        elif 534 <= y <= 566:
            self._offer_draw()
        elif 628 <= y <= 676:
            self._new_game()

    def _handle_panel_click(self, x: int, y: int) -> None:
        col = (x - self.BTN_START_X) // (self.BTN_W + self.BTN_GAP_X)
        row = (y - self.BTN_START_Y) // (self.BTN_H + self.BTN_GAP_Y)
        if row == 1 and col == 1:
            self._toggle_mode()
        else:
            self._handle_click((x, y))

    def _handle_board_click(self, x: int, y: int) -> None:
        board_offset_x = (self.ARENA_WIDTH - self.BOARD_SIZE) // 2
        if x < board_offset_x or x >= board_offset_x + self.BOARD_SIZE:
            return
        if y < 48 or y >= 48 + self.BOARD_SIZE:
            return
        board_x = x - board_offset_x
        board_y = y - 48
        if self.ai_future is not None and not self.ai_future.done():
            return
        if self._is_game_over() or not self._is_human_turn():
            return
        if self.replay_index != len(self.records):
            self._restore_index(len(self.records))
        square = self._square_at(board_x, board_y)
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
        else:
            self.promotion_pending = None

    def _handle_right_click(self, x: int, y: int) -> None:
        replay_y = 631
        if replay_y <= y <= replay_y + 34:
            index = (x - 12) // 64
            if index == 0:
                self._restore_index(0)
            elif index == 1:
                self._restore_index(self.replay_index - 1)
            elif index == 2:
                self._toggle_pause()
            elif index == 3:
                self._restore_index(self.replay_index + 1)
            elif index == 4:
                self._restore_index(len(self.records))

    def _handle_lab_click(self, x: int, y: int) -> None:
        # 1. Nếu đang mở dropdown chọn Agent A / Agent B trong Tab Lab
        if self.lab_dropdown_open is not None:
            drop_x = 18 if self.lab_dropdown_open == "a" else 174
            drop_y = 330
            visible_rows = 5
            row_h = 28
            list_h = visible_rows * row_h
            max_scroll = max(0, len(self.lab_agents) - visible_rows)
            if drop_x <= x <= drop_x + 148 and drop_y <= y <= drop_y + list_h:
                if x >= drop_x + 134 and max_scroll > 0:
                    ratio = max(0.0, min(1.0, (y - drop_y) / float(list_h)))
                    self.lab_dropdown_scroll = round(ratio * max_scroll)
                    return
                r_idx = (y - drop_y) // row_h
                idx = self.lab_dropdown_scroll + r_idx
                if 0 <= idx < len(self.lab_agents):
                    chosen = self.lab_agents[idx]
                    if self.lab_dropdown_open == "a":
                        self.lab_agent_a = chosen
                    else:
                        self.lab_agent_b = chosen
                self.lab_dropdown_open = None
                return
            self.lab_dropdown_open = None
            return

        # 2. Chọn kịch bản thí nghiệm ở cột trái (y: 86..244)
        if 18 <= x <= 322:
            scenarios = [
                ("arena", 86),
                ("ablation1", 126),
                ("ablation2", 166),
                ("rl_krk", 206),
            ]
            for key, top in scenarios:
                if top <= y <= top + 34:
                    self.lab_scenario = key
                    if self.lab_view not in ("figures", "logs"):
                        self.lab_view = key
                    return

        # 3. Các nút điều chỉnh tham số Setup theo kịch bản đang chọn (y: 275..535)
        if 18 <= x <= 322:
            if self.lab_scenario == "arena":
                # Mở dropdown Agent A / Agent B
                if 296 <= y <= 328:
                    if 18 <= x <= 164:
                        self.lab_dropdown_open = "a"
                        self.lab_dropdown_scroll = 0
                        return
                    elif 176 <= x <= 322:
                        self.lab_dropdown_open = "b"
                        self.lab_dropdown_scroll = 0
                        return
                # Chọn số ván đấu (4, 10, 20, 50, 100)
                if 356 <= y <= 384:
                    game_opts = [4, 10, 20, 50, 100]
                    col = (x - 18) // 61
                    if 0 <= col < len(game_opts):
                        self.lab_num_games = game_opts[col]
                        return
                # Chọn chế độ ngân sách (Workload vs Time)
                if 412 <= y <= 438:
                    if 18 <= x <= 164:
                        self.lab_budget_mode = "workload"
                        return
                    elif 176 <= x <= 322:
                        self.lab_budget_mode = "time"
                        return
                # Thanh trượt ngân sách trong Lab
                if 444 <= y <= 480:
                    self.dragging_slider = "lab_slider"
                    self._update_slider_from_mouse(x)
                    return
                # Chọn số luồng Workers (1, 2, 4)
                if 506 <= y <= 534:
                    w_opts = [1, 2, 4]
                    col = (x - 110) // 70
                    if 0 <= col < len(w_opts):
                        self.lab_workers = w_opts[col]
                        return

            elif self.lab_scenario == "ablation1":
                # Số thế cờ WAC (5, 10, 25, 50)
                if 300 <= y <= 330:
                    pos_opts = [5, 10, 25, 50]
                    col = (x - 18) // 76
                    if 0 <= col < len(pos_opts):
                        self.lab_ab1_positions = pos_opts[col]
                        return
                # Độ sâu cơ sở d (2, 3, 4)
                if 362 <= y <= 392:
                    d_opts = [2, 3, 4]
                    col = (x - 18) // 102
                    if 0 <= col < len(d_opts):
                        self.lab_ab1_depth = d_opts[col]
                        return
                # Bật/tắt bỏ qua Minimax thuần khi chạy tập lớn
                if 418 <= y <= 450:
                    self.lab_ab1_skip_minimax = not self.lab_ab1_skip_minimax
                    return

            elif self.lab_scenario == "ablation2":
                # Số khai cuộc (2, 5, 10, 25) -> x2 = số ván mỗi mốc N_sim
                if 300 <= y <= 330:
                    op_opts = [2, 5, 10, 25]
                    col = (x - 18) // 76
                    if 0 <= col < len(op_opts):
                        self.lab_ab2_openings = op_opts[col]
                        return
                # Tập ngân sách N_sim: Fast [50,100,200] vs Full [50..1000]
                if 362 <= y <= 394:
                    if 18 <= x <= 164:
                        self.lab_ab2_Preset = "fast"
                        return
                    elif 176 <= x <= 322:
                        self.lab_ab2_Preset = "full"
                        return
                # Số luồng Workers (1, 2, 4)
                if 424 <= y <= 454:
                    w_opts = [1, 2, 4]
                    col = (x - 110) // 70
                    if 0 <= col < len(w_opts):
                        self.lab_workers = w_opts[col]
                        return

            elif self.lab_scenario == "rl_krk":
                # Số episodes Q-Learning mỗi cờ tàn (500, 1000, 2500, 5000)
                if 300 <= y <= 330:
                    ep_opts = [500, 1000, 2500, 5000]
                    col = (x - 18) // 76
                    if 0 <= col < len(ep_opts):
                        self.lab_rl_episodes = ep_opts[col]
                        return
                # Số episodes REINFORCE KRK (100, 500, 1000, 3000)
                if 362 <= y <= 392:
                    rf_opts = [100, 500, 1000, 3000]
                    col = (x - 18) // 76
                    if 0 <= col < len(rf_opts):
                        self.lab_rl_reinforce = rf_opts[col]
                        return

        # 4. Các nút Điều khiển chạy / dừng / nạp kết quả ở đáy cột trái
        if 18 <= x <= 214 and 610 <= y <= 650:
            self._start_lab()
            return
        elif 222 <= x <= 322 and 610 <= y <= 650:
            self.lab_stop_event.set()
            self.lab_message = "Đang yêu cầu dừng..."
            return
        elif 18 <= x <= 322 and 658 <= y <= 692:
            self._load_existing_lab_results()
            self.lab_message = "Đã nạp dữ liệu từ results/"
            return

        # 5. Thanh Sub-tab chuyển đổi chế độ xem ở khung bên phải (y: 48..84)
        if 340 <= x <= 1280 and 46 <= y <= 86:
            views = ["arena", "ablation1", "ablation2", "rl_krk", "figures", "logs"]
            for index, key in enumerate(views):
                bx = 354 + index * 152
                if bx <= x <= bx + 144:
                    self.lab_view = key
                    return

        # 6. Chuyển đổi ảnh biểu đồ khi đang ở Sub-tab "figures"
        if self.lab_view == "figures" and 370 <= x <= 1250 and 104 <= y <= 138:
            fig_files = self._list_figure_files()
            for idx in range(min(5, len(fig_files))):
                bx = 376 + idx * 174
                if bx <= x <= bx + 168:
                    self.lab_selected_figure_idx = idx
                    return

    def _handle_modal_click(self, x: int, y: int, pygame: Any) -> None:
        mx, my = 340, 115
        w, h = 600, 500
        if not (mx <= x <= mx + w and my <= y <= my + h):
            self.modal_dropdown_open = None
            return
        lx, ly = x - mx, y - my
        participants = ["human"] + list_full_board_agents()
        visible_rows = 5
        row_h = 34
        list_h = visible_rows * row_h
        max_scroll = max(0, len(participants) - visible_rows)

        # Nếu đang mở danh sách trượt (Dropdown) của bên TRẮNG hoặc bên ĐEN, ưu tiên xử lý click trên danh sách
        if self.modal_dropdown_open is not None:
            drop_lx = 24 if self.modal_dropdown_open == "white" else 316
            drop_ly = 132
            if drop_lx <= lx <= drop_lx + 260 and drop_ly <= ly <= drop_ly + list_h:
                # Nếu click vào dải thanh cuộn bên phải (16px cuối)
                if lx >= drop_lx + 242 and max_scroll > 0:
                    ratio = max(0.0, min(1.0, (ly - drop_ly) / float(list_h)))
                    self.modal_dropdown_scroll = round(ratio * max_scroll)
                    return
                row_idx = (ly - drop_ly) // row_h
                item_idx = self.modal_dropdown_scroll + row_idx
                if 0 <= item_idx < len(participants):
                    chosen = participants[item_idx]
                    if self.modal_dropdown_open == "white":
                        self.modal_white_side = chosen
                        if chosen != "human":
                            self.modal_white_agent = chosen
                    else:
                        self.modal_black_side = chosen
                        if chosen != "human":
                            self.modal_black_agent = chosen
                self.modal_dropdown_open = None
                return
            # Nếu click lại vào chính hộp header đang mở thì đóng lại
            if 88 <= ly <= 128:
                if self.modal_dropdown_open == "white" and 24 <= lx <= 284:
                    self.modal_dropdown_open = None
                    return
                if self.modal_dropdown_open == "black" and 316 <= lx <= 576:
                    self.modal_dropdown_open = None
                    return
            # Click ra ngoài danh sách -> đóng dropdown
            self.modal_dropdown_open = None

        # Nút Đổi bên Trắng <-> Đen nhanh ở góc phải tiêu đề
        if 445 <= lx <= 575 and 18 <= ly <= 46:
            self.modal_white_side, self.modal_black_side = (
                self.modal_black_side,
                self.modal_white_side,
            )
            return

        # Click mở Hộp chọn trượt (Dropdown) cho bên TRẮNG hoặc bên ĐEN
        if 88 <= ly <= 128:
            if 24 <= lx <= 284:
                self.modal_dropdown_open = "white"
                curr_idx = participants.index(self.modal_white_side) if self.modal_white_side in participants else 0
                self.modal_dropdown_scroll = max(0, min(max_scroll, curr_idx - 1))
                return
            elif 316 <= lx <= 576:
                self.modal_dropdown_open = "black"
                curr_idx = participants.index(self.modal_black_side) if self.modal_black_side in participants else 0
                self.modal_dropdown_scroll = max(0, min(max_scroll, curr_idx - 1))
                return

        # Chọn đồng hồ
        if 186 <= ly <= 256:
            col = (lx - 24) // 140
            row = (ly - 186) // 36
            if 0 <= col < 4 and 0 <= row < 2:
                index = row * 4 + col
                options = [(0, 0), (60, 0), (180, 2), (300, 0), (600, 5), (900, 10), (1800, 20)]
                if 0 <= index < len(options):
                    self.modal_base, self.modal_increment = options[index]

        # Chọn chế độ ngân sách AI & Thanh trượt trong Modal
        if 298 <= ly <= 330:
            if 24 <= lx <= 284:
                self.modal_budget = "time"
            elif 304 <= lx <= 576:
                self.modal_budget = "workload"
        if 340 <= ly <= 386 and 160 <= lx <= 580:
            self.dragging_slider = "modal_slider"
            self._update_slider_from_mouse(x)

        # Nút Hủy / Bắt đầu
        if 426 <= ly <= 470:
            if 60 <= lx <= 240:
                self.modal_open = False
                self.modal_dropdown_open = None
            elif 340 <= lx <= 540:
                self.modal_dropdown_open = None
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
        fallback: Optional[Any] = None
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
        if self.env.is_terminal():
            self.clock_running = False
            self._trigger_game_over_popup()

    def _agent_for_side(self, side: str) -> Optional[Any]:
        if side == "white":
            return self.white_agent
        return self.black_agent

    def _active_ai_agent(self) -> Optional[Any]:
        if self._is_game_over() or self.paused or self.replay_index != len(self.records):
            return None
        side = self._side_to_move()
        if self.game_mode == "human_ai" and side == self.player_color:
            return None
        if self.game_mode == "human_human":
            return None
        return self._agent_for_side(side)

    def _start_ai(self) -> None:
        if self._is_game_over() or self.paused or self.replay_index != len(self.records):
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
            reason = self.resign_reason or "đầu hàng/ngoài cuộc"
            return f"{winner} ({reason})"
        if self.env.is_terminal():
            result = self.env.result()
            winner = "Trắng thắng" if result == 1.0 else "Đen thắng" if result == -1.0 else "Hòa"
            reason = self.env.termination_reason() or "Kết thúc ván"
            return f"{winner} ({reason})"
        return "Lượt Trắng" if self._side_to_move() == "white" else "Lượt Đen"

    def _resign(self) -> None:
        if self._is_game_over():
            return
        side = self._side_to_move()
        if self.ai_future is not None and not self.ai_future.done():
            self.ai_future.cancel()
        self.resign_result = -1.0 if side == "white" else 1.0
        self.resign_reason = "Đen đầu hàng" if side == "black" else "Trắng đầu hàng"
        self.message = "Trắng thắng do Đen đầu hàng" if side == "black" else "Đen thắng do Trắng đầu hàng"
        self.clock_running = False
        self._trigger_game_over_popup()

    def _offer_draw(self) -> None:
        if self.game_mode == "ai_ai" or self._is_game_over():
            return
        if self.game_mode == "human_human":
            if self.draw_offer_pending:
                self.resign_result = 0.0
                self.resign_reason = "Hai bên thỏa thuận hòa"
                self.draw_offer_pending = False
                self.message = "Hai bên đồng ý hòa"
                self.clock_running = False
                self._trigger_game_over_popup()
            else:
                self.draw_offer_pending = True
                self.message = "Đề nghị hòa: bấm [D] lần nữa để đồng ý"
            return
        agent = self.agent
        stats = getattr(agent, "last_search_stats", {}) or {}
        eval_score = float(stats.get("eval", 0.0))
        if eval_score <= 20.0:
            self.resign_result = 0.0
            self.resign_reason = "AI chấp nhận thỏa thuận hòa"
            self.message = "AI đồng ý hòa"
            self.clock_running = False
            self._trigger_game_over_popup()
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
            y = 76 + index * 27
            item = pygame.Rect(12, y, 246, 25)
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
                        (20, y + 5))
            screen.blit(small.render(label, True, self.TEXT), (54, y + 5))
            if badge:
                badge_rect = pygame.Rect(216, y + 4, 34, 17)
                pygame.draw.rect(screen, self.PANEL_3, badge_rect, border_radius=4)
                screen.blit(mono.render(badge, True, self.MUTED),
                            badge_rect.move(7, 1))

        screen.blit(font.render("BUDGET & SEARCH", True, self.MUTED), (16, 272))
        is_time_mode = self.time_limit is not None

        curr_time = self.time_limit if self.time_limit is not None else self.modal_time
        self._draw_slider_control(
            screen, pygame, small, mono,
            x=12, y=296, width=246,
            title="Thời gian / nước",
            value_text=f"{curr_time:.1f}s",
            steps=self.TIME_STEPS,
            current_val=curr_time,
            active=is_time_mode,
        )

        curr_work = self.max_workload if self.max_workload is not None else self.modal_workload
        self._draw_slider_control(
            screen, pygame, small, mono,
            x=12, y=344, width=246,
            title="Số Nodes / Sims",
            value_text=f"{curr_work}",
            steps=self.WORKLOAD_STEPS,
            current_val=curr_work,
            active=not is_time_mode,
        )

        screen.blit(font.render("CONTROLS", True, self.MUTED), (16, 396))
        buttons = [
            ("Đảo bàn", "F", 420),
            ("Undo", "U", 458),
            ("Đầu hàng", "R", 496),
            ("Cầu hòa", "D", 534),
        ]
        for label, key, y in buttons:
            rect = pygame.Rect(12, y, 246, 32)
            visible = label != "Cầu hòa" or self.game_mode != "ai_ai"
            if visible:
                pygame.draw.rect(screen, self.SURFACE, rect, border_radius=7)
                pygame.draw.rect(screen, self.BORDER, rect, 1, border_radius=7)
                screen.blit(small.render(label, True, self.TEXT), (20, y + 7))
                key_rect = pygame.Rect(208, y + 5, 42, 22)
                pygame.draw.rect(screen, self.PANEL_3, key_rect, border_radius=5)
                key_text = mono.render(key, True, self.MUTED)
                screen.blit(key_text, key_text.get_rect(center=key_rect.center))
        new_rect = pygame.Rect(12, 628, 246, 48)
        pygame.draw.rect(screen, self.ACCENT, new_rect, border_radius=7)
        pygame.draw.rect(screen, (132, 190, 255), new_rect, 1, border_radius=7)
        screen.blit(font.render("VÁN MỚI", True, (10, 22, 38)), (64, 641))
        key_text = mono.render("[N]", True, (24, 48, 72))
        screen.blit(key_text, (174, 645))

    def _draw_slider_control(
        self,
        screen: Any,
        pygame: Any,
        small: Any,
        mono: Any,
        x: int,
        y: int,
        width: int,
        title: str,
        value_text: str,
        steps: list[Any],
        current_val: Any,
        active: bool,
    ) -> None:
        card = pygame.Rect(x, y, width, 36)
        pygame.draw.rect(screen, self.SURFACE if active else (21, 27, 38), card, border_radius=7)
        pygame.draw.rect(screen, self.ACCENT if active else self.BORDER, card, 1, border_radius=7)
        screen.blit(small.render(title, True, self.TEXT if active else self.MUTED), (x + 10, y + 3))
        val_surf = mono.render(value_text, True, self.ACCENT if active else self.MUTED)
        screen.blit(val_surf, (x + width - 10 - val_surf.get_width(), y + 4))

        track_x = x + 12
        track_w = width - 24
        track_y = y + 26
        pygame.draw.rect(screen, (12, 16, 24), pygame.Rect(track_x, track_y - 3, track_w, 6), border_radius=3)

        best_idx = min(range(len(steps)), key=lambda i: abs(float(steps[i]) - float(current_val)))
        ratio = best_idx / float(max(1, len(steps) - 1))
        fill_w = int(track_w * ratio)
        fill_color = self.ACCENT if active else (75, 88, 108)
        if fill_w > 0:
            pygame.draw.rect(screen, fill_color, pygame.Rect(track_x, track_y - 3, fill_w, 6), border_radius=3)
        for i in range(len(steps)):
            tx = track_x + int(track_w * (i / float(max(1, len(steps) - 1))))
            pygame.draw.circle(screen, fill_color if i <= best_idx else (52, 64, 82), (tx, track_y), 3)
        knob_x = track_x + fill_w
        pygame.draw.circle(screen, (240, 246, 255) if active else self.MUTED, (knob_x, track_y), 6)
        pygame.draw.circle(screen, fill_color, (knob_x, track_y), 6, 2)

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

    def _check_squares(self) -> tuple[Optional[int], set[int]]:
        """Trả về (ô_vua_đang_bị_chiếu, tập_các_ô_quân_đang_chiếu)."""
        import chess

        fen = self.env.render("fen")
        if not isinstance(fen, str):
            return None, set()
        board = chess.Board(fen)
        if not board.is_check():
            return None, set()
        king_sq = board.king(board.turn)
        checker_sqs = set(board.checkers())
        return king_sq, checker_sqs

    def _draw_board(self, screen: Any, pygame: Any, board_x: int, board_y: int) -> None:
        tile = self.BOARD_SIZE // 8
        board_rect = pygame.Rect(board_x - 3, board_y - 3, self.BOARD_SIZE + 6, self.BOARD_SIZE + 6)
        pygame.draw.rect(screen, (9, 12, 18), board_rect, border_radius=4)
        pygame.draw.rect(screen, (82, 94, 112), board_rect, 2, border_radius=4)
        pieces = self._pieces()
        checked_king_sq, checker_squares = self._check_squares()
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
            if square == checked_king_sq:
                king_overlay = pygame.Surface((tile, tile), pygame.SRCALPHA)
                king_overlay.fill((225, 38, 56, 185))
                pygame.draw.circle(king_overlay, (255, 75, 92, 235), (tile // 2, tile // 2), tile // 2 - 4)
                screen.blit(king_overlay, rect.topleft)
                pygame.draw.rect(screen, (255, 80, 95), rect, 3)
            elif square in checker_squares:
                atk_overlay = pygame.Surface((tile, tile), pygame.SRCALPHA)
                atk_overlay.fill((245, 130, 32, 175))
                screen.blit(atk_overlay, rect.topleft)
                pygame.draw.rect(screen, (255, 165, 50), rect, 3)
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

        if self._is_game_over() and not self.game_over_dismissed:
            self._draw_game_over_banner(screen, pygame, board_x, board_y)

    def _draw_game_over_banner(self, screen: Any, pygame: Any, board_x: int, board_y: int) -> None:
        # Hiệu ứng làm mờ bàn cờ + vòng sáng + thẻ thông báo kết quả
        elapsed = max(0.0, time.perf_counter() - self.game_over_shown_at)
        anim = min(1.0, elapsed / 0.22) if self.game_over_shown_at > 0 else 1.0

        dim = pygame.Surface((self.BOARD_SIZE, self.BOARD_SIZE), pygame.SRCALPHA)
        dim.fill((8, 12, 20, int(175 * anim)))
        screen.blit(dim, (board_x, board_y))

        result = self.resign_result if self.resign_result is not None else self.env.result()
        if result == 1.0:
            title = "CHIẾN THẮNG CHO BÊN TRẮNG!"
            score_str = "1 - 0"
            accent = self.GREEN
            badge_bg = (22, 78, 52)
            icon_glyph = "♔"
        elif result == -1.0:
            title = "CHIẾN THẮNG CHO BÊN ĐEN!"
            score_str = "0 - 1"
            accent = (86, 166, 255)
            badge_bg = (24, 58, 104)
            icon_glyph = "♚"
        else:
            title = "VÁN ĐẤU KẾT THÚC HÒA!"
            score_str = "½ - ½"
            accent = self.ORANGE
            badge_bg = (92, 64, 18)
            icon_glyph = "⚖"

        reason = (
            self.resign_reason
            if self.resign_result is not None
            else (self.env.termination_reason() or "Kết thúc ván cờ")
        )
        reason_map = {
            "checkmate": "Chiếu hết (Checkmate)",
            "stalemate": "Hòa Pat (Stalemate - Hết nước đi hợp lệ)",
            "insufficient_material": "Hòa do không đủ lực lượng chiếu hết",
            "seventyfive_moves": "Hòa theo luật 75 nước đi",
            "fivefold_repetition": "Hòa do lặp lại thế cờ 5 lần",
            "fifty_moves": "Hòa theo luật 50 nước đi",
            "threefold_repetition": "Hòa do lặp lại thế cờ 3 lần",
        }
        reason_display = reason_map.get(str(reason), str(reason))

        cx = board_x + self.BOARD_SIZE // 2
        cy = board_y + self.BOARD_SIZE // 2
        card_w, card_h = 420, 250
        card_rect = pygame.Rect(cx - card_w // 2, cy - card_h // 2, card_w, card_h)

        # Hiệu ứng viền phát sáng xung quanh hộp thoại
        glow_surface = pygame.Surface((card_w + 24, card_h + 24), pygame.SRCALPHA)
        pygame.draw.rect(
            glow_surface,
            (*accent, int(55 * anim)),
            pygame.Rect(0, 0, card_w + 24, card_h + 24),
            border_radius=18,
        )
        screen.blit(glow_surface, (card_rect.x - 12, card_rect.y - 12))

        pygame.draw.rect(screen, (20, 27, 39), card_rect, border_radius=14)
        pygame.draw.rect(screen, accent, card_rect, 2, border_radius=14)

        # Huy hiệu tỷ số ở đỉnh
        badge_rect = pygame.Rect(cx - 64, card_rect.y + 16, 128, 38)
        pygame.draw.rect(screen, badge_bg, badge_rect, border_radius=19)
        pygame.draw.rect(screen, accent, badge_rect, 1, border_radius=19)

        score_font = pygame.font.SysFont("consolas", 20, bold=True)
        title_font = pygame.font.SysFont("segoeui", 20, bold=True)
        sub_font = pygame.font.SysFont("segoeui", 14)
        btn_font = pygame.font.SysFont("segoeui", 14, bold=True)

        score_surf = score_font.render(f"{icon_glyph} {score_str}", True, self.TEXT)
        screen.blit(score_surf, score_surf.get_rect(center=badge_rect.center))

        title_surf = title_font.render(title, True, accent)
        screen.blit(title_surf, title_surf.get_rect(center=(cx, card_rect.y + 78)))

        reason_surf = sub_font.render(f"Lý do: {reason_display}", True, self.TEXT)
        screen.blit(reason_surf, reason_surf.get_rect(center=(cx, card_rect.y + 112)))

        moves_count = (len(self.records) + 1) // 2
        stats_surf = sub_font.render(
            f"Tổng số nước đi: {moves_count} nước ({len(self.records)} plies)",
            True,
            self.MUTED,
        )
        screen.blit(stats_surf, stats_surf.get_rect(center=(cx, card_rect.y + 140)))

        # 2 Nút hành động: Xem lại bàn cờ | Ván mới
        review_btn = pygame.Rect(card_rect.x + 28, card_rect.y + 182, 172, 42)
        new_btn = pygame.Rect(card_rect.x + 220, card_rect.y + 182, 172, 42)

        pygame.draw.rect(screen, self.PANEL_2, review_btn, border_radius=8)
        pygame.draw.rect(screen, self.BORDER_STRONG, review_btn, 1, border_radius=8)
        rev_text = btn_font.render("Xem bàn cờ", True, self.TEXT)
        screen.blit(rev_text, rev_text.get_rect(center=review_btn.center))

        pygame.draw.rect(screen, self.ACCENT, new_btn, border_radius=8)
        new_text = btn_font.render("Ván mới [N]", True, (10, 22, 38))
        screen.blit(new_text, new_text.get_rect(center=new_btn.center))

    def _draw_telemetry(self, screen: Any, pygame: Any) -> None:
        rect = pygame.Rect(self.LEFT_WIDTH + self.ARENA_WIDTH, self.TAB_HEIGHT, self.RIGHT_WIDTH, 716)
        pygame.draw.rect(screen, (20, 25, 36), rect)
        pygame.draw.line(screen, self.BORDER, (940, 44), (940, 760), 1)
        font = pygame.font.SysFont("segoeui", 14, bold=True)
        small = pygame.font.SysFont("segoeui", 12)
        mono = pygame.font.SysFont("consolas", 12)

        screen.blit(font.render("EVALUATION & WIN PROBABILITY", True, self.MUTED), (960, 58))
        active_agent = self.ai_agent or self.agent
        stats = getattr(active_agent, "last_search_stats", {}) or {}
        raw_eval = float(stats.get("eval", 0.0))
        agent_name = getattr(active_agent, "name", "")
        is_unit_scale = any(k in agent_name for k in ("MCTS", "PUCT", "Supervised", "AlphaZero"))

        # Chuẩn hóa về đơn vị Pawn (1 Tốt = 1.00) và Centipawn để vẽ thanh Eval & tính % Thắng
        if abs(raw_eval) >= 19000.0:
            mate_plies = max(1, int(round(20000.0 - abs(raw_eval))))
            mate_moves = (mate_plies + 1) // 2
            eval_display_str = f"+M{mate_moves}" if raw_eval > 0 else f"-M{mate_moves}"
            unit_label = "MATE IN N"
            cp_equiv = 1000.0 if raw_eval > 0 else -1000.0
            win_probability = 99.9 if raw_eval > 0 else 0.1
        elif is_unit_scale:
            # MCTS / Neural Network trả về Q-value trong [-1.0, +1.0]
            q_val = max(-1.0, min(1.0, raw_eval))
            eval_display_str = f"{q_val:+.2f}"
            unit_label = "Q-VALUE [-1, +1]"
            cp_equiv = q_val * 600.0
            win_probability = 50.0 + 50.0 * q_val
        else:
            # Heuristic / Alpha-Beta trả về Centipawns (100 cp = 1.00 Pawn) -> chia 100 sang đơn vị Pawn
            pawns = raw_eval / 100.0
            eval_display_str = f"{pawns:+.2f}"
            unit_label = "PAWNS (CP / 100)"
            cp_equiv = max(-1000.0, min(1000.0, raw_eval))
            win_probability = 50.0 + 50.0 * (raw_eval / (abs(raw_eval) + 250.0))

        summary_card = pygame.Rect(952, 82, 316, 132)
        pygame.draw.rect(screen, self.SURFACE, summary_card, border_radius=9)
        pygame.draw.rect(screen, self.BORDER, summary_card, 1, border_radius=9)
        pygame.draw.rect(screen, (12, 17, 25), pygame.Rect(968, 94, 18, 106), border_radius=5)
        fill = int(106 * ((cp_equiv + 1000.0) / 2000.0))
        fill = max(4, min(106, fill))
        bar_y = 200 - fill
        bar_color = self.GREEN if raw_eval >= 0 else (248, 113, 113)
        pygame.draw.rect(screen, bar_color, pygame.Rect(968, bar_y, 18, fill), border_radius=5)
        eval_font = pygame.font.SysFont("consolas", 20, bold=True)
        screen.blit(eval_font.render(eval_display_str, True, self.TEXT), (998, 104))
        screen.blit(small.render("EVALUATION", True, self.MUTED), (998, 132))
        screen.blit(pygame.font.SysFont("consolas", 10).render(unit_label, True, self.ACCENT), (998, 152))
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
        pause_label = "▶" if self.paused else "||"
        labels = ["|<", "<", pause_label, ">", ">|"]
        for index, label in enumerate(labels):
            rect = pygame.Rect(952 + index * 64, replay_y, 58, 34)
            is_pause_btn = index == 2
            bg_color = (
                (34, 67, 112)
                if (is_pause_btn and self.paused)
                else self.SURFACE
            )
            border_color = (
                self.ORANGE
                if (is_pause_btn and self.paused)
                else (self.ACCENT if is_pause_btn and self.game_mode == "ai_ai" else self.BORDER)
            )
            pygame.draw.rect(screen, bg_color, rect, border_radius=7)
            pygame.draw.rect(screen, border_color, rect, 2 if (is_pause_btn and self.paused) else 1, border_radius=7)
            text_color = self.ORANGE if (is_pause_btn and self.paused) else self.TEXT
            label_surface = mono.render(label, True, text_color)
            screen.blit(label_surface, label_surface.get_rect(center=rect.center))

    def _draw_modal(self, screen: Any, pygame: Any) -> None:
        overlay = pygame.Surface(self.WINDOW_SIZE, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 175))
        screen.blit(overlay, (0, 0))
        mx, my, w, h = 340, 115, 600, 500
        rect = pygame.Rect(mx, my, w, h)
        pygame.draw.rect(screen, self.PANEL, rect, border_radius=12)
        pygame.draw.rect(screen, self.BORDER_STRONG, rect, 2, border_radius=12)
        font = pygame.font.SysFont("segoeui", 20, bold=True)
        sec_font = pygame.font.SysFont("segoeui", 14, bold=True)
        small = pygame.font.SysFont("segoeui", 13)
        mono = pygame.font.SysFont("consolas", 12, bold=True)

        screen.blit(font.render("THIẾT LẬP VÁN ĐẤU MỚI", True, self.TEXT), (mx + 24, my + 18))

        # Nút Đổi bên Trắng <-> Đen nhanh
        swap_btn = pygame.Rect(mx + 445, my + 18, 130, 28)
        pygame.draw.rect(screen, self.PANEL_2, swap_btn, border_radius=6)
        pygame.draw.rect(screen, self.BORDER, swap_btn, 1, border_radius=6)
        swap_lbl = small.render("⇄ Đổi Trắng/Đen", True, self.ACCENT)
        screen.blit(swap_lbl, swap_lbl.get_rect(center=swap_btn.center))

        participants = ["human"] + list_full_board_agents()
        short_names = {
            "human": "👤 Người chơi",
            "level1": "L1 · Random",
            "level2": "L2 · Heuristic",
            "level3": "L3 · Alpha-Beta",
            "level4": "L4 · MCTS UCT",
            "level5": "L5 · ResNet",
            "level7": "L7 · PUCT",
            "level8": "L8 · AlphaZero",
        }

        # 1. Chọn hai bên Trắng & Đen bằng Hộp thoại trượt (Scrollable Dropdown)
        screen.blit(sec_font.render("1. Chọn Bên TRẮNG (Đi trước):", True, self.MUTED), (mx + 24, my + 62))
        screen.blit(sec_font.render("2. Chọn Bên ĐEN (Đi sau):", True, self.MUTED), (mx + 316, my + 62))

        for side_key, box_x, current_pid in (
            ("white", mx + 24, self.modal_white_side),
            ("black", mx + 316, self.modal_black_side),
        ):
            is_open = self.modal_dropdown_open == side_key
            box_rect = pygame.Rect(box_x, my + 88, 260, 40)
            pygame.draw.rect(screen, (34, 67, 112) if is_open else self.PANEL_2, box_rect, border_radius=7)
            pygame.draw.rect(
                screen,
                self.ACCENT if is_open else self.BORDER_STRONG,
                box_rect,
                2 if is_open else 1,
                border_radius=7,
            )
            label_str = short_names.get(current_pid, current_pid)
            screen.blit(sec_font.render(label_str, True, self.TEXT), (box_x + 14, my + 98))
            arrow = "▲" if is_open else "▼"
            screen.blit(small.render(arrow, True, self.ACCENT if is_open else self.MUTED), (box_x + 236, my + 100))

        # 3. Đồng hồ thi đấu
        screen.blit(sec_font.render("3. Đồng hồ ván đấu (Phút + Giây cộng thêm):", True, self.MUTED), (mx + 24, my + 160))
        options = [(0, 0), (60, 0), (180, 2), (300, 0), (600, 5), (900, 10), (1800, 20)]
        for index, (base, inc) in enumerate(options):
            col, row = index % 4, index // 4
            button = pygame.Rect(mx + 24 + col * 140, my + 186 + row * 36, 132, 30)
            selected = self.modal_base == base and self.modal_increment == inc
            pygame.draw.rect(screen, (34, 67, 112) if selected else self.PANEL_2, button, border_radius=6)
            pygame.draw.rect(screen, self.ACCENT if selected else self.BORDER, button, 2 if selected else 1, border_radius=6)
            label = "Không giới hạn (∞)" if base == 0 else f"{base // 60}p + {inc}s"
            lbl_surf = small.render(label, True, self.TEXT if selected else self.MUTED)
            screen.blit(lbl_surf, lbl_surf.get_rect(center=button.center))

        # 4. Ngân sách AI & Thanh trượt đa mức độ
        screen.blit(sec_font.render("4. Ngân sách suy nghĩ cho Agent:", True, self.MUTED), (mx + 24, my + 272))
        time_btn = pygame.Rect(mx + 24, my + 298, 260, 32)
        work_btn = pygame.Rect(mx + 304, my + 298, 272, 32)
        is_time = self.modal_budget == "time"
        pygame.draw.rect(screen, (34, 67, 112) if is_time else self.PANEL_2, time_btn, border_radius=6)
        pygame.draw.rect(screen, self.ACCENT if is_time else self.BORDER, time_btn, 2 if is_time else 1, border_radius=6)
        pygame.draw.rect(screen, (34, 67, 112) if not is_time else self.PANEL_2, work_btn, border_radius=6)
        pygame.draw.rect(screen, self.ACCENT if not is_time else self.BORDER, work_btn, 2 if not is_time else 1, border_radius=6)
        t_lbl = small.render(f"Giới hạn Thời gian ({self.modal_time:.1f}s / nước)", True, self.TEXT if is_time else self.MUTED)
        w_lbl = small.render(f"Giới hạn Số nước duyệt ({self.modal_workload} Nodes/Sims)", True, self.TEXT if not is_time else self.MUTED)
        screen.blit(t_lbl, t_lbl.get_rect(center=time_btn.center))
        screen.blit(w_lbl, w_lbl.get_rect(center=work_btn.center))

        # Vẽ thanh trượt trong Modal
        steps = self.TIME_STEPS if is_time else self.WORKLOAD_STEPS
        curr_val = self.modal_time if is_time else self.modal_workload
        val_str = f"{self.modal_time:.1f}s" if is_time else f"{self.modal_workload}"
        slider_box = pygame.Rect(mx + 24, my + 340, 552, 46)
        pygame.draw.rect(screen, self.SURFACE, slider_box, border_radius=8)
        pygame.draw.rect(screen, self.BORDER, slider_box, 1, border_radius=8)
        screen.blit(small.render("Mức độ:" if is_time else "Số Nodes/Sims:", True, self.MUTED), (mx + 36, my + 353))
        screen.blit(mono.render(val_str, True, self.ACCENT), (mx + 118, my + 354))

        track_x = mx + 170
        track_w = 395
        track_y = my + 362
        pygame.draw.rect(screen, (12, 16, 24), pygame.Rect(track_x, track_y - 3, track_w, 6), border_radius=3)
        best_idx = min(range(len(steps)), key=lambda i: abs(float(steps[i]) - float(curr_val)))
        ratio = best_idx / float(max(1, len(steps) - 1))
        fill_w = int(track_w * ratio)
        if fill_w > 0:
            pygame.draw.rect(screen, self.ACCENT, pygame.Rect(track_x, track_y - 3, fill_w, 6), border_radius=3)
        for i, step_v in enumerate(steps):
            tx = track_x + int(track_w * (i / float(max(1, len(steps) - 1))))
            pygame.draw.circle(screen, self.ACCENT if i <= best_idx else (58, 70, 88), (tx, track_y), 4)
            step_str = f"{step_v:g}s" if is_time else (f"{step_v // 1000}k" if step_v >= 1000 and step_v % 1000 == 0 else str(step_v))
            s_lbl = pygame.font.SysFont("consolas", 10).render(step_str, True, self.ACCENT if i == best_idx else self.MUTED)
            screen.blit(s_lbl, s_lbl.get_rect(center=(tx, track_y + 14)))
        knob_x = track_x + fill_w
        pygame.draw.circle(screen, (240, 246, 255), (knob_x, track_y), 7)
        pygame.draw.circle(screen, self.ACCENT, (knob_x, track_y), 7, 2)

        cancel = pygame.Rect(mx + 60, my + 426, 180, 44)
        start = pygame.Rect(mx + 340, my + 426, 200, 44)
        pygame.draw.rect(screen, self.PANEL_2, cancel, border_radius=8)
        pygame.draw.rect(screen, self.BORDER, cancel, 1, border_radius=8)
        pygame.draw.rect(screen, self.ACCENT, start, border_radius=8)
        c_lbl = sec_font.render("Hủy [Esc]", True, self.TEXT)
        s_lbl = sec_font.render("Bắt đầu ván đấu", True, (10, 22, 38))
        screen.blit(c_lbl, c_lbl.get_rect(center=cancel.center))
        screen.blit(s_lbl, s_lbl.get_rect(center=start.center))

        # Vẽ danh sách thả xuống có thanh cuộn trượt (nằm đè lên trên cùng khi đang mở)
        if self.modal_dropdown_open is not None:
            drop_x = mx + (24 if self.modal_dropdown_open == "white" else 316)
            drop_y = my + 132
            visible_rows = 5
            row_h = 34
            list_h = visible_rows * row_h
            current_pid = self.modal_white_side if self.modal_dropdown_open == "white" else self.modal_black_side

            drop_rect = pygame.Rect(drop_x, drop_y, 260, list_h)
            pygame.draw.rect(screen, (16, 22, 32), drop_rect, border_radius=8)
            pygame.draw.rect(screen, self.ACCENT, drop_rect, 2, border_radius=8)

            visible_slice = participants[self.modal_dropdown_scroll : self.modal_dropdown_scroll + visible_rows]
            for r_idx, pid in enumerate(visible_slice):
                item_rect = pygame.Rect(drop_x + 4, drop_y + r_idx * row_h + 3, 238, row_h - 6)
                is_sel = pid == current_pid
                if is_sel:
                    pygame.draw.rect(screen, (34, 67, 112), item_rect, border_radius=6)
                lbl = small.render(short_names.get(pid, pid), True, self.ACCENT if is_sel else self.TEXT)
                screen.blit(lbl, (item_rect.x + 10, item_rect.y + 5))

            # Thanh cuộn (Scrollbar) bên phải hộp thoại trượt
            max_scroll = max(1, len(participants) - visible_rows)
            sb_track = pygame.Rect(drop_x + 246, drop_y + 6, 8, list_h - 12)
            pygame.draw.rect(screen, (10, 14, 20), sb_track, border_radius=4)
            thumb_h = max(24, int(sb_track.height * (visible_rows / float(len(participants)))))
            thumb_y = sb_track.y + int((sb_track.height - thumb_h) * (self.modal_dropdown_scroll / float(max_scroll)))
            pygame.draw.rect(screen, self.ACCENT, pygame.Rect(sb_track.x, thumb_y, 8, thumb_h), border_radius=4)

    def _append_lab_log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.lab_logs.append(f"[{stamp}] {text}")
        if len(self.lab_logs) > 250:
            self.lab_logs = self.lab_logs[-250:]

    def _list_figure_files(self) -> list[tuple[str, Path]]:
        sandbox_dir = self.lab_figures_dir
        report_dir = Path("results/figures")
        names = [
            ("Ablation 1A: Progressive", "ablation1_progressive.png"),
            ("Ablation 1B: Depth", "ablation1_depth_sensitivity.png"),
            ("Ablation 2: MCTS Curve", "ablation2_mcts_sensitivity.png"),
            ("Cấp 6 RL: Learning Curve", "rl_krk_learning_curve.png"),
            ("Arena: Matchup Chart", "arena_matchup.png"),
        ]
        resolved: list[tuple[str, Path]] = []
        for title, fname in names:
            p_sandbox = sandbox_dir / fname
            p_report = report_dir / fname
            if p_sandbox.exists():
                resolved.append((title, p_sandbox))
            elif p_report.exists():
                resolved.append((title, p_report))
        return resolved

    def _regenerate_matplotlib_figures(self) -> None:
        """Dùng thư viện Matplotlib vẽ biểu đồ từ kết quả thí nghiệm hiện tại ra results/ui_sandbox/figures/ (độc lập với results/figures/ của báo cáo)."""
        try:
            from scripts.reproduce_experiment import (
                _plot_ablation1,
                _plot_ablation2,
                _plot_arena_matchup,
                _plot_rl_krk,
            )

            out_dir = self.lab_figures_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            if "ablation1" in self.lab_results:
                _plot_ablation1(self.lab_results["ablation1"], out_dir)
            if "ablation2" in self.lab_results:
                _plot_ablation2(self.lab_results["ablation2"], out_dir)
            if "rl_krk" in self.lab_results:
                _plot_rl_krk(self.lab_results["rl_krk"], out_dir)
            if "arena" in self.lab_results:
                _plot_arena_matchup(self.lab_results["arena"], out_dir)
            self.lab_figure_cache.clear()
        except Exception as exc:
            self._append_lab_log(f"Không thể vẽ Matplotlib: {exc}")

    def _get_matplotlib_surface(self, filename: str, pygame: Any, max_w: int = 860, max_h: int = 300) -> Optional[Any]:
        """Nạp và scale mượt ảnh biểu đồ Matplotlib từ results/ui_sandbox/figures/ (hoặc fallback về results/figures/)."""
        fig_path = self.lab_figures_dir / filename
        if not fig_path.exists():
            self._regenerate_matplotlib_figures()
        if not fig_path.exists():
            fig_path = Path("results/figures") / filename
        if not fig_path.exists():
            return None
        cache_key = f"{fig_path}_{max_w}x{max_h}"
        if cache_key not in self.lab_figure_cache:
            try:
                img = pygame.image.load(str(fig_path))
                iw, ih = img.get_size()
                scale = min(max_w / float(max(1, iw)), max_h / float(max(1, ih)))
                new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
                self.lab_figure_cache[cache_key] = pygame.transform.smoothscale(img, new_size)
            except Exception:
                return None
        surf = self.lab_figure_cache.get(cache_key)
        return surf if not isinstance(surf, str) else None

    def _load_existing_lab_results(self) -> None:
        mapping = {
            "arena": Path("results/tournament_summary.json"),
            "ablation1": Path("results/ablation1/ablation1_results.json"),
            "ablation2": Path("results/ablation2/ablation2_results.json"),
            "rl_krk": Path("results/training/krk_learning_curve.json"),
        }
        loaded = []
        for key, file_path in mapping.items():
            if file_path.exists():
                try:
                    self.lab_results[key] = json.loads(file_path.read_text(encoding="utf-8"))
                    loaded.append(file_path.name)
                except Exception:
                    continue
        self._regenerate_matplotlib_figures()
        if loaded:
            self._append_lab_log(f"Đã nạp dữ liệu & vẽ biểu đồ Matplotlib: {', '.join(loaded)}")

    def _draw_lab_tab(self, screen: Any, pygame: Any) -> None:
        pygame.draw.rect(screen, (16, 21, 30), pygame.Rect(0, 44, 1280, 716))
        pygame.draw.rect(screen, (20, 25, 36), pygame.Rect(0, 44, 340, 716))
        pygame.draw.line(screen, self.BORDER, (339, 44), (339, 760), 1)
        font = pygame.font.SysFont("segoeui", 15, bold=True)
        sec_font = pygame.font.SysFont("segoeui", 13, bold=True)
        small = pygame.font.SysFont("segoeui", 12)
        mono = pygame.font.SysFont("consolas", 12)

        # --- CỘT TRÁI: 1. CHỌN KỊCH BẢN THÍ NGHIỆM ---
        screen.blit(font.render("1. CHỌN KỊCH BẢN THÍ NGHIỆM", True, self.MUTED), (18, 56))
        scenarios = [
            ("arena", "⚔  Đấu Cặp / Giải Đấu (Arena)"),
            ("ablation1", "🔍  Ablation 1: Alpha-Beta (WAC-50)"),
            ("ablation2", "🌲  Ablation 2: MCTS (Độ nhạy N_sim)"),
            ("rl_krk", "🧠  Cấp 6 RL: Cờ tàn KQK / KRK / KPK"),
        ]
        for index, (key, label) in enumerate(scenarios):
            y = 86 + index * 40
            selected = self.lab_scenario == key
            card = pygame.Rect(18, y, 304, 34)
            pygame.draw.rect(
                screen,
                (34, 67, 112) if selected else self.SURFACE,
                card,
                border_radius=7,
            )
            pygame.draw.rect(
                screen,
                self.ACCENT if selected else self.BORDER,
                card,
                2 if selected else 1,
                border_radius=7,
            )
            screen.blit(sec_font.render(label, True, self.TEXT if selected else self.MUTED), (28, y + 8))
            if selected:
                pygame.draw.rect(
                    screen,
                    self.ACCENT,
                    pygame.Rect(310, y + 8, 5, 18),
                    border_radius=2,
                )

        # --- CỘT TRÁI: 2. THIẾT LẬP THAM SỐ (SETUP PANEL) ---
        screen.blit(font.render("2. THIẾT LẬP THAM SỐ (SETUP)", True, self.MUTED), (18, 252))
        setup_card = pygame.Rect(14, 276, 312, 266)
        pygame.draw.rect(screen, (15, 20, 29), setup_card, border_radius=9)
        pygame.draw.rect(screen, self.BORDER, setup_card, 1, border_radius=9)

        short_names = {
            "level1": "L1 · Random",
            "level2": "L2 · Heuristic",
            "level3": "L3 · Alpha-Beta",
            "level4": "L4 · MCTS UCT",
            "level5": "L5 · ResNet",
            "level7": "L7 · PUCT",
            "level8": "L8 · AlphaZero",
        }

        if self.lab_scenario == "arena":
            screen.blit(small.render("Chọn Agent A:", True, self.MUTED), (20, 279))
            screen.blit(small.render("Chọn Agent B:", True, self.MUTED), (176, 279))
            for key_ab, bx, aid in (("a", 18, self.lab_agent_a), ("b", 176, self.lab_agent_b)):
                is_open = self.lab_dropdown_open == key_ab
                d_rect = pygame.Rect(bx, 296, 146, 30)
                pygame.draw.rect(screen, (34, 67, 112) if is_open else self.SURFACE, d_rect, border_radius=6)
                pygame.draw.rect(screen, self.ACCENT if is_open else self.BORDER, d_rect, 1, border_radius=6)
                screen.blit(small.render(short_names.get(aid, aid), True, self.TEXT), (bx + 8, 303))
                screen.blit(small.render("▲" if is_open else "▼", True, self.ACCENT), (bx + 128, 303))

            screen.blit(small.render("Số ván đấu (đối xứng 2 màu):", True, self.MUTED), (20, 336))
            game_opts = [4, 10, 20, 50, 100]
            for idx, g_num in enumerate(game_opts):
                btn = pygame.Rect(18 + idx * 61, 356, 57, 28)
                sel = self.lab_num_games == g_num
                pygame.draw.rect(screen, (34, 67, 112) if sel else self.SURFACE, btn, border_radius=5)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=5)
                lbl = mono.render(str(g_num), True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))

            screen.blit(small.render("Chế độ Ngân sách:", True, self.MUTED), (20, 392))
            for b_mode, b_lbl, bx in (("workload", "Workload (Nodes/Sims)", 18), ("time", "Time Limit (s)", 176)):
                sel = self.lab_budget_mode == b_mode
                btn = pygame.Rect(bx, 412, 146, 26)
                pygame.draw.rect(screen, (34, 67, 112) if sel else self.SURFACE, btn, border_radius=5)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=5)
                lbl = small.render(b_lbl, True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))

            is_time = self.lab_budget_mode == "time"
            self._draw_slider_control(
                screen, pygame, small, mono,
                x=18, y=444, width=304,
                title="Thời gian / nước" if is_time else "Số Nodes / Sims",
                value_text=f"{self.lab_time_limit:.1f}s" if is_time else f"{self.lab_workload}",
                steps=self.TIME_STEPS if is_time else self.WORKLOAD_STEPS,
                current_val=self.lab_time_limit if is_time else self.lab_workload,
                active=True,
            )

            screen.blit(small.render("Workers:", True, self.MUTED), (20, 512))
            for idx, w_num in enumerate((1, 2, 4)):
                btn = pygame.Rect(110 + idx * 70, 506, 62, 28)
                sel = self.lab_workers == w_num
                pygame.draw.rect(screen, (34, 67, 112) if sel else self.SURFACE, btn, border_radius=5)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=5)
                lbl = mono.render(f"{w_num} CPU", True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))

        elif self.lab_scenario == "ablation1":
            screen.blit(small.render("Số thế cờ chiến thuật WAC (tối đa 50):", True, self.MUTED), (20, 280))
            for idx, p_num in enumerate((5, 10, 25, 50)):
                btn = pygame.Rect(18 + idx * 76, 300, 70, 30)
                sel = self.lab_ab1_positions == p_num
                pygame.draw.rect(screen, (34, 67, 112) if sel else self.SURFACE, btn, border_radius=6)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=6)
                lbl = mono.render(f"{p_num} FEN", True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))

            screen.blit(small.render("Độ sâu cơ sở (Base Depth d):", True, self.MUTED), (20, 342))
            for idx, d_num in enumerate((2, 3, 4)):
                btn = pygame.Rect(18 + idx * 102, 362, 96, 30)
                sel = self.lab_ab1_depth == d_num
                pygame.draw.rect(screen, (34, 67, 112) if sel else self.SURFACE, btn, border_radius=6)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=6)
                lbl = mono.render(f"Depth d = {d_num}", True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))

            chk_rect = pygame.Rect(18, 418, 304, 32)
            pygame.draw.rect(
                screen,
                (34, 67, 112) if self.lab_ab1_skip_minimax else self.SURFACE,
                chk_rect,
                border_radius=6,
            )
            pygame.draw.rect(screen, self.ACCENT if self.lab_ab1_skip_minimax else self.BORDER, chk_rect, 1, border_radius=6)
            chk_mark = "☑" if self.lab_ab1_skip_minimax else "☐"
            screen.blit(small.render(f"{chk_mark} Bỏ qua Config_A (Minimax thuần) để chạy nhanh", True, self.TEXT), (26, 426))
            screen.blit(small.render("Đánh giá Config_A -> Config_D trên WAC.001..050", True, self.MUTED), (20, 470))
            screen.blit(small.render("Đo: Tactical Accuracy, Nodes Reduced %, Latency", True, self.MUTED), (20, 492))

        elif self.lab_scenario == "ablation2":
            screen.blit(small.render("Số khai cuộc mỗi mức N_sim (x2 ván):", True, self.MUTED), (20, 280))
            for idx, op_num in enumerate((2, 5, 10, 25)):
                btn = pygame.Rect(18 + idx * 76, 300, 70, 30)
                sel = self.lab_ab2_openings == op_num
                pygame.draw.rect(screen, (34, 67, 112) if sel else self.SURFACE, btn, border_radius=6)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=6)
                lbl = mono.render(f"{op_num * 2} ván", True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))

            screen.blit(small.render("Tập mốc ngân sách mô phỏng N_sim:", True, self.MUTED), (20, 342))
            for p_key, p_lbl, bx in (("fast", "Nhanh [50, 100, 200]", 18), ("full", "Đầy đủ [50..1000]", 176)):
                sel = self.lab_ab2_Preset == p_key
                btn = pygame.Rect(bx, 362, 146, 30)
                pygame.draw.rect(screen, (34, 67, 112) if sel else self.SURFACE, btn, border_radius=6)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=6)
                lbl = small.render(p_lbl, True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))

            screen.blit(small.render("Workers:", True, self.MUTED), (20, 430))
            for idx, w_num in enumerate((1, 2, 4)):
                btn = pygame.Rect(110 + idx * 70, 424, 62, 28)
                sel = self.lab_workers == w_num
                pygame.draw.rect(screen, (34, 67, 112) if sel else self.SURFACE, btn, border_radius=5)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=5)
                lbl = mono.render(f"{w_num} CPU", True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))
            screen.blit(small.render("Đo đường cong điểm S của MCTS theo N_sim", True, self.MUTED), (20, 474))

        elif self.lab_scenario == "rl_krk":
            screen.blit(small.render("Số Episodes Tabular Q-Learning / cờ tàn:", True, self.MUTED), (20, 280))
            for idx, ep_num in enumerate((500, 1000, 2500, 5000)):
                btn = pygame.Rect(18 + idx * 76, 300, 70, 30)
                sel = self.lab_rl_episodes == ep_num
                pygame.draw.rect(screen, (34, 67, 112) if sel else self.SURFACE, btn, border_radius=6)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=6)
                lbl = mono.render(str(ep_num), True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))

            screen.blit(small.render("Số Episodes Policy Gradient (REINFORCE KRK):", True, self.MUTED), (20, 342))
            for idx, rf_num in enumerate((100, 500, 1000, 3000)):
                btn = pygame.Rect(18 + idx * 76, 362, 70, 30)
                sel = self.lab_rl_reinforce == rf_num
                pygame.draw.rect(screen, (34, 67, 112) if sel else self.SURFACE, btn, border_radius=6)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=6)
                lbl = mono.render(str(rf_num), True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))

            screen.blit(small.render("Cờ tàn huấn luyện: KQK -> KRK -> KPK (Curriculum)", True, self.MUTED), (20, 416))
            screen.blit(small.render("Lưu Q-Table sandbox: results/ui_sandbox/l6_qtable.pkl", True, self.MUTED), (20, 440))
            screen.blit(small.render("Lưu Log sandbox: results/ui_sandbox/krk_curve.json", True, self.MUTED), (20, 464))

        # --- CỘT TRÁI: 3. TIẾN ĐỘ & NÚT ĐIỀU KHIỂN ---
        progress_width = int(304 * max(0.0, min(1.0, self.lab_progress)))
        pygame.draw.rect(screen, (10, 15, 23), pygame.Rect(18, 554, 304, 18), border_radius=9)
        pygame.draw.rect(screen, self.BORDER, pygame.Rect(18, 554, 304, 18), 1, border_radius=9)
        if progress_width > 0:
            pygame.draw.rect(
                screen,
                self.ORANGE if self.lab_running else self.GREEN,
                pygame.Rect(18, 554, progress_width, 18),
                border_radius=9,
            )
        status_line = f"{int(self.lab_progress * 100):3d}%  {self.lab_message}"
        screen.blit(mono.render(status_line[:38], True, self.TEXT), (18, 578))

        run_btn = pygame.Rect(18, 610, 196, 40)
        stop_btn = pygame.Rect(222, 610, 100, 40)
        load_btn = pygame.Rect(18, 658, 304, 34)

        pygame.draw.rect(screen, (52, 92, 140) if self.lab_running else self.ACCENT, run_btn, border_radius=7)
        pygame.draw.rect(screen, (180, 74, 74), stop_btn, border_radius=7)
        pygame.draw.rect(screen, self.SURFACE, load_btn, border_radius=7)
        pygame.draw.rect(screen, self.BORDER_STRONG, load_btn, 1, border_radius=7)

        r_txt = sec_font.render("Đang chạy..." if self.lab_running else "▶ Chạy thực nghiệm", True, (10, 22, 38))
        s_txt = sec_font.render("■ Dừng", True, self.TEXT)
        l_txt = small.render("📂 Nạp lại kết quả có sẵn từ results/", True, self.ACCENT)
        screen.blit(r_txt, r_txt.get_rect(center=run_btn.center))
        screen.blit(s_txt, s_txt.get_rect(center=stop_btn.center))
        screen.blit(l_txt, l_txt.get_rect(center=load_btn.center))

        # --- Vẽ Dropdown chọn Agent A / B nếu đang mở ---
        if self.lab_dropdown_open is not None and self.lab_scenario == "arena":
            drop_x = 18 if self.lab_dropdown_open == "a" else 176
            drop_y = 328
            visible_rows = 5
            row_h = 28
            list_h = visible_rows * row_h
            curr_aid = self.lab_agent_a if self.lab_dropdown_open == "a" else self.lab_agent_b
            d_box = pygame.Rect(drop_x, drop_y, 146, list_h)
            pygame.draw.rect(screen, (14, 19, 28), d_box, border_radius=7)
            pygame.draw.rect(screen, self.ACCENT, d_box, 2, border_radius=7)
            vis_agents = self.lab_agents[self.lab_dropdown_scroll : self.lab_dropdown_scroll + visible_rows]
            for r_i, aid in enumerate(vis_agents):
                row_rect = pygame.Rect(drop_x + 3, drop_y + r_i * row_h + 2, 130, row_h - 4)
                if aid == curr_aid:
                    pygame.draw.rect(screen, (34, 67, 112), row_rect, border_radius=5)
                screen.blit(
                    small.render(short_names.get(aid, aid), True, self.ACCENT if aid == curr_aid else self.TEXT),
                    (row_rect.x + 6, row_rect.y + 4),
                )
            max_sc = max(1, len(self.lab_agents) - visible_rows)
            sb = pygame.Rect(drop_x + 136, drop_y + 4, 6, list_h - 8)
            pygame.draw.rect(screen, (10, 14, 20), sb, border_radius=3)
            th_h = max(18, int(sb.height * (visible_rows / float(len(self.lab_agents)))))
            th_y = sb.y + int((sb.height - th_h) * (self.lab_dropdown_scroll / float(max_sc)))
            pygame.draw.rect(screen, self.ACCENT, pygame.Rect(sb.x, th_y, 6, th_h), border_radius=3)

        # --- KHUNG PHẢI: THANH SUB-TABS & BẢNG KẾT QUẢ / BIỂU ĐỒ ---
        pygame.draw.rect(screen, (20, 25, 36), pygame.Rect(340, 44, 940, 44))
        view_tabs = [
            ("arena", "⚔ Arena"),
            ("ablation1", "🔍 Ablation 1"),
            ("ablation2", "🌲 Ablation 2"),
            ("rl_krk", "🧠 Cấp 6 RL"),
            ("figures", "📈 Biểu đồ PNG"),
            ("logs", "📝 Nhật ký"),
        ]
        for index, (key, title_v) in enumerate(view_tabs):
            rect = pygame.Rect(354 + index * 152, 50, 144, 32)
            selected = self.lab_view == key
            pygame.draw.rect(
                screen,
                (34, 67, 112) if selected else self.SURFACE,
                rect,
                border_radius=7,
            )
            pygame.draw.rect(
                screen,
                self.ACCENT if selected else self.BORDER,
                rect,
                2 if selected else 1,
                border_radius=7,
            )
            lbl_v = small.render(title_v, True, self.TEXT if selected else self.MUTED)
            screen.blit(lbl_v, lbl_v.get_rect(center=rect.center))

        self._draw_lab_right_content(screen, pygame, font, sec_font, small, mono)

    def _draw_lab_right_content(
        self, screen: Any, pygame: Any, font: Any, sec_font: Any, small: Any, mono: Any
    ) -> None:
        panel_rect = pygame.Rect(356, 98, 906, 646)
        pygame.draw.rect(screen, (12, 17, 25), panel_rect, border_radius=10)
        pygame.draw.rect(screen, self.BORDER, panel_rect, 1, border_radius=10)

        if self.lab_view == "arena":
            screen.blit(font.render("KẾT QUẢ ĐẤU CẶP ĐỐI XỨNG (ARENA PAIRED MATCHUP)", True, self.ACCENT), (376, 114))
            data = self.lab_results.get("arena")
            if isinstance(data, dict) and "agent_a" in data:
                a_name = data.get("agent_a", "Agent A")
                b_name = data.get("agent_b", "Agent B")
                games = int(data.get("games", 0))
                wins = int(data.get("wins", 0))
                draws = int(data.get("draws", 0))
                losses = int(data.get("losses", 0))
                score = float(data.get("score", 0.0))
                w_ci = data.get("wilson_ci_95", [0.0, 0.0])
                b_ci = data.get("bootstrap_ci_95", [0.0, 0.0])

                screen.blit(sec_font.render(f"Cặp đấu:  {a_name}   vs   {b_name}   ({games} ván đối xứng 2 màu)", True, self.TEXT), (376, 150))

                # Thanh tỷ lệ Thắng / Hòa / Thua trực quan
                bar_rect = pygame.Rect(376, 184, 860, 32)
                pygame.draw.rect(screen, self.SURFACE, bar_rect, border_radius=8)
                if games > 0:
                    w_w = int(860 * (wins / games))
                    d_w = int(860 * (draws / games))
                    l_w = 860 - w_w - d_w
                    if w_w > 0:
                        pygame.draw.rect(screen, self.GREEN, pygame.Rect(376, 184, w_w, 32), border_radius=6)
                    if d_w > 0:
                        pygame.draw.rect(screen, self.ORANGE, pygame.Rect(376 + w_w, 184, d_w, 32))
                    if l_w > 0:
                        pygame.draw.rect(screen, (239, 68, 68), pygame.Rect(376 + w_w + d_w, 184, l_w, 32), border_radius=6)

                # Các thẻ chỉ số thống kê rút gọn ở trên + Biểu đồ Matplotlib ở dưới
                cards = [
                    ("Thắng / Hòa / Thua (A)", f"+{wins}  = {draws}  -{losses}  ({games} ván)"),
                    ("Điểm chuẩn hóa (S)", f"{score:.3f}  (W: {data.get('white_score', 0):.2f} | B: {data.get('black_score', 0):.2f})"),
                    ("Wilson / Bootstrap 95% CI", f"W:[{w_ci[0]:.3f}, {w_ci[1]:.3f}]  B:[{b_ci[0]:.3f}, {b_ci[1]:.3f}]"),
                    ("Độ trễ TB / p95 (A vs B)", f"A: {data.get('agent_a_avg_latency_ms', 0):.1f}ms  |  B: {data.get('agent_b_avg_latency_ms', 0):.1f}ms"),
                ]
                for idx, (k_lbl, v_lbl) in enumerate(cards):
                    col, row = idx % 2, idx // 2
                    c_rect = pygame.Rect(376 + col * 436, 224 + row * 56, 420, 48)
                    pygame.draw.rect(screen, self.SURFACE, c_rect, border_radius=8)
                    pygame.draw.rect(screen, self.BORDER, c_rect, 1, border_radius=8)
                    screen.blit(small.render(k_lbl, True, self.MUTED), (c_rect.x + 12, c_rect.y + 4))
                    screen.blit(mono.render(v_lbl, True, self.TEXT), (c_rect.x + 12, c_rect.y + 24))

                chart_surf = self._get_matplotlib_surface("arena_matchup.png", pygame, max_w=856, max_h=380)
                if chart_surf is not None:
                    screen.blit(chart_surf, chart_surf.get_rect(center=(809, 535)))
            else:
                self._draw_empty_lab_hint(screen, small, "Chưa có kết quả Arena. Hãy chọn 2 Agent bên trái rồi bấm 'Chạy thực nghiệm'.")

        elif self.lab_view == "ablation1":
            screen.blit(font.render("ABLATION 1: HIỆU QUẢ CẮT TỈA ALPHA-BETA, TT & QUIESCENCE (WAC-50)", True, self.ACCENT), (376, 114))
            data = self.lab_results.get("ablation1")
            if isinstance(data, dict) and "part_1a_progressive_configs" in data:
                configs = data.get("part_1a_progressive_configs", [])
                headers = ["Cấu hình", "Độ sâu", "Giải đúng", "Accuracy", "Wilson 95% CI", "Nodes TB", "Giảm nút %", "Trễ TB (ms)"]
                col_x = [376, 478, 540, 625, 715, 865, 985, 1105]
                pygame.draw.rect(screen, self.PANEL_2, pygame.Rect(370, 146, 876, 28), border_radius=6)
                for h_txt, cx in zip(headers, col_x):
                    screen.blit(sec_font.render(h_txt, True, self.ACCENT), (cx, 152))

                for idx, cfg in enumerate(configs):
                    ry = 178 + idx * 32
                    row_rect = pygame.Rect(370, ry, 876, 29)
                    pygame.draw.rect(screen, self.SURFACE if idx % 2 == 0 else (18, 24, 35), row_rect, border_radius=5)
                    ci = cfg.get("wilson_ci_95", [0, 0])
                    red = cfg.get("node_reduction_vs_minimax_pct", 0.0)
                    vals = [
                        str(cfg.get("name", "")),
                        f"d={cfg.get('depth', 3)}",
                        f"{cfg.get('solved_count', 0)}/{cfg.get('total_positions', 0)}",
                        f"{float(cfg.get('tactical_accuracy', 0)) * 100:.1f}%",
                        f"[{ci[0]:.2f}, {ci[1]:.2f}]",
                        f"{float(cfg.get('avg_nodes', 0)):,.0f}",
                        f"-{red:.1f}%" if red > 0 else "Baseline",
                        f"{float(cfg.get('avg_latency_ms', 0)):.1f} ms",
                    ]
                    for v_txt, cx in zip(vals, col_x):
                        screen.blit(mono.render(v_txt, True, self.TEXT), (cx, ry + 6))

                # Hiển thị trực tiếp biểu đồ Matplotlib Ablation 1A bên dưới bảng
                chart_surf = self._get_matplotlib_surface("ablation1_progressive.png", pygame, max_w=856, max_h=385)
                if chart_surf is not None:
                    screen.blit(chart_surf, chart_surf.get_rect(center=(809, 540)))
            else:
                self._draw_empty_lab_hint(screen, small, "Chưa có dữ liệu Ablation 1. Hãy bấm 'Chạy thực nghiệm' hoặc 'Nạp lại kết quả'.")

        elif self.lab_view == "ablation2":
            screen.blit(font.render("ABLATION 2: ĐỘ NHẠY NGÂN SÁCH MÔ PHỎNG MCTS (N_sim)", True, self.ACCENT), (376, 114))
            data = self.lab_results.get("ablation2")
            if isinstance(data, dict) and "part_2a_uct_level4" in data:
                curve = data.get("part_2a_uct_level4", [])
                headers = ["N_sim", "Số ván", "Thắng / Hòa / Thua", "Điểm S", "Wilson 95% CI", "Bootstrap 95% CI", "Độ trễ TB (ms)"]
                col_x = [376, 465, 555, 725, 835, 995, 1140]
                pygame.draw.rect(screen, self.PANEL_2, pygame.Rect(370, 146, 876, 28), border_radius=6)
                for h_txt, cx in zip(headers, col_x):
                    screen.blit(sec_font.render(h_txt, True, self.ACCENT), (cx, 152))

                for idx, row in enumerate(curve):
                    ry = 178 + idx * 30
                    row_rect = pygame.Rect(370, ry, 876, 27)
                    pygame.draw.rect(screen, self.SURFACE if idx % 2 == 0 else (18, 24, 35), row_rect, border_radius=5)
                    w_ci = row.get("wilson_ci_95", [0, 0])
                    b_ci = row.get("bootstrap_ci_95", [0, 0])
                    vals = [
                        str(row.get("simulations", "")),
                        str(row.get("games", 0)),
                        f"+{row.get('wins', 0)} / ={row.get('draws', 0)} / -{row.get('losses', 0)}",
                        f"{float(row.get('score', 0)):.3f}",
                        f"[{w_ci[0]:.3f}, {w_ci[1]:.3f}]",
                        f"[{b_ci[0]:.3f}, {b_ci[1]:.3f}]",
                        f"{float(row.get('agent_a_avg_latency_ms', 0)):.1f} ms",
                    ]
                    for v_txt, cx in zip(vals, col_x):
                        screen.blit(mono.render(v_txt, True, self.TEXT), (cx, ry + 5))

                # Hiển thị trực tiếp biểu đồ Matplotlib Ablation 2 bên dưới bảng
                chart_surf = self._get_matplotlib_surface("ablation2_mcts_sensitivity.png", pygame, max_w=856, max_h=370)
                if chart_surf is not None:
                    screen.blit(chart_surf, chart_surf.get_rect(center=(809, 550)))
            else:
                self._draw_empty_lab_hint(screen, small, "Chưa có dữ liệu Ablation 2. Hãy bấm 'Chạy thực nghiệm' hoặc 'Nạp lại kết quả'.")

        elif self.lab_view == "rl_krk":
            screen.blit(font.render("CẤP 6 RL: HỌC TĂNG CƯỜNG CỜ TÀN KQK / KRK / KPK & REINFORCE", True, self.ACCENT), (376, 114))
            data = self.lab_results.get("rl_krk")
            if isinstance(data, dict) and "stage_summary" in data:
                summary = data.get("stage_summary", {})
                headers = ["Giai đoạn / Cờ tàn", "Số trạng thái Q", "Win Rate (Train)", "Win Rate (Eval)", "Số plies TB", "Thời gian chạy"]
                col_x = [376, 560, 710, 860, 1005, 1130]
                pygame.draw.rect(screen, self.PANEL_2, pygame.Rect(370, 146, 876, 28), border_radius=6)
                for h_txt, cx in zip(headers, col_x):
                    screen.blit(sec_font.render(h_txt, True, self.ACCENT), (cx, 152))

                for idx, (stage_key, info) in enumerate(summary.items()):
                    ry = 178 + idx * 32
                    row_rect = pygame.Rect(370, ry, 876, 29)
                    pygame.draw.rect(screen, self.SURFACE if idx % 2 == 0 else (18, 24, 35), row_rect, border_radius=5)
                    vals = [
                        stage_key,
                        f"{int(info.get('q_states', 0)):,}" if "q_states" in info else "Policy Net",
                        f"{float(info.get('final_train_win_rate', 0)) * 100:.1f}%",
                        f"{float(info.get('greedy_win_rate', 0)) * 100:.1f}%" if "greedy_win_rate" in info else "-",
                        f"{float(info.get('greedy_avg_plies', info.get('final_train_avg_plies', 0))):.1f}",
                        f"{float(info.get('elapsed_sec', 0)):.1f}s",
                    ]
                    for v_txt, cx in zip(vals, col_x):
                        screen.blit(mono.render(v_txt, True, self.TEXT), (cx, ry + 6))

                # Hiển thị trực tiếp biểu đồ Matplotlib đường cong học tập RL bên dưới bảng
                chart_surf = self._get_matplotlib_surface("rl_krk_learning_curve.png", pygame, max_w=856, max_h=410)
                if chart_surf is not None:
                    screen.blit(chart_surf, chart_surf.get_rect(center=(809, 526)))
            else:
                self._draw_empty_lab_hint(screen, small, "Chưa có dữ liệu Cấp 6 RL. Hãy bấm 'Chạy thực nghiệm' hoặc 'Nạp lại kết quả'.")

        elif self.lab_view == "figures":
            fig_files = self._list_figure_files()
            if fig_files:
                self.lab_selected_figure_idx = min(self.lab_selected_figure_idx, len(fig_files) - 1)
                for idx, (f_title, _) in enumerate(fig_files[:5]):
                    btn = pygame.Rect(376 + idx * 174, 108, 168, 30)
                    sel = idx == self.lab_selected_figure_idx
                    pygame.draw.rect(screen, (34, 67, 112) if sel else self.SURFACE, btn, border_radius=6)
                    pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=6)
                    lbl = small.render(f_title, True, self.TEXT if sel else self.MUTED)
                    screen.blit(lbl, lbl.get_rect(center=btn.center))

                _, f_path = fig_files[self.lab_selected_figure_idx]
                cached = self._get_matplotlib_surface(f_path.name, pygame, max_w=870, max_h=575)
                if cached is not None:
                    img_rect = cached.get_rect(center=(809, 440))
                    pygame.draw.rect(screen, self.BORDER_STRONG, img_rect.inflate(6, 6), 1, border_radius=8)
                    screen.blit(cached, img_rect)
            else:
                self._draw_empty_lab_hint(screen, small, "Không tìm thấy tệp biểu đồ PNG trong results/figures/.")

        elif self.lab_view == "logs":
            screen.blit(font.render("NHẬT KÝ TIẾN TRÌNH THỰC NGHIỆM (EXPERIMENT LOGS)", True, self.ACCENT), (376, 114))
            y = 152
            for line in self.lab_logs[-24:]:
                color = self.GREEN if "Hoàn tất" in line else (self.ORANGE if "Lỗi" in line else self.TEXT)
                screen.blit(mono.render(line[:108], True, color), (376, y))
                y += 23

    def _draw_empty_lab_hint(self, screen: Any, small: Any, hint_text: str) -> None:
        empty = small.render("Chưa có dữ liệu cho mục này", True, self.TEXT)
        screen.blit(empty, empty.get_rect(center=(809, 380)))
        hint = small.render(hint_text, True, self.MUTED)
        screen.blit(hint, hint.get_rect(center=(809, 408)))

    def _start_lab(self) -> None:
        if self.lab_running:
            return
        self.lab_running = True
        self.lab_stop_event.clear()
        self.lab_progress = 0.05
        self.lab_view = self.lab_scenario
        thread = threading.Thread(target=self._run_lab_worker, daemon=True)
        thread.start()

    def _run_lab_worker(self) -> None:
        sandbox_root = Path("results/ui_sandbox")
        sandbox_root.mkdir(parents=True, exist_ok=True)
        try:
            if self.lab_scenario == "ablation1":
                from chess_ai.experiments.ablation_search import run_ablation_search

                self.lab_message = f"Đang chạy WAC ({self.lab_ab1_positions} thế cờ, d={self.lab_ab1_depth})..."
                self._append_lab_log(
                    f"[UI Sandbox] Bắt đầu Ablation 1: max_positions={self.lab_ab1_positions}, base_depth={self.lab_ab1_depth}, skip_minimax={self.lab_ab1_skip_minimax}"
                )
                self.lab_progress = 0.25
                res = run_ablation_search(
                    output_path=str(sandbox_root / "ablation1_results.json"),
                    max_positions=self.lab_ab1_positions,
                    base_depth=self.lab_ab1_depth,
                    skip_minimax_if_large=self.lab_ab1_skip_minimax,
                )
                self.lab_results["ablation1"] = res
                self._append_lab_log("Hoàn tất Ablation 1 (Lưu tại results/ui_sandbox/ablation1_results.json).")

            elif self.lab_scenario == "ablation2":
                from chess_ai.experiments.ablation_mcts import run_ablation_mcts

                budgets = [50, 100, 200] if self.lab_ab2_Preset == "fast" else [50, 100, 200, 400, 800, 1000]
                self.lab_message = f"Đang chạy MCTS N_sim {budgets}..."
                self._append_lab_log(
                    f"[UI Sandbox] Bắt đầu Ablation 2: openings={self.lab_ab2_openings}, budgets={budgets}, workers={self.lab_workers}"
                )
                self.lab_progress = 0.2
                res = run_ablation_mcts(
                    output_path=str(sandbox_root / "ablation2_results.json"),
                    sim_budgets=budgets,
                    num_openings_2a=self.lab_ab2_openings,
                    num_openings_2b=self.lab_ab2_openings,
                    workers=self.lab_workers,
                )
                self.lab_results["ablation2"] = res
                self._append_lab_log("Hoàn tất Ablation 2 (Lưu tại results/ui_sandbox/ablation2_results.json).")

            elif self.lab_scenario == "arena":
                from chess_ai.arena.tournament import run_paired_matchup
                from chess_ai.utils.epd import load_epd_file

                agent_a = create_agent(self.lab_agent_a)
                agent_b = create_agent(self.lab_agent_b)
                num_openings = max(1, self.lab_num_games // 2)
                openings = load_epd_file("resources/openings_50.epd")[:num_openings]
                b_mode = "time_control" if self.lab_budget_mode == "time" else "workload_control"
                self.lab_message = f"Đang đấu {self.lab_agent_a} vs {self.lab_agent_b} ({num_openings * 2} ván)..."
                self._append_lab_log(
                    f"[UI Sandbox] Bắt đầu Arena: {self.lab_agent_a} vs {self.lab_agent_b} | {num_openings * 2} ván | mode={b_mode} | workers={self.lab_workers}"
                )
                self.lab_progress = 0.25
                res = run_paired_matchup(
                    agent_a,
                    agent_b,
                    openings,
                    seeds=[42],
                    budget_mode=b_mode,
                    time_limit_sec=self.lab_time_limit if self.lab_budget_mode == "time" else None,
                    workload_a=self.lab_workload if self.lab_budget_mode == "workload" else None,
                    workload_b=self.lab_workload if self.lab_budget_mode == "workload" else None,
                    jsonl_path=str(sandbox_root / "matches.jsonl"),
                    pgn_path=str(sandbox_root / "games.pgn"),
                    workers=self.lab_workers,
                )
                (sandbox_root / "arena_matchup.json").write_text(
                    json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                self.lab_results["arena"] = res
                self._append_lab_log(
                    f"Hoàn tất Arena: {res.get('agent_a')} đạt S={res.get('score'):.3f} (+{res.get('wins')}/={res.get('draws')}/-{res.get('losses')})"
                )

            elif self.lab_scenario == "rl_krk":
                from chess_ai.training.train_rl_krk import train_krk_rl

                self.lab_message = f"Đang huấn luyện RL ({self.lab_rl_episodes} eps)..."
                self._append_lab_log(
                    f"[UI Sandbox] Bắt đầu huấn luyện Cấp 6 RL: Q-episodes={self.lab_rl_episodes}, REINFORCE={self.lab_rl_reinforce}"
                )
                self.lab_progress = 0.2
                res = train_krk_rl(
                    output_qtable=str(sandbox_root / "l6_krk_qtable_sandbox.pkl"),
                    output_metrics=str(sandbox_root / "krk_learning_curve.json"),
                    episodes_override=self.lab_rl_episodes,
                    reinforce_episodes=self.lab_rl_reinforce,
                    verbose=False,
                )
                self.lab_results["rl_krk"] = res
                self._append_lab_log("Hoàn tất huấn luyện Cấp 6 RL (Lưu tại results/ui_sandbox/).")

            self._regenerate_matplotlib_figures()
            self.lab_progress = 1.0
            self.lab_message = "Hoàn tất thực nghiệm!"
        except Exception as exc:
            self.lab_message = f"Lỗi: {exc}"
            self._append_lab_log(f"Lỗi thực nghiệm: {exc}")
        finally:
            self.lab_running = False


def launch_desktop_app() -> None:
    ChessPygameApp().run()


if __name__ == "__main__":
    launch_desktop_app()
