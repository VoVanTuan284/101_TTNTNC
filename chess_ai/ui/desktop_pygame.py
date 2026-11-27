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
    TAB_BAR = (11, 15, 23)
    TAB_INACTIVE = (14, 19, 28)
    SIDEBAR = (20, 25, 36)
    LAB_BG = (16, 21, 30)
    CARD_DARK = (15, 21, 31)
    CLOCK_BG = (10, 15, 23)
    SELECTED_BG = (34, 67, 112)
    ACCENT_TEXT = (10, 22, 38)
    ROW_ALT = (18, 24, 35)
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

    UI_THEMES = {
        "dark": {
            "BG": (15, 19, 28),
            "TAB_BAR": (11, 15, 23),
            "TAB_INACTIVE": (14, 19, 28),
            "SIDEBAR": (20, 25, 36),
            "LAB_BG": (16, 21, 30),
            "CARD_DARK": (15, 21, 31),
            "CLOCK_BG": (10, 15, 23),
            "SELECTED_BG": (34, 67, 112),
            "ACCENT_TEXT": (10, 22, 38),
            "ROW_ALT": (18, 24, 35),
            "PANEL": (24, 30, 42),
            "PANEL_2": (33, 42, 56),
            "PANEL_3": (42, 54, 72),
            "BORDER": (60, 72, 92),
            "BORDER_STRONG": (76, 92, 116),
            "SURFACE": (28, 35, 48),
            "SURFACE_HOVER": (38, 48, 64),
            "TEXT": (235, 239, 244),
            "MUTED": (165, 174, 188),
            "ACCENT": (86, 166, 255),
            "GREEN": (74, 222, 128),
            "ORANGE": (245, 158, 11),
        },
        "light": {
            "BG": (243, 245, 249),
            "TAB_BAR": (232, 236, 243),
            "TAB_INACTIVE": (225, 230, 238),
            "SIDEBAR": (237, 240, 246),
            "LAB_BG": (243, 245, 249),
            "CARD_DARK": (232, 236, 244),
            "CLOCK_BG": (224, 229, 238),
            "SELECTED_BG": (219, 233, 252),
            "ACCENT_TEXT": (255, 255, 255),
            "ROW_ALT": (245, 247, 251),
            "PANEL": (252, 253, 255),
            "PANEL_2": (235, 239, 246),
            "PANEL_3": (222, 228, 238),
            "BORDER": (190, 200, 216),
            "BORDER_STRONG": (152, 166, 188),
            "SURFACE": (255, 255, 255),
            "SURFACE_HOVER": (240, 245, 253),
            "TEXT": (18, 24, 36),
            "MUTED": (68, 79, 98),
            "ACCENT": (45, 108, 210),
            "GREEN": (21, 128, 61),
            "ORANGE": (180, 83, 9),
        },
    }

    ACCENT_THEMES = [
        ("blue", "Xanh Dương", (86, 166, 255), (45, 108, 210)),
        ("teal", "Ngọc Lam", (45, 212, 191), (13, 148, 136)),
        ("purple", "Tím", (167, 139, 250), (109, 40, 217)),
        ("emerald", "Xanh Lá", (74, 222, 128), (21, 128, 61)),
        ("amber", "Cam Ấm", (251, 191, 36), (180, 83, 9)),
        ("rose", "San Hô", (251, 113, 133), (190, 18, 60)),
    ]

    BOARD_THEMES = [
        ("classic_green", "Xanh Cổ Điển", (238, 238, 210), (118, 150, 86)),
        ("wood", "Gỗ Ấm", (240, 217, 181), (181, 136, 99)),
        ("slate_blue", "Xanh Xám", (222, 227, 230), (140, 162, 173)),
        ("purple", "Tím Hiện Đại", (235, 230, 245), (132, 118, 186)),
        ("emerald", "Ngọc Bích", (234, 242, 234), (82, 138, 108)),
        ("walnut", "Óc Chó", (226, 209, 179), (138, 103, 74)),
    ]

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
        self.lab_tab_locked = False
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
        self.lab_live_games: list[dict[str, Any]] = []
        self.lab_games_scroll = 0
        self.lab_live_lock = threading.Lock()
        self._mini_piece_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._mini_board_cache: dict[tuple[Any, ...], Any] = {}
        self._icon_cache: dict[tuple[str, int, tuple[int, int, int]], Any] = {}
        self.ui_theme = "dark"
        self.accent_theme = "blue"
        self.board_theme = "classic_green"
        self.settings_open = False
        self.settings_tab = "appearance"
        self.settings_piece_scroll = 0
        default_bg = self._default_background_path()
        self.bg_image_path: Optional[str] = str(default_bg) if default_bg is not None else None
        self.bg_image_surface: Optional[Any] = None
        self.bg_image_scroll = 0
        self.music_path: Optional[str] = None
        self.music_scroll = 0
        self.sfx_volume = 0.8
        self.music_volume = 0.15
        self.mouse_pos: tuple[int, int] = (-1, -1)
        self.sound_enabled = True
        self.sound_pack = "standard"
        self._sound_cache: dict[tuple[str, str], Any] = {}
        self._mixer_ready: Optional[bool] = None
        self._piece_preview_cache: dict[str, Any] = {}
        self.available_piece_sets = self._discover_piece_sets()
        self._apply_theme("dark")
        self._apply_board_theme("classic_green")
        self._load_existing_lab_results()

    def run(self) -> None:
        import pygame

        pygame.init()
        screen = pygame.display.set_mode(self.WINDOW_SIZE, pygame.RESIZABLE)
        pygame.display.set_caption("Chess AI - Play & Telemetry")
        canvas = pygame.Surface(self.WINDOW_SIZE)
        clock = pygame.time.Clock()
        self._load_piece_images(pygame)
        self._preload_sounds()
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
                    elif event.type == pygame.MOUSEWHEEL and self.settings_open:
                        if self.settings_tab == "board":
                            total_rows = (len(self.available_piece_sets) + 1) // 2
                            max_scroll = max(0, total_rows - 4)
                            self.settings_piece_scroll = max(0, min(max_scroll, self.settings_piece_scroll - event.y))
                        elif self.settings_tab == "appearance":
                            max_scroll = max(0, len(self._list_background_images()) - 4)
                            self.bg_image_scroll = max(0, min(max_scroll, self.bg_image_scroll - event.y))
                        elif self.settings_tab == "audio":
                            max_scroll = max(0, len(self._list_music_files()) - 4)
                            self.music_scroll = max(0, min(max_scroll, self.music_scroll - event.y))
                    elif event.type == pygame.MOUSEWHEEL and self.modal_open and self.modal_dropdown_open is not None:
                        max_scroll = max(0, (1 + len(list_full_board_agents())) - 5)
                        self.modal_dropdown_scroll = max(0, min(max_scroll, self.modal_dropdown_scroll - event.y))
                    elif event.type == pygame.MOUSEWHEEL and self.active_tab == "lab" and self.lab_dropdown_open is not None:
                        max_scroll = max(0, len(self.lab_agents) - 5)
                        self.lab_dropdown_scroll = max(0, min(max_scroll, self.lab_dropdown_scroll - event.y))
                    elif event.type == pygame.MOUSEWHEEL and self.active_tab == "lab" and self.lab_view == "arena" and self.lab_live_games:
                        total_rows = (len(self.lab_live_games) + 2) // 3
                        max_scroll = max(0, total_rows - 2)
                        self.lab_games_scroll = max(0, min(max_scroll, self.lab_games_scroll - event.y))
                    elif event.type == pygame.MOUSEMOTION:
                        self.mouse_pos = self._fit_window_position(event.pos, screen, pygame)
                        if self.dragging_slider is not None:
                            if pygame.mouse.get_pressed()[0]:
                                self._update_slider_from_mouse(self.mouse_pos[0])
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
                if screen.get_width() > 4 and screen.get_height() > 4:
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
        self.piece_images.clear()
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
        has_display = pygame.display.get_surface() is not None
        for symbol, filename in names.items():
            for suffix in (".png", ".svg"):
                path = piece_dir / f"{filename}{suffix}"
                if path.exists():
                    try:
                        image = pygame.image.load(str(path))
                        if has_display:
                            image = image.convert_alpha()
                        self.piece_images[symbol] = pygame.transform.smoothscale(
                            image, (60, 60)
                        )
                        break
                    except Exception:
                        continue

    _PRELOAD_SOUNDS = (
        "Move",
        "Capture",
        "Check",
        "OutOfBound",
        "GenericNotify",
        "Victory",
        "Defeat",
        "Draw",
    )

    def _ensure_mixer(self) -> bool:
        try:
            if not pygame.get_init():
                return False
            if self._mixer_ready is None:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                self._mixer_ready = bool(pygame.mixer.get_init())
        except Exception:
            self._mixer_ready = False
        return bool(self._mixer_ready)

    def _load_sound(self, name: str) -> Any:
        key = (self.sound_pack, name)
        if key in self._sound_cache:
            return self._sound_cache[key]

        sound_root = Path(__file__).resolve().parent / "assets" / "sound"
        candidates: list[Path] = [
            sound_root / self.sound_pack / f"{name}.mp3",
        ]
        if name == "OutOfBound":
            candidates.append(sound_root / self.sound_pack / "Error.mp3")
        # Fallback sang bộ piano nếu file ở bộ standard là symlink chưa checkout hoặc lỗi định dạng
        for fallback_pack in ("piano", "standard"):
            if fallback_pack != self.sound_pack:
                candidates.append(sound_root / fallback_pack / f"{name}.mp3")

        snd = None
        for path in candidates:
            if not path.exists():
                continue
            try:
                if path.stat().st_size < 256:
                    target = path.read_text(encoding="utf-8", errors="ignore").strip()
                    resolved = (path.parent / target).resolve()
                    if resolved.exists() and resolved.is_file() and resolved.stat().st_size >= 256 and "Silence" not in resolved.name:
                        path = resolved
                    else:
                        continue
                snd = pygame.mixer.Sound(str(path))
                break
            except Exception:
                continue

        self._sound_cache[key] = snd
        return snd

    def _preload_sounds(self) -> None:
        """Tải trước toàn bộ 9 file âm thanh vào RAM ngay khi khởi động."""
        if not self._ensure_mixer():
            return
        for name in self._PRELOAD_SOUNDS:
            self._load_sound(name)

    def _play_sound(self, name: str) -> None:
        if not self.sound_enabled or not self._ensure_mixer():
            return
        snd = self._load_sound(name)
        if snd is not None:
            try:
                snd.set_volume(self.sfx_volume)
                snd.play()
            except Exception:
                pass

    def _play_move_sound(self, san: str, is_terminal: bool, result: Optional[float]) -> None:
        """Phát đúng 1 âm thanh theo thứ tự ưu tiên: Victory/Defeat/Draw -> Check -> Capture -> Move."""
        if is_terminal:
            self._play_game_over_sound(result)
        elif "+" in san or "#" in san:
            self._play_sound("Check")
        elif "x" in san:
            self._play_sound("Capture")
        else:
            self._play_sound("Move")

    def _play_game_over_sound(self, result: Optional[float]) -> None:
        if result is None or result == 0.0:
            self._play_sound("Draw")
            return
        if self.game_mode == "human_ai":
            human_won = (
                (result == 1.0 and self.player_color == "white")
                or (result == -1.0 and self.player_color == "black")
            )
            self._play_sound("Victory" if human_won else "Defeat")
        else:
            self._play_sound("Victory")

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
            clk = self.white_clock if side == "white" else self.black_clock
            clk.tick(elapsed)
            if clk.seconds <= 0:
                self.resign_result = -1.0 if side == "white" else 1.0
                self.resign_reason = "Hết thời gian (Flag)"
                self.message = f"Hết giờ - {'Đen' if side == 'white' else 'Trắng'} thắng"
                self.clock_running = False
                self._trigger_game_over_popup()
                self._play_game_over_sound(self.resign_result)

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
        self._play_sound("GenericNotify")

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

        sw = max(1, min(4096, int(screen.get_width())))
        sh = max(1, min(4096, int(screen.get_height())))
        if (sw, sh) == tuple(self.WINDOW_SIZE):
            return canvas
        scale = min(sw / self.WINDOW_SIZE[0], sh / self.WINDOW_SIZE[1])
        size = (
            max(1, int(self.WINDOW_SIZE[0] * scale)),
            max(1, int(self.WINDOW_SIZE[1] * scale)),
        )
        try:
            return pygame.transform.smoothscale(canvas, size)
        except Exception:
            return pygame.transform.scale(canvas, size)

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
        if self.settings_open:
            if key == pygame.K_ESCAPE:
                self.settings_open = False
            return True
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
            if 1104 <= x <= 1138 and 6 <= y <= 38:
                self.sound_enabled = not self.sound_enabled
                try:
                    if pygame.get_init() and pygame.mixer.get_init():
                        pygame.mixer.music.set_volume(self.music_volume if self.sound_enabled else 0.0)
                except Exception:
                    pass
                return
            if 1146 <= x <= 1268 and 6 <= y <= 38:
                self.settings_open = not self.settings_open
                if self.settings_open:
                    self.modal_open = False
                    self.modal_dropdown_open = None
                    self.lab_dropdown_open = None
                return
            if 12 <= x <= 230 and 6 <= y <= 38:
                self.settings_open = False
                self.active_tab = "play"
            elif 240 <= x <= 480 and 6 <= y <= 38:
                self.settings_open = False
                if not self.lab_tab_locked:
                    self.active_tab = "lab"
                else:
                    self.message = "Tab Nghiên cứu đang tạm khóa"
            return
        if self.settings_open:
            self._handle_settings_click(x, y, pygame_mod)
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
        elif self.dragging_slider == "settings_sfx":
            mx = 300
            track_x = mx + 220
            track_w = 414
            ratio = max(0.0, min(1.0, (mouse_x - track_x) / float(track_w)))
            self._set_sfx_volume(ratio)
        elif self.dragging_slider == "settings_music":
            mx = 300
            track_x = mx + 220
            track_w = 414
            ratio = max(0.0, min(1.0, (mouse_x - track_x) / float(track_w)))
            self._set_music_volume(ratio * 0.24)
        elif self.dragging_slider == "lab_slider":
            ratio = max(0.0, min(1.0, (mouse_x - 30) / 280.0))
            if self.lab_budget_mode == "time":
                idx = round(ratio * (len(self.TIME_STEPS) - 1))
                self.lab_time_limit = self.TIME_STEPS[idx]
            else:
                idx = round(ratio * (len(self.WORKLOAD_STEPS) - 1))
                self.lab_workload = self.WORKLOAD_STEPS[idx]

    def _handle_left_click(self, x: int, y: int, pygame: Any = None) -> None:
        if 84 <= y <= 124:
            self.dragging_slider = "left_time"
            self._update_slider_from_mouse(x)
        elif 132 <= y <= 172:
            self.dragging_slider = "left_workload"
            self._update_slider_from_mouse(x)
        elif 216 <= y <= 248:
            self.flipped = not self.flipped
        elif 254 <= y <= 286:
            self._undo()
        elif 292 <= y <= 324:
            self._resign()
        elif 330 <= y <= 362:
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
            if square != self.selected_square and not self._legal_from(square):
                self._play_sound("OutOfBound")
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

        # Thanh cuộn danh sách bàn cờ trực tiếp ở Tab Arena
        if self.lab_view == "arena" and self.lab_live_games and 1228 <= x <= 1252 and 332 <= y <= 732:
            total_rows = (len(self.lab_live_games) + 2) // 3
            max_scroll = max(0, total_rows - 2)
            if max_scroll > 0:
                ratio = max(0.0, min(1.0, (y - 332) / 400.0))
                self.lab_games_scroll = round(ratio * max_scroll)
            return

        # 2. Kịch bản thí nghiệm mặc định ở cột trái (Đấu Cặp / Giải Đấu Arena)
        if 18 <= x <= 322 and 86 <= y <= 120:
            self.lab_scenario = "arena"
            self.lab_view = "arena"
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
            views = ["figures", "logs"]
            for index, key in enumerate(views):
                bx = 354 + index * 190
                if bx <= x <= bx + 180:
                    self.lab_view = key
                    return

        # 6. Chuyển đổi ảnh biểu đồ khi đang ở Sub-tab "figures" (lưới 2 hàng × 4 cột)
        if self.lab_view == "figures" and 370 <= x <= 1250 and 104 <= y <= 170:
            fig_files = self._list_figure_files()
            for idx in range(min(8, len(fig_files))):
                row, col = divmod(idx, 4)
                bx = 376 + col * 218
                by = 106 + row * 32
                if bx <= x <= bx + 210 and by <= y <= by + 28:
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
        self._play_move_sound(san, self.env.is_terminal(), self.env.result() if self.env.is_terminal() else None)

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
        self._play_game_over_sound(self.resign_result)

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
                self._play_sound("Draw")
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
            self._play_sound("Draw")
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

    def _get_icon(
        self,
        pygame: Any,
        name: str,
        size: int,
        color: tuple[int, int, int],
    ) -> Any:
        key = (name, size, color)
        if key in self._icon_cache:
            return self._icon_cache[key]
        path = Path(__file__).resolve().parent / "assets" / "icon" / f"{name}.png"
        if not path.exists():
            self._icon_cache[key] = None
            return None
        try:
            raw = pygame.image.load(str(path))
            if pygame.display.get_surface() is not None:
                raw = raw.convert_alpha()
            scaled = pygame.transform.smoothscale(raw, (size, size))
            tinted = scaled.copy()
            tinted.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MIN)
            tinted.fill((color[0], color[1], color[2], 0), special_flags=pygame.BLEND_RGBA_ADD)
            self._icon_cache[key] = tinted
            return tinted
        except Exception:
            self._icon_cache[key] = None
            return None

    def _blit_icon_text(
        self,
        screen: Any,
        pygame: Any,
        icon_name: str,
        icon_size: int,
        color: tuple[int, int, int],
        text_surf: Any,
        pos: Optional[tuple[int, int]] = None,
        center_rect: Any = None,
        gap: int = 7,
    ) -> None:
        icon = self._get_icon(pygame, icon_name, icon_size, color)
        if center_rect is not None:
            if icon is None:
                screen.blit(text_surf, text_surf.get_rect(center=center_rect.center))
                return
            total_w = icon.get_width() + gap + text_surf.get_width()
            start_x = center_rect.centerx - total_w // 2
            icon_y = center_rect.centery - icon.get_height() // 2
            text_y = center_rect.centery - text_surf.get_height() // 2
            screen.blit(icon, (start_x, icon_y))
            screen.blit(text_surf, (start_x + icon.get_width() + gap, text_y))
            return
        if pos is None:
            return
        x, y = pos
        if icon is None:
            screen.blit(text_surf, (x, y))
            return
        row_h = max(icon.get_height(), text_surf.get_height())
        icon_y = y + (row_h - icon.get_height()) // 2
        text_y = y + (row_h - text_surf.get_height()) // 2
        screen.blit(icon, (x, icon_y))
        screen.blit(text_surf, (x + icon.get_width() + gap, text_y))

    @staticmethod
    def _discover_piece_sets() -> list[str]:
        piece_root = Path(__file__).resolve().parent / "assets" / "piece"
        if not piece_root.exists():
            return ["cburnett"]
        found = sorted(p.name for p in piece_root.iterdir() if p.is_dir())
        preferred = ["cburnett", "merida", "alpha", "cardinal", "california", "staunty", "maestro", "fresca"]
        ordered = [name for name in preferred if name in found]
        ordered.extend(name for name in found if name not in ordered)
        return ordered or ["cburnett"]

    def _apply_theme(self, theme_name: str) -> None:
        palette = self.UI_THEMES.get(theme_name, self.UI_THEMES["dark"])
        self.ui_theme = theme_name if theme_name in self.UI_THEMES else "dark"
        for attr, val in palette.items():
            setattr(self, attr, val)
        self._apply_accent_theme(self.accent_theme)

    def _apply_accent_theme(self, accent_id: str) -> None:
        for aid, _, dark_c, light_c in self.ACCENT_THEMES:
            if aid == accent_id:
                self.accent_theme = aid
                self.ACCENT = light_c if self.ui_theme == "light" else dark_c
                return

    @staticmethod
    def _default_background_path() -> Optional[Path]:
        p = Path(__file__).resolve().parent / "assets" / "image" / "background" / "Nitro_Wallpaper_05_3840x2400.jpg"
        return p if p.exists() else None

    @staticmethod
    def _list_background_images() -> list[Path]:
        bg_dir = Path(__file__).resolve().parent / "assets" / "image" / "background"
        if not bg_dir.exists():
            return []
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        return sorted(p for p in bg_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)

    @staticmethod
    def _list_music_files() -> list[Path]:
        music_dir = Path(__file__).resolve().parent / "assets" / "music"
        if not music_dir.exists():
            return []
        exts = {".mp3", ".wav", ".ogg", ".flac"}
        return sorted(p for p in music_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)

    def _set_background_image(self, path: Optional[Path], pygame_mod: Any = None) -> None:
        if path is None:
            self.bg_image_path = None
            self.bg_image_surface = None
            return
        pg = pygame_mod or pygame
        try:
            img = pg.image.load(str(path))
            if pg.display.get_surface() is not None:
                img = img.convert()
            iw, ih = img.get_size()
            tw, th = self.WINDOW_SIZE
            scale = max(tw / float(max(1, iw)), th / float(max(1, ih)))
            sw, sh = max(1, int(iw * scale)), max(1, int(ih * scale))
            scaled = pg.transform.smoothscale(img, (sw, sh))
            surface = pg.Surface(self.WINDOW_SIZE)
            surface.blit(scaled, ((tw - sw) // 2, (th - sh) // 2))
            self.bg_image_path = str(path)
            self.bg_image_surface = surface
        except Exception:
            pass

    def _set_background_music(self, path: Optional[Path]) -> None:
        if path is None:
            self.music_path = None
            try:
                if pygame.get_init() and pygame.mixer.get_init():
                    pygame.mixer.music.stop()
            except Exception:
                pass
            return
        if not self._ensure_mixer():
            self.music_path = str(path)
            return
        try:
            pygame.mixer.music.load(str(path))
            pygame.mixer.music.set_volume(self.music_volume if self.sound_enabled else 0.0)
            pygame.mixer.music.play(-1)
            self.music_path = str(path)
        except Exception:
            pass

    def _set_sfx_volume(self, value: float) -> None:
        self.sfx_volume = max(0.0, min(1.0, float(value)))
        for snd in self._sound_cache.values():
            if snd is not None:
                try:
                    snd.set_volume(self.sfx_volume)
                except Exception:
                    pass

    def _set_music_volume(self, value: float) -> None:
        self.music_volume = max(0.0, min(0.24, float(value)))
        try:
            if pygame.get_init() and pygame.mixer.get_init():
                pygame.mixer.music.set_volume(self.music_volume if self.sound_enabled else 0.0)
        except Exception:
            pass

    @staticmethod
    def _browse_local_file(title: str, filetypes: list[tuple[str, str]]) -> Optional[Path]:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askopenfilename(title=title, filetypes=filetypes)
            root.destroy()
            if selected:
                return Path(selected)
        except Exception:
            pass
        return None

    def _draw_background_layer(self, screen: Any, pygame: Any, area_rect: Any) -> None:
        if self.bg_image_surface is None and self.bg_image_path is not None:
            self._set_background_image(Path(self.bg_image_path), pygame)
        if self.bg_image_surface is None:
            pygame.draw.rect(screen, self.BG, area_rect)
            return
        screen.blit(self.bg_image_surface, area_rect.topleft, area_rect)
        tint = pygame.Surface(area_rect.size, pygame.SRCALPHA)
        if self.ui_theme == "light":
            tint.fill((243, 245, 249, 130))
        else:
            tint.fill((15, 19, 28, 135))
        screen.blit(tint, area_rect.topleft)

    def _apply_board_theme(self, board_theme_id: str) -> None:
        for tid, _, light, dark in self.BOARD_THEMES:
            if tid == board_theme_id:
                self.board_theme = tid
                self.LIGHT = light
                self.DARK = dark
                return

    def _get_piece_preview(self, pygame: Any, piece_set: str) -> Any:
        if piece_set in self._piece_preview_cache:
            return self._piece_preview_cache[piece_set]
        piece_dir = Path(__file__).resolve().parent / "assets" / "piece" / piece_set
        has_display = pygame.display.get_surface() is not None
        for fname in ("wN", "wQ", "wK"):
            for suffix in (".png", ".svg"):
                p = piece_dir / f"{fname}{suffix}"
                if p.exists():
                    try:
                        img = pygame.image.load(str(p))
                        if has_display:
                            img = img.convert_alpha()
                        scaled = pygame.transform.smoothscale(img, (26, 26))
                        self._piece_preview_cache[piece_set] = scaled
                        return scaled
                    except Exception:
                        continue
        self._piece_preview_cache[piece_set] = None
        return None

    def _handle_settings_click(self, x: int, y: int, pygame_mod: Any) -> None:
        mx, my, w, h = 300, 85, 680, 580
        if not (mx <= x <= mx + w and my <= y <= my + h):
            self.settings_open = False
            return
        lx, ly = x - mx, y - my
        # Nút đóng (X)
        if w - 46 <= lx <= w - 14 and 14 <= ly <= 46:
            self.settings_open = False
            return
        # Sub-tabs trong Settings (Giao diện | Bàn cờ & Quân | Âm thanh)
        if 56 <= ly <= 90:
            for idx, tab_key in enumerate(("appearance", "board", "audio")):
                tx = 24 + idx * 214
                if tx <= lx <= tx + 204:
                    self.settings_tab = tab_key
                    return

        if self.settings_tab == "appearance":
            # 1. Chế độ Sáng / Tối
            if 132 <= ly <= 170:
                if 24 <= lx <= 330:
                    self._apply_theme("dark")
                    return
                if 350 <= lx <= 656:
                    self._apply_theme("light")
                    return
            # 2. Tông màu chủ đạo (6 ô trên 1 hàng hoặc 3x2)
            for idx, (aid, _, _, _) in enumerate(self.ACCENT_THEMES):
                col, row = idx % 3, idx // 3
                bx = 24 + col * 214
                by = 212 + row * 44
                if bx <= lx <= bx + 204 and by <= ly <= by + 36:
                    self._apply_accent_theme(aid)
                    return
            # 3. Hình nền (Nút tải từ máy / Xóa nền + danh sách background)
            if 316 <= ly <= 346:
                if 416 <= lx <= 564:
                    chosen = self._browse_local_file(
                        "Chọn hình nền",
                        [("Hình ảnh", "*.png *.jpg *.jpeg *.webp *.bmp"), ("Tất cả tệp", "*.*")],
                    )
                    if chosen is not None:
                        self._set_background_image(chosen, pygame_mod)
                    return
                if 574 <= lx <= 656:
                    self._set_background_image(None)
                    return
            bg_files = self._list_background_images()
            list_x, list_y, list_w, list_h = 24, 354, 616, 152
            max_scroll = max(0, len(bg_files) - 4)
            if list_x + list_w + 4 <= lx <= list_x + list_w + 18 and list_y <= ly <= list_y + list_h and max_scroll > 0:
                ratio = max(0.0, min(1.0, (ly - list_y) / float(list_h)))
                self.bg_image_scroll = round(ratio * max_scroll)
                return
            if list_x <= lx <= list_x + list_w and list_y <= ly <= list_y + list_h:
                row_idx = (ly - list_y) // 38
                item_idx = self.bg_image_scroll + row_idx
                if 0 <= item_idx < len(bg_files):
                    self._set_background_image(bg_files[item_idx], pygame_mod)
                return

        elif self.settings_tab == "board":
            # 1. Màu sắc bàn cờ (3x2 grid)
            for idx, (tid, _, _, _) in enumerate(self.BOARD_THEMES):
                col, row = idx % 3, idx // 3
                bx = 24 + col * 214
                by = 132 + row * 48
                if bx <= lx <= bx + 204 and by <= ly <= by + 40:
                    self._apply_board_theme(tid)
                    return
            # 2. Kiểu dáng quân cờ (scrollable 2-column grid)
            list_x, list_y, list_w, list_h = 24, 272, 616, 232
            total_rows = (len(self.available_piece_sets) + 1) // 2
            max_scroll = max(0, total_rows - 5)
            if list_x + list_w + 4 <= lx <= list_x + list_w + 18 and list_y <= ly <= list_y + list_h and max_scroll > 0:
                ratio = max(0.0, min(1.0, (ly - list_y) / float(list_h)))
                self.settings_piece_scroll = round(ratio * max_scroll)
                return
            if list_x <= lx <= list_x + list_w and list_y <= ly <= list_y + list_h:
                rel_row = (ly - list_y) // 46
                col = 0 if lx < list_x + 308 else 1
                item_idx = (self.settings_piece_scroll + rel_row) * 2 + col
                if 0 <= item_idx < len(self.available_piece_sets):
                    chosen = self.available_piece_sets[item_idx]
                    if chosen != self.piece_set:
                        self.piece_set = chosen
                        if pygame_mod is not None:
                            self._load_piece_images(pygame_mod)
                return

        elif self.settings_tab == "audio":
            # 1. Thanh trượt Âm lượng SFX & Nhạc nền
            if 132 <= ly <= 174 and 210 <= lx <= 644:
                self.dragging_slider = "settings_sfx"
                self._update_slider_from_mouse(x)
                return
            if 184 <= ly <= 226 and 210 <= lx <= 644:
                self.dragging_slider = "settings_music"
                self._update_slider_from_mouse(x)
                return
            # 2. Nút Tải nhạc từ máy / Tắt nhạc
            if 246 <= ly <= 276:
                if 416 <= lx <= 564:
                    chosen = self._browse_local_file(
                        "Chọn nhạc nền",
                        [("Âm thanh", "*.mp3 *.wav *.ogg *.flac"), ("Tất cả tệp", "*.*")],
                    )
                    if chosen is not None:
                        self._set_background_music(chosen)
                    return
                if 574 <= lx <= 656:
                    self._set_background_music(None)
                    return
            music_files = self._list_music_files()
            list_x, list_y, list_w, list_h = 24, 286, 616, 218
            max_scroll = max(0, len(music_files) - 5)
            if list_x + list_w + 4 <= lx <= list_x + list_w + 18 and list_y <= ly <= list_y + list_h and max_scroll > 0:
                ratio = max(0.0, min(1.0, (ly - list_y) / float(list_h)))
                self.music_scroll = round(ratio * max_scroll)
                return
            if list_x <= lx <= list_x + list_w and list_y <= ly <= list_y + list_h:
                row_idx = (ly - list_y) // 42
                item_idx = self.music_scroll + row_idx
                if 0 <= item_idx < len(music_files):
                    self._set_background_music(music_files[item_idx])
                return

        # Nút Mặc định & Xong
        if 524 <= ly <= 560:
            if 24 <= lx <= 164:
                self.accent_theme = "blue"
                self._apply_theme("dark")
                self._apply_board_theme("classic_green")
                self._set_background_image(self._default_background_path(), pygame_mod)
                self._set_background_music(None)
                self._set_sfx_volume(0.8)
                self._set_music_volume(0.15)
                if self.piece_set != "cburnett":
                    self.piece_set = "cburnett"
                    if pygame_mod is not None:
                        self._load_piece_images(pygame_mod)
                self.settings_piece_scroll = 0
                return
            if 516 <= lx <= 656:
                self.settings_open = False
                return

    def _draw_settings_modal(self, screen: Any, pygame: Any) -> None:
        overlay = pygame.Surface(self.WINDOW_SIZE, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 165))
        screen.blit(overlay, (0, 0))
        mx, my, w, h = 300, 85, 680, 580
        rect = pygame.Rect(mx, my, w, h)
        pygame.draw.rect(screen, self.PANEL, rect, border_radius=12)
        pygame.draw.rect(screen, self.BORDER_STRONG, rect, 2, border_radius=12)

        font = pygame.font.SysFont("segoeui", 19, bold=True)
        sec_font = pygame.font.SysFont("segoeui", 14, bold=True)
        small = pygame.font.SysFont("segoeui", 13)
        mono = pygame.font.SysFont("consolas", 12, bold=True)

        self._blit_icon_text(
            screen, pygame, "settings", 20, self.ACCENT,
            font.render("CÀI ĐẶT HỆ THỐNG", True, self.TEXT),
            pos=(mx + 24, my + 16), gap=9,
        )
        close_btn = pygame.Rect(mx + w - 44, my + 14, 28, 28)
        pygame.draw.rect(screen, self.PANEL_2, close_btn, border_radius=6)
        pygame.draw.rect(screen, self.BORDER, close_btn, 1, border_radius=6)
        x_lbl = sec_font.render("X", True, self.MUTED)
        screen.blit(x_lbl, x_lbl.get_rect(center=close_btn.center))

        # Thanh sub-tabs trong Settings
        s_tabs = [
            ("appearance", "palette", "Giao diện & Nền"),
            ("board", "trophy", "Bàn cờ & Quân cờ"),
            ("audio", "sixteenth-notes", "Âm thanh & Nhạc"),
        ]
        for idx, (t_key, t_icon, t_label) in enumerate(s_tabs):
            t_rect = pygame.Rect(mx + 24 + idx * 214, my + 56, 204, 34)
            sel = self.settings_tab == t_key
            pygame.draw.rect(screen, self.SELECTED_BG if sel else self.PANEL_2, t_rect, border_radius=7)
            pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, t_rect, 2 if sel else 1, border_radius=7)
            t_color = self.ACCENT if sel else self.TEXT
            self._blit_icon_text(
                screen, pygame, t_icon, 14, self.ACCENT if sel else self.MUTED,
                sec_font.render(t_label, True, t_color),
                center_rect=t_rect, gap=7,
            )

        if self.settings_tab == "appearance":
            screen.blit(sec_font.render("1. Chế độ hiển thị:", True, self.MUTED), (mx + 24, my + 104))
            for key, icon_name, label, bx in (
                ("dark", "eclipse", "Chế độ Tối (Dark)", mx + 24),
                ("light", "palette", "Chế độ Sáng (Light)", mx + 350),
            ):
                btn = pygame.Rect(bx, my + 132, 306, 38)
                sel = self.ui_theme == key
                pygame.draw.rect(screen, self.SELECTED_BG if sel else self.PANEL_2, btn, border_radius=8)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 2 if sel else 1, border_radius=8)
                self._blit_icon_text(
                    screen, pygame, icon_name, 16, self.ACCENT if sel else self.MUTED,
                    sec_font.render(label, True, self.TEXT if sel else self.MUTED),
                    center_rect=btn, gap=8,
                )

            screen.blit(sec_font.render("2. Tông màu chủ đạo:", True, self.MUTED), (mx + 24, my + 184))
            for idx, (aid, title, dark_c, light_c) in enumerate(self.ACCENT_THEMES):
                col, row = idx % 3, idx // 3
                btn = pygame.Rect(mx + 24 + col * 214, my + 212 + row * 44, 204, 36)
                sel = self.accent_theme == aid
                swatch = light_c if self.ui_theme == "light" else dark_c
                pygame.draw.rect(screen, self.SELECTED_BG if sel else self.PANEL_2, btn, border_radius=7)
                pygame.draw.rect(screen, swatch if sel else self.BORDER, btn, 2 if sel else 1, border_radius=7)
                pygame.draw.circle(screen, swatch, (btn.x + 20, btn.centery), 8)
                screen.blit(small.render(title, True, self.TEXT if sel else self.MUTED), (btn.x + 36, btn.y + 9))

            curr_bg = Path(self.bg_image_path).name if self.bg_image_path else "Mặc định (Không)"
            screen.blit(sec_font.render(f"3. Hình nền ({curr_bg[:26]}):", True, self.MUTED), (mx + 24, my + 320))
            up_btn = pygame.Rect(mx + 416, my + 316, 148, 30)
            clr_btn = pygame.Rect(mx + 574, my + 316, 82, 30)
            pygame.draw.rect(screen, self.PANEL_2, up_btn, border_radius=6)
            pygame.draw.rect(screen, self.BORDER, up_btn, 1, border_radius=6)
            self._blit_icon_text(
                screen, pygame, "upload", 13, self.ACCENT,
                small.render("Tải từ máy...", True, self.ACCENT),
                center_rect=up_btn, gap=5,
            )
            pygame.draw.rect(screen, self.PANEL_2, clr_btn, border_radius=6)
            pygame.draw.rect(screen, self.BORDER, clr_btn, 1, border_radius=6)
            c_txt = small.render("Xóa nền", True, self.TEXT)
            screen.blit(c_txt, c_txt.get_rect(center=clr_btn.center))

            bg_files = self._list_background_images()
            list_rect = pygame.Rect(mx + 24, my + 354, 616, 152)
            pygame.draw.rect(screen, self.CARD_DARK, list_rect, border_radius=8)
            pygame.draw.rect(screen, self.BORDER, list_rect, 1, border_radius=8)
            if not bg_files:
                hint = small.render("Thư mục assets/image/background/ chưa có ảnh. Hãy bấm 'Tải từ máy...'!", True, self.MUTED)
                screen.blit(hint, hint.get_rect(center=list_rect.center))
            else:
                for r_i, p in enumerate(bg_files[self.bg_image_scroll : self.bg_image_scroll + 4]):
                    item_rect = pygame.Rect(mx + 30, my + 360 + r_i * 36, 604, 32)
                    sel = self.bg_image_path == str(p)
                    pygame.draw.rect(screen, self.SELECTED_BG if sel else self.SURFACE, item_rect, border_radius=6)
                    pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, item_rect, 2 if sel else 1, border_radius=6)
                    screen.blit(small.render(p.name, True, self.TEXT if sel else self.MUTED), (item_rect.x + 12, item_rect.y + 7))

        elif self.settings_tab == "board":
            screen.blit(sec_font.render("1. Màu sắc bàn cờ:", True, self.MUTED), (mx + 24, my + 104))
            for idx, (tid, title, light_c, dark_c) in enumerate(self.BOARD_THEMES):
                col, row = idx % 3, idx // 3
                btn = pygame.Rect(mx + 24 + col * 214, my + 132 + row * 48, 204, 40)
                sel = self.board_theme == tid
                pygame.draw.rect(screen, self.SELECTED_BG if sel else self.PANEL_2, btn, border_radius=7)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 2 if sel else 1, border_radius=7)
                pygame.draw.rect(screen, light_c, pygame.Rect(btn.x + 10, btn.y + 10, 18, 20), border_radius=3)
                pygame.draw.rect(screen, dark_c, pygame.Rect(btn.x + 28, btn.y + 10, 18, 20), border_radius=3)
                pygame.draw.rect(screen, self.BORDER, pygame.Rect(btn.x + 10, btn.y + 10, 36, 20), 1, border_radius=3)
                screen.blit(small.render(title, True, self.TEXT if sel else self.MUTED), (btn.x + 54, btn.y + 11))

            screen.blit(sec_font.render(f"2. Kiểu dáng quân cờ ({self.piece_set}):", True, self.MUTED), (mx + 24, my + 244))
            list_rect = pygame.Rect(mx + 24, my + 272, 616, 232)
            pygame.draw.rect(screen, self.CARD_DARK, list_rect, border_radius=8)
            pygame.draw.rect(screen, self.BORDER, list_rect, 1, border_radius=8)

            total_rows = (len(self.available_piece_sets) + 1) // 2
            visible_rows = 5
            for r_i in range(visible_rows):
                actual_row = self.settings_piece_scroll + r_i
                for col in range(2):
                    item_idx = actual_row * 2 + col
                    if item_idx >= len(self.available_piece_sets):
                        continue
                    pset = self.available_piece_sets[item_idx]
                    item_rect = pygame.Rect(mx + 30 + col * 304, my + 278 + r_i * 45, 296, 38)
                    sel = pset == self.piece_set
                    pygame.draw.rect(screen, self.SELECTED_BG if sel else self.SURFACE, item_rect, border_radius=6)
                    pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, item_rect, 2 if sel else 1, border_radius=6)
                    preview = self._get_piece_preview(pygame, pset)
                    if preview is not None:
                        screen.blit(preview, (item_rect.x + 8, item_rect.y + 6))
                    screen.blit(small.render(pset, True, self.TEXT if sel else self.MUTED), (item_rect.x + 42, item_rect.y + 10))

            max_sc = max(1, total_rows - visible_rows)
            sb = pygame.Rect(mx + 646, my + 272, 8, 232)
            pygame.draw.rect(screen, self.TAB_BAR, sb, border_radius=4)
            th_h = max(24, int(sb.height * (visible_rows / float(max(1, total_rows)))))
            th_y = sb.y + int((sb.height - th_h) * (self.settings_piece_scroll / float(max_sc)))
            pygame.draw.rect(screen, self.ACCENT, pygame.Rect(sb.x, th_y, 8, th_h), border_radius=4)

        elif self.settings_tab == "audio":
            screen.blit(sec_font.render("1. Điều chỉnh âm lượng:", True, self.MUTED), (mx + 24, my + 104))
            for row_y, label, val, max_v in (
                (my + 132, "Âm lượng Hiệu ứng (SFX):", self.sfx_volume, 1.0),
                (my + 184, "Âm lượng Nhạc nền:", self.music_volume, 0.24),
            ):
                box = pygame.Rect(mx + 24, row_y, 632, 42)
                pygame.draw.rect(screen, self.SURFACE, box, border_radius=8)
                pygame.draw.rect(screen, self.BORDER, box, 1, border_radius=8)
                screen.blit(small.render(label, True, self.TEXT), (box.x + 14, box.y + 12))
                track_x, track_w, track_y = mx + 220, 414, box.centery
                pygame.draw.rect(screen, self.CLOCK_BG, pygame.Rect(track_x, track_y - 3, track_w, 6), border_radius=3)
                fill_w = int(track_w * min(1.0, val / max_v))
                if fill_w > 0:
                    pygame.draw.rect(screen, self.ACCENT, pygame.Rect(track_x, track_y - 3, fill_w, 6), border_radius=3)
                knob_x = track_x + fill_w
                pygame.draw.circle(screen, self.SURFACE, (knob_x, track_y), 7)
                pygame.draw.circle(screen, self.ACCENT, (knob_x, track_y), 7, 2)

            curr_mus = Path(self.music_path).name if self.music_path else "Chưa chọn nhạc nền"
            screen.blit(sec_font.render(f"2. Nhạc nền ({curr_mus[:28]}):", True, self.MUTED), (mx + 24, my + 248))
            up_btn = pygame.Rect(mx + 416, my + 244, 148, 30)
            stop_btn = pygame.Rect(mx + 574, my + 244, 82, 30)
            pygame.draw.rect(screen, self.PANEL_2, up_btn, border_radius=6)
            pygame.draw.rect(screen, self.BORDER, up_btn, 1, border_radius=6)
            self._blit_icon_text(
                screen, pygame, "upload", 13, self.ACCENT,
                small.render("Tải từ máy...", True, self.ACCENT),
                center_rect=up_btn, gap=5,
            )
            pygame.draw.rect(screen, self.PANEL_2, stop_btn, border_radius=6)
            pygame.draw.rect(screen, self.BORDER, stop_btn, 1, border_radius=6)
            st_txt = small.render("Tắt nhạc", True, self.TEXT)
            screen.blit(st_txt, st_txt.get_rect(center=stop_btn.center))

            music_files = self._list_music_files()
            list_rect = pygame.Rect(mx + 24, my + 286, 616, 218)
            pygame.draw.rect(screen, self.CARD_DARK, list_rect, border_radius=8)
            pygame.draw.rect(screen, self.BORDER, list_rect, 1, border_radius=8)
            if not music_files:
                hint = small.render("Thư mục assets/music/ chưa có nhạc. Hãy bấm 'Tải từ máy...'!", True, self.MUTED)
                screen.blit(hint, hint.get_rect(center=list_rect.center))
            else:
                for r_i, p in enumerate(music_files[self.music_scroll : self.music_scroll + 5]):
                    item_rect = pygame.Rect(mx + 30, my + 292 + r_i * 41, 604, 36)
                    sel = self.music_path == str(p)
                    pygame.draw.rect(screen, self.SELECTED_BG if sel else self.SURFACE, item_rect, border_radius=6)
                    pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, item_rect, 2 if sel else 1, border_radius=6)
                    self._blit_icon_text(
                        screen, pygame, "sixteenth-notes", 14, self.ACCENT if sel else self.MUTED,
                        small.render(p.name, True, self.TEXT if sel else self.MUTED),
                        pos=(item_rect.x + 12, item_rect.y + 8), gap=8,
                    )

        # Nút Mặc định & Xong
        reset_btn = pygame.Rect(mx + 24, my + 524, 140, 36)
        done_btn = pygame.Rect(mx + 516, my + 524, 140, 36)
        pygame.draw.rect(screen, self.PANEL_2, reset_btn, border_radius=7)
        pygame.draw.rect(screen, self.BORDER, reset_btn, 1, border_radius=7)
        self._blit_icon_text(
            screen, pygame, "refresh-cw", 14, self.MUTED,
            sec_font.render("Mặc định", True, self.TEXT),
            center_rect=reset_btn, gap=6,
        )
        pygame.draw.rect(screen, self.ACCENT, done_btn, border_radius=7)
        done_lbl = sec_font.render("Xong", True, self.ACCENT_TEXT)
        screen.blit(done_lbl, done_lbl.get_rect(center=done_btn.center))

    def _draw(self, screen: Any, pygame: Any) -> None:
        screen.fill(self.BG)
        self._draw_tabs(screen, pygame)
        if self.active_tab == "play":
            self._draw_play_tab(screen, pygame)
        else:
            self._draw_lab_tab(screen, pygame)
        if self.modal_open:
            self._draw_modal(screen, pygame)
        if self.settings_open:
            self._draw_settings_modal(screen, pygame)

    def _draw_tabs(self, screen: Any, pygame: Any) -> None:
        pygame.draw.rect(screen, self.TAB_BAR, pygame.Rect(0, 0, 1280, 44))
        pygame.draw.line(screen, self.BORDER, (0, 43), (1280, 43), 1)
        tabs = [
            ("play", "play", "Play", pygame.Rect(12, 6, 218, 32)),
            (
                "lab",
                "settings",
                "Experiment (Tạm khóa)" if self.lab_tab_locked else "Experiment",
                pygame.Rect(240, 6, 240, 32),
            ),
        ]
        for key, icon_name, label, rect in tabs:
            is_active = self.active_tab == key
            pygame.draw.rect(
                screen,
                self.SELECTED_BG if is_active else self.SURFACE,
                rect,
                border_radius=7,
            )
            pygame.draw.rect(
                screen,
                self.ACCENT if is_active else self.BORDER,
                rect,
                2 if is_active else 1,
                border_radius=7,
            )
            font = pygame.font.SysFont("segoeui", 13 if not (key == "lab" and self.lab_tab_locked) else 12, bold=True)
            if key == "lab" and self.lab_tab_locked:
                muted = self.MUTED
            else:
                muted = self.ACCENT if is_active else self.TEXT
            text_surf = font.render(label, True, muted)
            self._blit_icon_text(
                screen,
                pygame,
                icon_name,
                15,
                muted,
                text_surf,
                center_rect=rect,
                gap=7,
            )

        # Nút Bật/Tắt âm thanh & Cài đặt giao diện ở góc trên cùng bên phải
        sound_btn = pygame.Rect(1104, 6, 34, 32)
        pygame.draw.rect(
            screen,
            self.SELECTED_BG if self.sound_enabled else self.SURFACE,
            sound_btn,
            border_radius=7,
        )
        pygame.draw.rect(
            screen,
            self.ACCENT if self.sound_enabled else self.BORDER,
            sound_btn,
            1,
            border_radius=7,
        )
        snd_icon = self._get_icon(
            pygame, "sixteenth-notes", 16, self.ACCENT if self.sound_enabled else self.MUTED
        )
        if snd_icon is not None:
            screen.blit(snd_icon, snd_icon.get_rect(center=sound_btn.center))
        if not self.sound_enabled:
            pygame.draw.line(
                screen,
                (235, 85, 85),
                (sound_btn.x + 7, sound_btn.bottom - 7),
                (sound_btn.right - 7, sound_btn.y + 7),
                2,
            )

        settings_btn = pygame.Rect(1146, 6, 122, 32)
        pygame.draw.rect(
            screen,
            self.SELECTED_BG if self.settings_open else self.SURFACE,
            settings_btn,
            border_radius=7,
        )
        pygame.draw.rect(
            screen,
            self.ACCENT if self.settings_open else self.BORDER,
            settings_btn,
            2 if self.settings_open else 1,
            border_radius=7,
        )
        s_font = pygame.font.SysFont("segoeui", 13, bold=True)
        s_color = self.ACCENT if self.settings_open else self.TEXT
        self._blit_icon_text(
            screen,
            pygame,
            "settings",
            15,
            s_color,
            s_font.render("Cài đặt", True, s_color),
            center_rect=settings_btn,
            gap=7,
        )

    def _draw_play_tab(self, screen: Any, pygame: Any) -> None:
        self._draw_config(screen, pygame)
        self._draw_arena(screen, pygame)
        self._draw_telemetry(screen, pygame)

    def _draw_config(self, screen: Any, pygame: Any) -> None:
        rect = pygame.Rect(0, self.TAB_HEIGHT, self.LEFT_WIDTH, 716)
        pygame.draw.rect(screen, self.SIDEBAR, rect)
        pygame.draw.line(screen, self.BORDER, (269, 44), (269, 760), 1)
        font = pygame.font.SysFont("segoeui", 15, bold=True)
        small = pygame.font.SysFont("segoeui", 13)
        mono = pygame.font.SysFont("consolas", 12)

        self._blit_icon_text(
            screen, pygame, "eclipse", 15, self.MUTED,
            font.render("BUDGET & SEARCH", True, self.MUTED),
            pos=(16, 56), gap=7,
        )
        is_time_mode = self.time_limit is not None

        curr_time = self.time_limit if self.time_limit is not None else self.modal_time
        self._draw_slider_control(
            screen, pygame, small, mono,
            x=12, y=84, width=246,
            title="Thời gian / nước",
            value_text=f"{curr_time:.1f}s",
            steps=self.TIME_STEPS,
            current_val=curr_time,
            active=is_time_mode,
        )

        curr_work = self.max_workload if self.max_workload is not None else self.modal_workload
        self._draw_slider_control(
            screen, pygame, small, mono,
            x=12, y=132, width=246,
            title="Số Nodes / Sims",
            value_text=f"{curr_work}",
            steps=self.WORKLOAD_STEPS,
            current_val=curr_work,
            active=not is_time_mode,
        )

        self._blit_icon_text(
            screen, pygame, "settings", 15, self.MUTED,
            font.render("CONTROLS", True, self.MUTED),
            pos=(16, 188), gap=7,
        )
        buttons = [
            ("Đảo bàn", "F", 216, "refresh-cw"),
            ("Undo", "U", 254, "rotate-ccw"),
            ("Đầu hàng", "R", 292, "power"),
            ("Cầu hòa", "D", 330, "flag"),
        ]
        for label, key, y, icon_name in buttons:
            rect = pygame.Rect(12, y, 246, 32)
            visible = label != "Cầu hòa" or self.game_mode != "ai_ai"
            if visible:
                pygame.draw.rect(screen, self.SURFACE, rect, border_radius=7)
                pygame.draw.rect(screen, self.BORDER, rect, 1, border_radius=7)
                lbl_surf = small.render(label, True, self.TEXT)
                if icon_name is not None:
                    self._blit_icon_text(
                        screen, pygame, icon_name, 14, self.MUTED,
                        lbl_surf, pos=(20, y + 7), gap=7,
                    )
                else:
                    screen.blit(lbl_surf, (20, y + 7))
                key_rect = pygame.Rect(208, y + 5, 42, 22)
                pygame.draw.rect(screen, self.PANEL_3, key_rect, border_radius=5)
                key_text = mono.render(key, True, self.MUTED)
                screen.blit(key_text, key_text.get_rect(center=key_rect.center))
        new_rect = pygame.Rect(12, 628, 246, 48)
        pygame.draw.rect(screen, self.ACCENT, new_rect, border_radius=7)
        pygame.draw.rect(screen, self.BORDER_STRONG, new_rect, 1, border_radius=7)
        self._blit_icon_text(
            screen, pygame, "play", 16, self.ACCENT_TEXT,
            font.render("VÁN MỚI", True, self.ACCENT_TEXT),
            pos=(54, 641), gap=8,
        )
        key_text = mono.render("[N]", True, self.ACCENT_TEXT)
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
        pygame.draw.rect(screen, self.SURFACE if active else self.PANEL_2, card, border_radius=7)
        pygame.draw.rect(screen, self.ACCENT if active else self.BORDER, card, 1, border_radius=7)
        screen.blit(small.render(title, True, self.TEXT if active else self.MUTED), (x + 10, y + 3))
        val_surf = mono.render(value_text, True, self.ACCENT if active else self.MUTED)
        screen.blit(val_surf, (x + width - 10 - val_surf.get_width(), y + 4))

        track_x = x + 12
        track_w = width - 24
        track_y = y + 26
        pygame.draw.rect(screen, self.CLOCK_BG, pygame.Rect(track_x, track_y - 3, track_w, 6), border_radius=3)

        best_idx = min(range(len(steps)), key=lambda i: abs(float(steps[i]) - float(current_val)))
        ratio = best_idx / float(max(1, len(steps) - 1))
        fill_w = int(track_w * ratio)
        fill_color = self.ACCENT if active else self.MUTED
        if fill_w > 0:
            pygame.draw.rect(screen, fill_color, pygame.Rect(track_x, track_y - 3, fill_w, 6), border_radius=3)
        for i in range(len(steps)):
            tx = track_x + int(track_w * (i / float(max(1, len(steps) - 1))))
            pygame.draw.circle(screen, fill_color if i <= best_idx else self.BORDER_STRONG, (tx, track_y), 3)
        knob_x = track_x + fill_w
        pygame.draw.circle(screen, self.SURFACE if active else self.MUTED, (knob_x, track_y), 6)
        pygame.draw.circle(screen, fill_color, (knob_x, track_y), 6, 2)

    def _draw_arena(self, screen: Any, pygame: Any) -> None:
        arena_x = self.LEFT_WIDTH
        rect = pygame.Rect(arena_x, self.TAB_HEIGHT, self.Arena_width if hasattr(self, "Arena_width") else self.ARENA_WIDTH, 716)
        self._draw_background_layer(screen, pygame, rect)
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
        pygame.draw.rect(screen, self.CARD_DARK, footer, border_radius=7)
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
        pygame.draw.rect(screen, self.SELECTED_BG if is_turn else self.SURFACE, rect, border_radius=8)
        if is_turn:
            pygame.draw.rect(screen, self.ACCENT, rect, 2, border_radius=8)
        font = pygame.font.SysFont("segoeui", 15, bold=True)
        mono = pygame.font.SysFont("consolas", 17, bold=True)
        is_human = (
            (side == "Trắng" and self.player_color == "white")
            or (side == "Đen" and self.player_color == "black")
            or self.game_mode == "human_human"
        )
        name = "Người chơi" if is_human else (
            getattr(self._agent_for_side("white" if side == "Trắng" else "black"), "name", "AI")
        )
        icon_name = "user" if is_human else "bot"
        self._blit_icon_text(
            screen,
            pygame,
            icon_name,
            16,
            self.ACCENT if is_turn else self.MUTED,
            font.render(f"{side}  ·  {name}", True, self.TEXT),
            pos=(x + 12, y + 9),
            gap=8,
        )
        clock_text = self._format_clock(clock)
        clock_surface = mono.render(clock_text, True, self.ORANGE if is_turn else self.TEXT)
        clock_box = pygame.Rect(x + width - 94, y + 6, 82, 28)
        pygame.draw.rect(screen, self.CLOCK_BG, clock_box, border_radius=6)
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
        pygame.draw.rect(screen, self.CARD_DARK, board_rect, border_radius=4)
        pygame.draw.rect(screen, self.BORDER_STRONG, board_rect, 2, border_radius=4)
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
            is_light_sq = (i + 7) % 2 == 0
            coord_color = self.DARK if is_light_sq else self.LIGHT
            label = small.render(file_name, True, coord_color)
            label_rect = label.get_rect(center=(board_x + i * tile + tile // 2, board_y + self.BOARD_SIZE - 11))
            screen.blit(label, label_rect)
        for i, rank_name in enumerate(ranks):
            is_light_sq = i % 2 == 0
            coord_color = self.DARK if is_light_sq else self.LIGHT
            label = small.render(rank_name, True, coord_color)
            label_rect = label.get_rect(center=(board_x + 10, board_y + i * tile + 12))
            screen.blit(label, label_rect)

        if self.promotion_pending is not None:
            cx, cy = board_x + 150, board_y + 220
            pygame.draw.rect(screen, self.PANEL, pygame.Rect(cx - 10, cy - 10, 236, 70), border_radius=8)
            pygame.draw.rect(screen, self.BORDER_STRONG, pygame.Rect(cx - 10, cy - 10, 236, 70), 1, border_radius=8)
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
            badge_bg = (220, 245, 230) if self.ui_theme == "light" else (22, 78, 52)
            icon_glyph = "♔"
        elif result == -1.0:
            title = "CHIẾN THẮNG CHO BÊN ĐEN!"
            score_str = "0 - 1"
            accent = self.ACCENT
            badge_bg = (219, 234, 254) if self.ui_theme == "light" else (24, 58, 104)
            icon_glyph = "♚"
        else:
            title = "VÁN ĐẤU KẾT THÚC HÒA!"
            score_str = "½ - ½"
            accent = self.ORANGE
            badge_bg = (254, 243, 199) if self.ui_theme == "light" else (92, 64, 18)
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

        pygame.draw.rect(screen, self.PANEL, card_rect, border_radius=14)
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
        new_text = btn_font.render("Ván mới [N]", True, self.ACCENT_TEXT)
        screen.blit(new_text, new_text.get_rect(center=new_btn.center))

    def _draw_telemetry(self, screen: Any, pygame: Any) -> None:
        rect = pygame.Rect(self.LEFT_WIDTH + self.ARENA_WIDTH, self.TAB_HEIGHT, self.RIGHT_WIDTH, 716)
        pygame.draw.rect(screen, self.SIDEBAR, rect)
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
        pygame.draw.rect(screen, self.CLOCK_BG, pygame.Rect(968, 94, 18, 106), border_radius=5)
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
        pygame.draw.circle(screen, self.CLOCK_BG, gauge.center, 48)
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
        pygame.draw.rect(screen, self.CARD_DARK, pygame.Rect(952, 468, 316, 184), border_radius=9)
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
                self.SELECTED_BG
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
            if is_pause_btn:
                icon = self._get_icon(pygame, "play" if self.paused else "square-pause", 16, text_color)
                if icon is not None:
                    screen.blit(icon, icon.get_rect(center=rect.center))
                else:
                    label_surface = mono.render(label, True, text_color)
                    screen.blit(label_surface, label_surface.get_rect(center=rect.center))
            else:
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

        self._blit_icon_text(
            screen, pygame, "settings", 20, self.ACCENT,
            font.render("THIẾT LẬP VÁN ĐẤU MỚI", True, self.TEXT),
            pos=(mx + 24, my + 18), gap=9,
        )

        # Nút Đổi bên Trắng <-> Đen nhanh
        swap_btn = pygame.Rect(mx + 445, my + 18, 130, 28)
        pygame.draw.rect(screen, self.PANEL_2, swap_btn, border_radius=6)
        pygame.draw.rect(screen, self.BORDER, swap_btn, 1, border_radius=6)
        swap_lbl = small.render("Đổi Trắng/Đen", True, self.ACCENT)
        self._blit_icon_text(
            screen, pygame, "refresh-cw", 13, self.ACCENT,
            swap_lbl, center_rect=swap_btn, gap=6,
        )

        participants = ["human"] + list_full_board_agents()
        short_names = {
            "human": "Người chơi",
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
            is_hover = box_rect.collidepoint(self.mouse_pos)
            box_bg = self.SELECTED_BG if is_open else (self.SURFACE_HOVER if is_hover else self.PANEL_2)
            box_border = self.ACCENT if (is_open or is_hover) else self.BORDER_STRONG
            pygame.draw.rect(screen, box_bg, box_rect, border_radius=7)
            pygame.draw.rect(
                screen,
                box_border,
                box_rect,
                2 if is_open else 1,
                border_radius=7,
            )
            label_str = short_names.get(current_pid, current_pid)
            p_icon = "user" if current_pid == "human" else "bot"
            self._blit_icon_text(
                screen, pygame, p_icon, 15, self.ACCENT if (is_open or is_hover) else self.MUTED,
                sec_font.render(label_str, True, self.TEXT),
                pos=(box_x + 14, my + 98), gap=7,
            )
            arrow = "▲" if is_open else "▼"
            screen.blit(small.render(arrow, True, self.ACCENT if (is_open or is_hover) else self.MUTED), (box_x + 236, my + 100))

        # 3. Đồng hồ thi đấu
        screen.blit(sec_font.render("3. Đồng hồ ván đấu (Phút + Giây cộng thêm):", True, self.MUTED), (mx + 24, my + 160))
        options = [(0, 0), (60, 0), (180, 2), (300, 0), (600, 5), (900, 10), (1800, 20)]
        for index, (base, inc) in enumerate(options):
            col, row = index % 4, index // 4
            button = pygame.Rect(mx + 24 + col * 140, my + 186 + row * 36, 132, 30)
            selected = self.modal_base == base and self.modal_increment == inc
            pygame.draw.rect(screen, self.SELECTED_BG if selected else self.PANEL_2, button, border_radius=6)
            pygame.draw.rect(screen, self.ACCENT if selected else self.BORDER, button, 2 if selected else 1, border_radius=6)
            label = "Không giới hạn (∞)" if base == 0 else f"{base // 60}p + {inc}s"
            lbl_surf = small.render(label, True, self.TEXT if selected else self.MUTED)
            screen.blit(lbl_surf, lbl_surf.get_rect(center=button.center))

        # 4. Ngân sách AI & Thanh trượt đa mức độ
        screen.blit(sec_font.render("4. Ngân sách suy nghĩ cho Agent:", True, self.MUTED), (mx + 24, my + 272))
        time_btn = pygame.Rect(mx + 24, my + 298, 260, 32)
        work_btn = pygame.Rect(mx + 304, my + 298, 272, 32)
        is_time = self.modal_budget == "time"
        pygame.draw.rect(screen, self.SELECTED_BG if is_time else self.PANEL_2, time_btn, border_radius=6)
        pygame.draw.rect(screen, self.ACCENT if is_time else self.BORDER, time_btn, 2 if is_time else 1, border_radius=6)
        pygame.draw.rect(screen, self.SELECTED_BG if not is_time else self.PANEL_2, work_btn, border_radius=6)
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
        pygame.draw.rect(screen, self.CLOCK_BG, pygame.Rect(track_x, track_y - 3, track_w, 6), border_radius=3)
        best_idx = min(range(len(steps)), key=lambda i: abs(float(steps[i]) - float(curr_val)))
        ratio = best_idx / float(max(1, len(steps) - 1))
        fill_w = int(track_w * ratio)
        if fill_w > 0:
            pygame.draw.rect(screen, self.ACCENT, pygame.Rect(track_x, track_y - 3, fill_w, 6), border_radius=3)
        for i, step_v in enumerate(steps):
            tx = track_x + int(track_w * (i / float(max(1, len(steps) - 1))))
            pygame.draw.circle(screen, self.ACCENT if i <= best_idx else self.BORDER_STRONG, (tx, track_y), 4)
            step_str = f"{step_v:g}s" if is_time else (f"{step_v // 1000}k" if step_v >= 1000 and step_v % 1000 == 0 else str(step_v))
            s_lbl = pygame.font.SysFont("consolas", 10).render(step_str, True, self.ACCENT if i == best_idx else self.MUTED)
            screen.blit(s_lbl, s_lbl.get_rect(center=(tx, track_y + 14)))
        knob_x = track_x + fill_w
        pygame.draw.circle(screen, self.SURFACE, (knob_x, track_y), 7)
        pygame.draw.circle(screen, self.ACCENT, (knob_x, track_y), 7, 2)

        cancel = pygame.Rect(mx + 60, my + 426, 180, 44)
        start = pygame.Rect(mx + 340, my + 426, 200, 44)
        pygame.draw.rect(screen, self.PANEL_2, cancel, border_radius=8)
        pygame.draw.rect(screen, self.BORDER, cancel, 1, border_radius=8)
        pygame.draw.rect(screen, self.ACCENT, start, border_radius=8)
        c_lbl = sec_font.render("Hủy [Esc]", True, self.TEXT)
        s_lbl = sec_font.render("Bắt đầu ván đấu", True, self.ACCENT_TEXT)
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
            pygame.draw.rect(screen, self.CARD_DARK, drop_rect, border_radius=8)
            pygame.draw.rect(screen, self.ACCENT, drop_rect, 2, border_radius=8)

            visible_slice = participants[self.modal_dropdown_scroll : self.modal_dropdown_scroll + visible_rows]
            for r_idx, pid in enumerate(visible_slice):
                item_rect = pygame.Rect(drop_x + 4, drop_y + r_idx * row_h + 3, 238, row_h - 6)
                is_sel = pid == current_pid
                is_hover = item_rect.collidepoint(self.mouse_pos)
                if is_sel:
                    pygame.draw.rect(screen, self.SELECTED_BG, item_rect, border_radius=6)
                elif is_hover:
                    pygame.draw.rect(screen, self.SURFACE_HOVER, item_rect, border_radius=6)
                    pygame.draw.rect(screen, self.ACCENT, item_rect, 1, border_radius=6)
                p_icon = "user" if pid == "human" else "bot"
                txt_color = self.ACCENT if (is_sel or is_hover) else self.TEXT
                lbl = small.render(short_names.get(pid, pid), True, txt_color)
                self._blit_icon_text(
                    screen, pygame, p_icon, 13, txt_color if (is_sel or is_hover) else self.MUTED,
                    lbl, pos=(item_rect.x + 10, item_rect.y + 5), gap=7,
                )

            # Thanh cuộn (Scrollbar) bên phải hộp thoại trượt
            max_scroll = max(1, len(participants) - visible_rows)
            sb_track = pygame.Rect(drop_x + 246, drop_y + 6, 8, list_h - 12)
            pygame.draw.rect(screen, self.CLOCK_BG, sb_track, border_radius=4)
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
            ("Cấp 5 SL: Loss & Acc", "fig_level5_training_curves.png"),
            ("Cấp 8 AZ: Self-Play Gating", "fig_level8_selfplay_gating.png"),
            ("Round-Robin 7×7 (4.200 ván)", "fig_round_robin_7x7_200.png"),
            ("Dual-Mode: WL vs TC", "fig_dual_mode_200games.png"),
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
                _plot_ablation1(self.lab_results["ablation1"], out_dir, theme="dark")
            if "ablation2" in self.lab_results:
                _plot_ablation2(self.lab_results["ablation2"], out_dir, theme="dark")
            if "rl_krk" in self.lab_results:
                _plot_rl_krk(self.lab_results["rl_krk"], out_dir, theme="dark")
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
        arena_candidates = [
            Path("results/round_robin_200/summary_7x7_200.json"),
            Path("results/round_robin_200/summary_6x6_200.json"),
            Path("results/full_leaderboard/summary.json"),
            Path("results/round_robin/tournament_summary.json"),
            Path("results/tournament_summary.json"),
        ]
        arena_path = next((p for p in arena_candidates if p.exists()), arena_candidates[-1])
        mapping = {
            "arena": arena_path,
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
        self._draw_background_layer(screen, pygame, pygame.Rect(340, 44, 940, 716))
        pygame.draw.rect(screen, self.SIDEBAR, pygame.Rect(0, 44, 340, 716))
        pygame.draw.line(screen, self.BORDER, (339, 44), (339, 760), 1)
        font = pygame.font.SysFont("segoeui", 15, bold=True)
        sec_font = pygame.font.SysFont("segoeui", 13, bold=True)
        small = pygame.font.SysFont("segoeui", 12)
        mono = pygame.font.SysFont("consolas", 12)

        # --- CỘT TRÁI: 1. KỊCH BẢN THÍ NGHIỆM ---
        screen.blit(font.render("1. KỊCH BẢN THÍ NGHIỆM", True, self.MUTED), (18, 56))
        scenarios = [
            ("arena", "trophy", "Đấu Cặp / Giải Đấu (Arena)"),
        ]
        for index, (key, icon_name, label) in enumerate(scenarios):
            y = 86 + index * 40
            selected = self.lab_scenario == key
            card = pygame.Rect(18, y, 304, 34)
            pygame.draw.rect(
                screen,
                self.SELECTED_BG if selected else self.SURFACE,
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
            txt_color = self.TEXT if selected else self.MUTED
            self._blit_icon_text(
                screen, pygame, icon_name, 15, self.ACCENT if selected else self.MUTED,
                sec_font.render(label, True, txt_color),
                pos=(28, y + 8), gap=8,
            )
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
        pygame.draw.rect(screen, self.CARD_DARK, setup_card, border_radius=9)
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
                pygame.draw.rect(screen, self.SELECTED_BG if is_open else self.SURFACE, d_rect, border_radius=6)
                pygame.draw.rect(screen, self.ACCENT if is_open else self.BORDER, d_rect, 1, border_radius=6)
                screen.blit(small.render(short_names.get(aid, aid), True, self.TEXT), (bx + 8, 303))
                screen.blit(small.render("▲" if is_open else "▼", True, self.ACCENT), (bx + 128, 303))

            screen.blit(small.render("Số ván đấu (đối xứng 2 màu):", True, self.MUTED), (20, 336))
            game_opts = [4, 10, 20, 50, 100]
            for idx, g_num in enumerate(game_opts):
                btn = pygame.Rect(18 + idx * 61, 356, 57, 28)
                sel = self.lab_num_games == g_num
                pygame.draw.rect(screen, self.SELECTED_BG if sel else self.SURFACE, btn, border_radius=5)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=5)
                lbl = mono.render(str(g_num), True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))

            screen.blit(small.render("Chế độ Ngân sách:", True, self.MUTED), (20, 392))
            for b_mode, b_lbl, bx in (("workload", "Workload (Nodes/Sims)", 18), ("time", "Time Limit (s)", 176)):
                sel = self.lab_budget_mode == b_mode
                btn = pygame.Rect(bx, 412, 146, 26)
                pygame.draw.rect(screen, self.SELECTED_BG if sel else self.SURFACE, btn, border_radius=5)
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
                pygame.draw.rect(screen, self.SELECTED_BG if sel else self.SURFACE, btn, border_radius=5)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=5)
                lbl = mono.render(f"{w_num} CPU", True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))

        elif self.lab_scenario == "ablation1":
            screen.blit(small.render("Số thế cờ chiến thuật WAC (tối đa 50):", True, self.MUTED), (20, 280))
            for idx, p_num in enumerate((5, 10, 25, 50)):
                btn = pygame.Rect(18 + idx * 76, 300, 70, 30)
                sel = self.lab_ab1_positions == p_num
                pygame.draw.rect(screen, self.SELECTED_BG if sel else self.SURFACE, btn, border_radius=6)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=6)
                lbl = mono.render(f"{p_num} FEN", True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))

            screen.blit(small.render("Độ sâu cơ sở (Base Depth d):", True, self.MUTED), (20, 342))
            for idx, d_num in enumerate((2, 3, 4)):
                btn = pygame.Rect(18 + idx * 102, 362, 96, 30)
                sel = self.lab_ab1_depth == d_num
                pygame.draw.rect(screen, self.SELECTED_BG if sel else self.SURFACE, btn, border_radius=6)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=6)
                lbl = mono.render(f"Depth d = {d_num}", True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))

            chk_rect = pygame.Rect(18, 418, 304, 32)
            pygame.draw.rect(
                screen,
                self.SELECTED_BG if self.lab_ab1_skip_minimax else self.SURFACE,
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
                pygame.draw.rect(screen, self.SELECTED_BG if sel else self.SURFACE, btn, border_radius=6)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=6)
                lbl = mono.render(f"{op_num * 2} ván", True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))

            screen.blit(small.render("Tập mốc ngân sách mô phỏng N_sim:", True, self.MUTED), (20, 342))
            for p_key, p_lbl, bx in (("fast", "Nhanh [50, 100, 200]", 18), ("full", "Đầy đủ [50..1000]", 176)):
                sel = self.lab_ab2_Preset == p_key
                btn = pygame.Rect(bx, 362, 146, 30)
                pygame.draw.rect(screen, self.SELECTED_BG if sel else self.SURFACE, btn, border_radius=6)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=6)
                lbl = small.render(p_lbl, True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))

            screen.blit(small.render("Workers:", True, self.MUTED), (20, 430))
            for idx, w_num in enumerate((1, 2, 4)):
                btn = pygame.Rect(110 + idx * 70, 424, 62, 28)
                sel = self.lab_workers == w_num
                pygame.draw.rect(screen, self.SELECTED_BG if sel else self.SURFACE, btn, border_radius=5)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=5)
                lbl = mono.render(f"{w_num} CPU", True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))
            screen.blit(small.render("Đo đường cong điểm S của MCTS theo N_sim", True, self.MUTED), (20, 474))

        elif self.lab_scenario == "rl_krk":
            screen.blit(small.render("Số Episodes Tabular Q-Learning / cờ tàn:", True, self.MUTED), (20, 280))
            for idx, ep_num in enumerate((500, 1000, 2500, 5000)):
                btn = pygame.Rect(18 + idx * 76, 300, 70, 30)
                sel = self.lab_rl_episodes == ep_num
                pygame.draw.rect(screen, self.SELECTED_BG if sel else self.SURFACE, btn, border_radius=6)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=6)
                lbl = mono.render(str(ep_num), True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))

            screen.blit(small.render("Số Episodes Policy Gradient (REINFORCE KRK):", True, self.MUTED), (20, 342))
            for idx, rf_num in enumerate((100, 500, 1000, 3000)):
                btn = pygame.Rect(18 + idx * 76, 362, 70, 30)
                sel = self.lab_rl_reinforce == rf_num
                pygame.draw.rect(screen, self.SELECTED_BG if sel else self.SURFACE, btn, border_radius=6)
                pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=6)
                lbl = mono.render(str(rf_num), True, self.TEXT if sel else self.MUTED)
                screen.blit(lbl, lbl.get_rect(center=btn.center))

            screen.blit(small.render("Cờ tàn huấn luyện: KQK -> KRK -> KPK (Curriculum)", True, self.MUTED), (20, 416))
            screen.blit(small.render("Lưu Q-Table sandbox: results/ui_sandbox/l6_qtable.pkl", True, self.MUTED), (20, 440))
            screen.blit(small.render("Lưu Log sandbox: results/ui_sandbox/krk_curve.json", True, self.MUTED), (20, 464))

        # --- CỘT TRÁI: 3. TIẾN ĐỘ & NÚT ĐIỀU KHIỂN ---
        progress_width = int(304 * max(0.0, min(1.0, self.lab_progress)))
        pygame.draw.rect(screen, self.CLOCK_BG, pygame.Rect(18, 554, 304, 18), border_radius=9)
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

        r_txt = sec_font.render("Đang chạy..." if self.lab_running else "Chạy thực nghiệm", True, self.ACCENT_TEXT)
        s_txt = sec_font.render("Dừng", True, (255, 255, 255))
        l_txt = small.render("Nạp lại kết quả có sẵn từ results/", True, self.ACCENT)
        self._blit_icon_text(
            screen, pygame, "refresh-cw" if self.lab_running else "play", 15, self.ACCENT_TEXT,
            r_txt, center_rect=run_btn, gap=7,
        )
        self._blit_icon_text(
            screen, pygame, "power", 14, (255, 255, 255),
            s_txt, center_rect=stop_btn, gap=6,
        )
        self._blit_icon_text(
            screen, pygame, "upload", 14, self.ACCENT,
            l_txt, center_rect=load_btn, gap=7,
        )

        # --- Vẽ Dropdown chọn Agent A / B nếu đang mở ---
        if self.lab_dropdown_open is not None and self.lab_scenario == "arena":
            drop_x = 18 if self.lab_dropdown_open == "a" else 176
            drop_y = 328
            visible_rows = 5
            row_h = 28
            list_h = visible_rows * row_h
            curr_aid = self.lab_agent_a if self.lab_dropdown_open == "a" else self.lab_agent_b
            d_box = pygame.Rect(drop_x, drop_y, 146, list_h)
            pygame.draw.rect(screen, self.TAB_INACTIVE, d_box, border_radius=7)
            pygame.draw.rect(screen, self.ACCENT, d_box, 2, border_radius=7)
            vis_agents = self.lab_agents[self.lab_dropdown_scroll : self.lab_dropdown_scroll + visible_rows]
            for r_i, aid in enumerate(vis_agents):
                row_rect = pygame.Rect(drop_x + 3, drop_y + r_i * row_h + 2, 130, row_h - 4)
                if aid == curr_aid:
                    pygame.draw.rect(screen, self.SELECTED_BG, row_rect, border_radius=5)
                screen.blit(
                    small.render(short_names.get(aid, aid), True, self.ACCENT if aid == curr_aid else self.TEXT),
                    (row_rect.x + 6, row_rect.y + 4),
                )
            max_sc = max(1, len(self.lab_agents) - visible_rows)
            sb = pygame.Rect(drop_x + 136, drop_y + 4, 6, list_h - 8)
            pygame.draw.rect(screen, self.CLOCK_BG, sb, border_radius=3)
            th_h = max(18, int(sb.height * (visible_rows / float(len(self.lab_agents)))))
            th_y = sb.y + int((sb.height - th_h) * (self.lab_dropdown_scroll / float(max_sc)))
            pygame.draw.rect(screen, self.ACCENT, pygame.Rect(sb.x, th_y, 6, th_h), border_radius=3)

        # --- KHUNG PHẢI: THANH SUB-TABS & BẢNG KẾT QUẢ / BIỂU ĐỒ ---
        pygame.draw.rect(screen, self.SIDEBAR, pygame.Rect(340, 44, 940, 44))
        view_tabs = [
            ("figures", "palette", "Biểu đồ"),
            ("logs", "database", "Nhật ký"),
        ]
        for index, (key, icon_name, title_v) in enumerate(view_tabs):
            rect = pygame.Rect(354 + index * 190, 50, 180, 32)
            selected = self.lab_view == key
            pygame.draw.rect(
                screen,
                self.SELECTED_BG if selected else self.SURFACE,
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
            txt_color = self.TEXT if selected else self.MUTED
            lbl_v = small.render(title_v, True, txt_color)
            self._blit_icon_text(
                screen, pygame, icon_name, 14, self.ACCENT if selected else self.MUTED,
                lbl_v, center_rect=rect, gap=6,
            )

        self._draw_lab_right_content(screen, pygame, font, sec_font, small, mono)

    def _get_mini_pieces(self, pygame: Any, target_px: int) -> dict[str, Any]:
        cache_key = (self.piece_set, target_px)
        if cache_key in self._mini_piece_cache:
            return self._mini_piece_cache[cache_key]
        piece_dir = Path(__file__).resolve().parent / "assets" / "piece" / self.piece_set
        names = {
            "K": "wK", "Q": "wQ", "R": "wR", "B": "wB", "N": "wN", "P": "wP",
            "k": "bK", "q": "bQ", "r": "bR", "b": "bB", "n": "bN", "p": "bP",
        }
        has_display = pygame.display.get_surface() is not None
        images: dict[str, Any] = {}
        for symbol, fname in names.items():
            for suffix in (".png", ".svg"):
                p = piece_dir / f"{fname}{suffix}"
                if p.exists():
                    try:
                        img = pygame.image.load(str(p))
                        if has_display:
                            img = img.convert_alpha()
                        images[symbol] = pygame.transform.smoothscale(img, (target_px, target_px))
                        break
                    except Exception:
                        continue
        self._mini_piece_cache[cache_key] = images
        return images

    def _render_mini_board(
        self,
        pygame: Any,
        fen: str,
        last_move_uci: Optional[str],
        board_px: int = 136,
    ) -> Any:
        key = (fen, last_move_uci, self.LIGHT, self.DARK, self.piece_set, board_px)
        if key in self._mini_board_cache:
            return self._mini_board_cache[key]
        import chess

        tile = board_px // 8
        actual = tile * 8
        surf = pygame.Surface((actual, actual))
        board = chess.Board(fen)
        last_sqs: set[int] = set()
        if last_move_uci:
            try:
                mv = chess.Move.from_uci(last_move_uci)
                last_sqs = {mv.from_square, mv.to_square}
            except Exception:
                pass
        pieces = self._get_mini_pieces(pygame, max(10, tile - 2))
        for sq in range(64):
            f_idx = chess.square_file(sq)
            r_idx = chess.square_rank(sq)
            col, row = f_idx, 7 - r_idx
            rect = pygame.Rect(col * tile, row * tile, tile, tile)
            color = self.LIGHT if (f_idx + r_idx) % 2 == 1 else self.DARK
            if sq in last_sqs:
                color = self.LAST_MOVE
            pygame.draw.rect(surf, color, rect)
            pc = board.piece_at(sq)
            if pc is not None:
                sp = pieces.get(pc.symbol())
                if sp is not None:
                    surf.blit(sp, sp.get_rect(center=rect.center))
        if len(self._mini_board_cache) > 96:
            self._mini_board_cache.clear()
        self._mini_board_cache[key] = surf
        return surf

    def _draw_lab_right_content(
        self, screen: Any, pygame: Any, font: Any, sec_font: Any, small: Any, mono: Any
    ) -> None:
        panel_rect = pygame.Rect(356, 98, 906, 646)
        pygame.draw.rect(screen, self.CLOCK_BG, panel_rect, border_radius=10)
        pygame.draw.rect(screen, self.BORDER, panel_rect, 1, border_radius=10)

        if self.lab_view == "arena":
            screen.blit(font.render("KẾT QUẢ ĐẤU CẶP ĐỐI XỨNG (ARENA PAIRED MATCHUP)", True, self.ACCENT), (376, 114))
            with self.lab_live_lock:
                data = dict(self.lab_results["arena"]) if isinstance(self.lab_results.get("arena"), dict) else None
                live_games = [dict(g) for g in self.lab_live_games]
            if isinstance(data, dict) and "agent_a" in data:
                a_name = data.get("agent_a", "Agent A")
                b_name = data.get("agent_b", "Agent B")
                games = int(data.get("games", 0))
                total_g = int(data.get("total_games", games) or games)
                wins = int(data.get("wins", 0))
                draws = int(data.get("draws", 0))
                losses = int(data.get("losses", 0))
                score = float(data.get("score", 0.0))
                w_ci = data.get("wilson_ci_95", [0.0, 0.0])
                b_ci = data.get("bootstrap_ci_95", [0.0, 0.0])

                screen.blit(
                    sec_font.render(
                        f"Cặp đấu:  {a_name}   vs   {b_name}   ({games}/{total_g} ván đối xứng 2 màu)",
                        True,
                        self.TEXT,
                    ),
                    (376, 146),
                )

                # Thanh tỷ lệ Thắng / Hòa / Thua trực quan
                bar_rect = pygame.Rect(376, 174, 860, 24)
                pygame.draw.rect(screen, self.SURFACE, bar_rect, border_radius=6)
                if games > 0:
                    w_w = int(860 * (wins / games))
                    d_w = int(860 * (draws / games))
                    l_w = 860 - w_w - d_w
                    if w_w > 0:
                        pygame.draw.rect(screen, self.GREEN, pygame.Rect(376, 174, w_w, 24), border_radius=5)
                    if d_w > 0:
                        pygame.draw.rect(screen, self.ORANGE, pygame.Rect(376 + w_w, 174, d_w, 24))
                    if l_w > 0:
                        pygame.draw.rect(screen, (239, 68, 68), pygame.Rect(376 + w_w + d_w, 174, l_w, 24), border_radius=5)

                # 4 thẻ chỉ số thống kê trực tiếp
                cards = [
                    ("Thắng / Hòa / Thua (A)", f"+{wins}  = {draws}  -{losses}  ({games} ván)"),
                    ("Điểm chuẩn hóa (S)", f"{score:.3f}  (W: {data.get('white_score', 0):.2f} | B: {data.get('black_score', 0):.2f})"),
                    ("Wilson / Bootstrap 95% CI", f"W:[{w_ci[0]:.3f}, {w_ci[1]:.3f}]  B:[{b_ci[0]:.3f}, {b_ci[1]:.3f}]"),
                    ("Độ trễ TB (A vs B)", f"A: {data.get('agent_a_avg_latency_ms', 0):.1f}ms  |  B: {data.get('agent_b_avg_latency_ms', 0):.1f}ms"),
                ]
                for idx, (k_lbl, v_lbl) in enumerate(cards):
                    col, row = idx % 2, idx // 2
                    c_rect = pygame.Rect(376 + col * 436, 206 + row * 48, 420, 42)
                    pygame.draw.rect(screen, self.SURFACE, c_rect, border_radius=8)
                    pygame.draw.rect(screen, self.BORDER, c_rect, 1, border_radius=8)
                    screen.blit(small.render(k_lbl, True, self.MUTED), (c_rect.x + 12, c_rect.y + 3))
                    screen.blit(mono.render(v_lbl, True, self.TEXT), (c_rect.x + 12, c_rect.y + 21))

                if live_games:
                    screen.blit(
                        sec_font.render(
                            f"DIỄN BIẾN CÁC VÁN ĐẤU ({games}/{total_g} hoàn tất · Cuộn chuột để xem tất cả)",
                            True,
                            self.ACCENT,
                        ),
                        (376, 308),
                    )
                    total_rows = (len(live_games) + 2) // 3
                    visible_rows = 2
                    max_sc = max(0, total_rows - visible_rows)
                    self.lab_games_scroll = max(0, min(max_sc, self.lab_games_scroll))

                    for r_i in range(visible_rows):
                        row_idx = self.lab_games_scroll + r_i
                        for col in range(3):
                            g_idx = row_idx * 3 + col
                            if g_idx >= len(live_games):
                                continue
                            g = live_games[g_idx]
                            cx = 376 + col * 284
                            cy = 332 + r_i * 202
                            card = pygame.Rect(cx, cy, 274, 192)
                            st = g.get("status", "waiting")
                            border_c = self.ACCENT if st == "playing" else (self.GREEN if st == "done" else self.BORDER)
                            pygame.draw.rect(screen, self.SURFACE, card, border_radius=8)
                            pygame.draw.rect(screen, border_c, card, 2 if st == "playing" else 1, border_radius=8)

                            w_lbl = "A" if g.get("a_is_white") else "B"
                            b_lbl = "B" if g.get("a_is_white") else "A"
                            title_g = f"#{g_idx + 1} · Trắng:{w_lbl} vs Đen:{b_lbl}"
                            screen.blit(small.render(title_g, True, self.TEXT), (cx + 10, cy + 6))

                            res_txt = g.get("result") or ("Đang đấu" if st == "playing" else "Chờ")
                            res_col = self.ACCENT if st == "playing" else (self.GREEN if g.get("result") == "1-0" else (self.ORANGE if g.get("result") == "1/2-1/2" else self.MUTED))
                            r_surf = mono.render(str(res_txt), True, res_col)
                            screen.blit(r_surf, (card.right - r_surf.get_width() - 8, cy + 7))

                            mini_surf = self._render_mini_board(
                                pygame,
                                str(g.get("fen", "")),
                                g.get("last_move"),
                                board_px=136,
                            )
                            screen.blit(mini_surf, (cx + (274 - mini_surf.get_width()) // 2, cy + 28))

                            plies_v = int(g.get("plies", 0))
                            foot = mono.render(f"Ply: {plies_v}  |  {g.get('opening_id', '')}", True, self.MUTED)
                            screen.blit(foot, (cx + 10, cy + 170))

                    # Thanh cuộn (Scrollbar) bên phải khu vực bàn cờ
                    sb = pygame.Rect(1234, 332, 10, 394)
                    pygame.draw.rect(screen, self.PANEL_2, sb, border_radius=5)
                    if max_sc > 0:
                        th_h = max(36, int(sb.height * (visible_rows / float(total_rows))))
                        th_y = sb.y + int((sb.height - th_h) * (self.lab_games_scroll / float(max_sc)))
                        pygame.draw.rect(screen, self.ACCENT, pygame.Rect(sb.x, th_y, 10, th_h), border_radius=5)
                else:
                    chart_surf = self._get_matplotlib_surface("arena_matchup.png", pygame, max_w=856, max_h=380)
                    if chart_surf is not None:
                        screen.blit(chart_surf, chart_surf.get_rect(center=(809, 535)))
            elif isinstance(data, dict) and "matchups" in data:
                matchups = data.get("matchups", [])
                headers = ["Cặp đấu (Agent A vs Agent B)", "Số ván", "W / D / L (A)", "Điểm S (A)", "Wilson 95% CI", "Bootstrap 95% CI", "Trễ TB A / B"]
                col_x = [376, 610, 675, 795, 880, 1015, 1145]
                pygame.draw.rect(screen, self.PANEL_2, pygame.Rect(370, 146, 876, 28), border_radius=6)
                for h_txt, cx in zip(headers, col_x):
                    screen.blit(sec_font.render(h_txt, True, self.ACCENT), (cx, 152))

                for idx, m in enumerate(matchups[:6]):
                    ry = 178 + idx * 30
                    row_rect = pygame.Rect(370, ry, 876, 27)
                    pygame.draw.rect(screen, self.SURFACE if idx % 2 == 0 else self.ROW_ALT, row_rect, border_radius=5)
                    w_ci = m.get("wilson_ci_95", [0, 0])
                    b_ci = m.get("bootstrap_ci_95", [0, 0])
                    a_short = str(m.get("agent_a", "")).replace("Level", "L")
                    b_short = str(m.get("agent_b", "")).replace("Level", "L")
                    vals = [
                        f"{a_short} vs {b_short}",
                        str(m.get("games", 0)),
                        f"+{m.get('wins', 0)}/={m.get('draws', 0)}/-{m.get('losses', 0)}",
                        f"{float(m.get('score', 0)):.3f}",
                        f"[{w_ci[0]:.2f}, {w_ci[1]:.2f}]",
                        f"[{b_ci[0]:.2f}, {b_ci[1]:.2f}]",
                        f"{float(m.get('agent_a_avg_latency_ms', 0)):.0f}/{float(m.get('agent_b_avg_latency_ms', 0)):.0f}ms",
                    ]
                    for v_txt, cx in zip(vals, col_x):
                        screen.blit(mono.render(v_txt, True, self.TEXT), (cx, ry + 5))

                chart_surf = self._get_matplotlib_surface("arena_matchup.png", pygame, max_w=856, max_h=365)
                if chart_surf is not None:
                    screen.blit(chart_surf, chart_surf.get_rect(center=(809, 550)))
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
                    pygame.draw.rect(screen, self.SURFACE if idx % 2 == 0 else self.ROW_ALT, row_rect, border_radius=5)
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
                    pygame.draw.rect(screen, self.SURFACE if idx % 2 == 0 else self.ROW_ALT, row_rect, border_radius=5)
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
                    pygame.draw.rect(screen, self.SURFACE if idx % 2 == 0 else self.ROW_ALT, row_rect, border_radius=5)
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
                shown_files = fig_files[:8]
                for idx, (f_title, _) in enumerate(shown_files):
                    row, col = divmod(idx, 4)
                    btn = pygame.Rect(376 + col * 218, 106 + row * 32, 210, 28)
                    sel = idx == self.lab_selected_figure_idx
                    pygame.draw.rect(screen, self.SELECTED_BG if sel else self.SURFACE, btn, border_radius=6)
                    pygame.draw.rect(screen, self.ACCENT if sel else self.BORDER, btn, 1, border_radius=6)
                    lbl = small.render(f_title, True, self.TEXT if sel else self.MUTED)
                    screen.blit(lbl, lbl.get_rect(center=btn.center))

                has_two_rows = len(shown_files) > 4
                _, f_path = fig_files[self.lab_selected_figure_idx]
                cached = self._get_matplotlib_surface(
                    f_path.name,
                    pygame,
                    max_w=870,
                    max_h=545 if has_two_rows else 575,
                )
                if cached is not None:
                    img_rect = cached.get_rect(center=(809, 452 if has_two_rows else 440))
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
                import chess
                from chess_ai.arena.match import play_single_match
                from chess_ai.arena.metrics import (
                    compute_normalized_score,
                    paired_bootstrap_confidence_interval,
                    wilson_confidence_interval,
                )
                from chess_ai.arena.pgn_exporter import append_pgn_file, export_game_to_pgn
                from chess_ai.arena.tournament import append_match_jsonl
                from chess_ai.utils.epd import load_epd_file

                num_openings = max(1, self.lab_num_games // 2)
                openings = load_epd_file("resources/openings_50.epd")[:num_openings]
                total_games = len(openings) * 2
                b_mode = "time_control" if self.lab_budget_mode == "time" else "workload_control"
                t_lim = self.lab_time_limit if self.lab_budget_mode == "time" else None
                w_lim = self.lab_workload if self.lab_budget_mode == "workload" else None

                sample_a = create_agent(self.lab_agent_a)
                sample_b = create_agent(self.lab_agent_b)
                a_name, b_name = sample_a.name, sample_b.name

                with self.lab_live_lock:
                    self.lab_games_scroll = 0
                    self.lab_live_games = []
                    for o_idx, rec in enumerate(openings):
                        for a_is_w in (True, False):
                            self.lab_live_games.append(
                                {
                                    "idx": len(self.lab_live_games),
                                    "opening_id": rec.record_id,
                                    "opening_fen": rec.fen,
                                    "a_is_white": a_is_w,
                                    "fen": rec.fen,
                                    "last_move": None,
                                    "plies": 0,
                                    "status": "waiting",
                                    "result": None,
                                }
                            )
                    self.lab_results["arena"] = {
                        "agent_a": a_name,
                        "agent_b": b_name,
                        "games": 0,
                        "total_games": total_games,
                        "wins": 0,
                        "draws": 0,
                        "losses": 0,
                        "score": 0.0,
                        "white_score": 0.0,
                        "black_score": 0.0,
                        "wilson_ci_95": [0.0, 0.0],
                        "bootstrap_ci_95": [0.0, 0.0],
                        "agent_a_avg_latency_ms": 0.0,
                        "agent_b_avg_latency_ms": 0.0,
                    }

                self.lab_message = f"Đang đấu {self.lab_agent_a} vs {self.lab_agent_b} ({total_games} ván)..."
                self._append_lab_log(
                    f"[UI Sandbox] Bắt đầu Arena trực tiếp: {self.lab_agent_a} vs {self.lab_agent_b} | {total_games} ván | mode={b_mode} | workers={self.lab_workers}"
                )

                pair_scores_map: dict[int, dict[bool, float]] = {}
                a_lats: list[float] = []
                b_lats: list[float] = []
                stats_acc = {"wins": 0, "draws": 0, "losses": 0, "games": 0, "w_pts": 0.0, "b_pts": 0.0}

                def _run_one_game(g_idx: int, rec: Any, a_is_w: bool, seed: int) -> dict[str, Any]:
                    ag_w = create_agent(self.lab_agent_a if a_is_w else self.lab_agent_b)
                    ag_b = create_agent(self.lab_agent_b if a_is_w else self.lab_agent_a)
                    last_push = [0.0]

                    def _on_move(fen: str, mv_uci: str, plies: int) -> None:
                        now = time.perf_counter()
                        if now - last_push[0] < 0.08:
                            return
                        last_push[0] = now
                        with self.lab_live_lock:
                            if g_idx < len(self.lab_live_games):
                                item = self.lab_live_games[g_idx]
                                item["fen"] = fen
                                item["last_move"] = mv_uci
                                item["plies"] = plies
                                item["status"] = "playing"

                    with self.lab_live_lock:
                        self.lab_live_games[g_idx]["status"] = "playing"

                    g = play_single_match(
                        white_agent=ag_w,
                        black_agent=ag_b,
                        initial_fen=rec.fen,
                        time_limit_sec=t_lim,
                        max_plies=400,
                        seed=seed,
                        white_workload=w_lim,
                        black_workload=w_lim,
                        on_move=_on_move,
                        stop_event=self.lab_stop_event,
                    )
                    g["opening_id"] = rec.record_id
                    g["budget_mode"] = b_mode
                    g["match_id"] = f"{a_name}_vs_{b_name}_{rec.record_id}_{'w' if a_is_w else 'b'}"
                    return g

                with ThreadPoolExecutor(max_workers=max(1, self.lab_workers)) as pool:
                    fut_map = {}
                    for o_idx, rec in enumerate(openings):
                        seed = 42 + (o_idx // 5)
                        for sub_i, a_is_w in enumerate((True, False)):
                            g_idx = o_idx * 2 + sub_i
                            fut = pool.submit(_run_one_game, g_idx, rec, a_is_w, seed)
                            fut_map[fut] = (g_idx, o_idx, a_is_w)

                    for fut in fut_map:
                        if self.lab_stop_event.is_set():
                            break
                        g_idx, o_idx, a_is_w = fut_map[fut]
                        g = fut.result()
                        s_a = float(g["white_score"] if a_is_w else g["black_score"])
                        append_match_jsonl(str(sandbox_root / "matches.jsonl"), g)
                        moves = [chess.Move.from_uci(u) for u in g["moves_uci"]]
                        pgn_txt = export_game_to_pgn(
                            moves=moves,
                            headers={
                                "Event": f"101_TTNTNC Arena ({b_mode})",
                                "Site": g["opening_id"],
                                "White": g["white"],
                                "Black": g["black"],
                                "Result": g["result"],
                            },
                            initial_fen=g["opening_fen"],
                        )
                        append_pgn_file(str(sandbox_root / "games.pgn"), pgn_txt)

                        with self.lab_live_lock:
                            item = self.lab_live_games[g_idx]
                            item["status"] = "done"
                            item["result"] = g["result"]
                            item["plies"] = int(g.get("plies", item["plies"]))
                            if g["moves_uci"]:
                                item["last_move"] = g["moves_uci"][-1]
                            try:
                                b_end = chess.Board(g["opening_fen"])
                                for u in g["moves_uci"]:
                                    b_end.push_uci(u)
                                item["fen"] = b_end.fen()
                            except Exception:
                                pass

                            stats_acc["games"] += 1
                            if s_a == 1.0:
                                stats_acc["wins"] += 1
                            elif s_a == 0.5:
                                stats_acc["draws"] += 1
                            else:
                                stats_acc["losses"] += 1
                            if a_is_w:
                                stats_acc["w_pts"] += s_a
                                a_lats.append(float(g["white_avg_latency_ms"]))
                                b_lats.append(float(g["black_avg_latency_ms"]))
                            else:
                                stats_acc["b_pts"] += s_a
                                a_lats.append(float(g["black_avg_latency_ms"]))
                                b_lats.append(float(g["white_avg_latency_ms"]))
                            pair_scores_map.setdefault(o_idx, {})[a_is_w] = s_a

                            done_g = stats_acc["games"]
                            w_cnt = stats_acc["wins"]
                            d_cnt = stats_acc["draws"]
                            l_cnt = stats_acc["losses"]
                            sc = compute_normalized_score(w_cnt, d_cnt, l_cnt)
                            w_ci = wilson_confidence_interval(sc, done_g)
                            completed_pairs = [
                                (ps[True] + ps[False]) / 2.0
                                for ps in pair_scores_map.values()
                                if True in ps and False in ps
                            ]
                            b_ci = (
                                paired_bootstrap_confidence_interval(completed_pairs, num_resamples=1000)
                                if completed_pairs
                                else (0.0, 0.0)
                            )
                            w_games = sum(1 for ps in pair_scores_map.values() if True in ps)
                            b_games = sum(1 for ps in pair_scores_map.values() if False in ps)
                            res_live = {
                                "agent_a": a_name,
                                "agent_b": b_name,
                                "games": done_g,
                                "total_games": total_games,
                                "wins": w_cnt,
                                "draws": d_cnt,
                                "losses": l_cnt,
                                "score": sc,
                                "white_score": stats_acc["w_pts"] / max(1, w_games),
                                "black_score": stats_acc["b_pts"] / max(1, b_games),
                                "wilson_ci_95": [w_ci[0], w_ci[1]],
                                "bootstrap_ci_95": [b_ci[0], b_ci[1]],
                                "agent_a_avg_latency_ms": sum(a_lats) / max(1, len(a_lats)),
                                "agent_b_avg_latency_ms": sum(b_lats) / max(1, len(b_lats)),
                            }
                            self.lab_results["arena"] = res_live
                            self.lab_progress = 0.05 + 0.95 * (done_g / float(max(1, total_games)))

                res = self.lab_results["arena"]
                (sandbox_root / "arena_matchup.json").write_text(
                    json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
                )
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
