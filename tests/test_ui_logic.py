"""
tests/test_ui_logic.py
Kiểm thử logic điều khiển của giao diện Pygame (không mở cửa sổ đồ họa):
- san(), termination_reason() trên ChessEnv
- _play_action, _undo, _restore_index, _find_action, _handle_panel_click, chế độ AI vs AI
"""

import chess

from chess_ai.ui.board_renderer import render_board_image
from chess_ai.ui.desktop_pygame import ChessPygameApp


def test_ui_play_undo_replay_and_status() -> None:
    app = ChessPygameApp(agent_id="level1", mode="human_vs_ai")
    try:
        # Đánh Fool's mate thủ công để kiểm tra san() và termination_reason()
        for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
            app._play_action(chess.Move.from_uci(uci))

        assert app.env.is_terminal()
        assert "Đen thắng" in app._status_text()
        assert "checkmate" in app._status_text()
        assert [r.san for r in app.records] == ["f3", "e5", "g4", "Qh4#"]
        # Kiểm tra nhận diện ô Vua bị chiếu (e1) và ô quân chiếu (h4)
        king_sq, checkers = app._check_squares()
        assert king_sq == chess.E1
        assert checkers == {chess.H4}

        # Kiểm tra Replay về nước 0 và quay lại cuối
        app._restore_index(0)
        assert app.replay_index == 0
        assert app._active_ai_agent() is None  # Không gọi AI khi đang xem Replay

        app._restore_index(4)
        assert app.replay_index == 4

        # Kiểm tra Undo
        app._undo()
        assert len(app.records) < 4
    finally:
        app.executor.shutdown(wait=False, cancel_futures=True)


def test_ui_panel_click_and_mode_toggle() -> None:
    app = ChessPygameApp(agent_id="level3", mode="human_vs_ai")
    try:
        # Bấm nút Chế độ [M] ở hàng 1, cột 1 (0-indexed: row=1, col=1)
        bx = app.BTN_START_X + 1 * (app.BTN_W + app.BTN_GAP_X) + 10
        by = app.BTN_START_Y + 1 * (app.BTN_H + app.BTN_GAP_Y) + 10
        app._handle_panel_click(bx, by)
        assert app.mode == "ai_vs_ai"
        assert app._active_ai_agent() is app.white_agent

        # Kiểm tra nút Tạm dừng / Tiếp tục (nút giữa "||" / "▶" trong hàng Replay) ở chế độ AI vs AI
        import pygame
        app._handle_click((952 + 2 * 64 + 20, 685), pygame)
        assert app.paused is True
        assert app._active_ai_agent() is None
        app._handle_click((952 + 2 * 64 + 20, 685), pygame)
        assert app.paused is False
        assert app._active_ai_agent() is app.white_agent

        # Kiểm tra _handle_click với tham số (position, pygame) như trong run()
        import pygame
        app._handle_click((300, 100), pygame)

        # Kiểm tra thanh trượt (Time Limit & Workload) trên panel trái
        app._handle_click((240, 315), pygame)  # Kéo thanh trượt Time tới mức cuối
        assert app.time_limit == app.TIME_STEPS[-1]
        assert app.max_workload is None

        app._handle_click((180, 360), pygame)  # Click thanh trượt Workload
        assert app.time_limit is None
        assert app.max_workload in app.WORKLOAD_STEPS

        # Kiểm tra Modal Ván mới: chọn hai bên bằng Hộp thoại trượt (Scrollable Dropdown)
        app._new_game()
        assert app.modal_open is True
        # Click mở hộp thoại trượt bên TRẮNG (mx=340, my=115, box_x=mx+24, box_y=my+88)
        app._handle_click((340 + 50, 115 + 100), pygame)
        assert app.modal_dropdown_open == "white"
        # Chọn mục thứ 3 trong danh sách thả xuống ("level2")
        app.modal_dropdown_scroll = 0
        app._handle_click((340 + 50, 115 + 132 + 2 * 34 + 10), pygame)
        assert app.modal_dropdown_open is None
        assert app.modal_white_side == "level2"

        # Click mở hộp thoại trượt bên ĐEN (box_x=mx+316, box_y=my+88)
        app._handle_click((340 + 350, 115 + 100), pygame)
        assert app.modal_dropdown_open == "black"
        # Chọn mục đầu tiên trong danh sách thả xuống ("human")
        app.modal_dropdown_scroll = 0
        app._handle_click((340 + 350, 115 + 132 + 0 * 34 + 10), pygame)
        assert app.modal_dropdown_open is None
        assert app.modal_black_side == "human"

        # Bấm Bắt đầu ván đấu (mx=340..540, my+426..470)
        app._handle_click((340 + 400, 115 + 445), pygame)
        assert app.modal_open is False
        assert app.game_mode == "human_ai"
        assert app.player_color == "black"
        assert app.white_agent is not None

        # Kiểm tra hộp thoại Kết thúc ván đấu khi đầu hàng và nút Xem bàn cờ
        app._resign()
        assert app._is_game_over() is True
        assert app.game_over_dismissed is False
        pygame.init()
        surface = pygame.Surface(app.WINDOW_SIZE)
        app._draw(surface, pygame)  # Đảm bảo vẽ banner kết thúc ván không lỗi
        # Click nút "Xem bàn cờ" trên hộp thoại kết thúc ván
        board_x = app.LEFT_WIDTH + (app.ARENA_WIDTH - app.BOARD_SIZE) // 2
        board_y = app.TAB_HEIGHT + 48
        cx = board_x + app.BOARD_SIZE // 2
        cy = board_y + app.BOARD_SIZE // 2
        app._handle_click((cx - 210 + 50, cy - 125 + 195), pygame)
        assert app.game_over_dismissed is True

        # Kiểm tra chuyển sang Tab 2 (Nghiên cứu & Thực nghiệm đã mở khóa) và vẽ không lỗi
        app._handle_click((450, 20), pygame)
        assert app.active_tab == "lab"
        app._draw(surface, pygame)
        # Quay lại Tab 1 (Đấu trường) và kiểm tra thêm khóa bảo vệ nếu bật lab_tab_locked
        app._handle_click((120, 20), pygame)
        assert app.active_tab == "play"
        app.lab_tab_locked = True
        app._handle_click((450, 20), pygame)
        assert app.active_tab == "play"
        assert "tạm khóa" in app.message
        app.lab_tab_locked = False
        app._draw(surface, pygame)
    finally:
        app.executor.shutdown(wait=False, cancel_futures=True)


def test_board_renderer_headless() -> None:
    board = chess.Board()
    img = render_board_image(
        board,
        move=chess.Move.from_uci("e2e4"),
        highlight_squares=(chess.D4, chess.D5, chess.E4, chess.E5),
        board_size=256,
    )
    assert img.shape == (256, 256, 3)
    assert img.dtype.name == "uint8"
