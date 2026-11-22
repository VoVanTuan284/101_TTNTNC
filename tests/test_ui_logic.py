"""
tests/test_ui_logic.py
Kiểm thử logic điều khiển của giao diện Pygame (không mở cửa sổ đồ họa):
- san(), termination_reason() trên ChessEnv
- _play_action, _undo, _restore_index, _find_action, _handle_panel_click, chế độ AI vs AI
"""

import chess

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

        # Kiểm tra _handle_click với tham số (position, pygame) như trong run()
        import pygame
        app._handle_click((300, 100), pygame)
    finally:
        app.executor.shutdown(wait=False, cancel_futures=True)
