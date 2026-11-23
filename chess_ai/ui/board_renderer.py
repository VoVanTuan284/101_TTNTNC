"""
chess_ai/ui/board_renderer.py
Tiện ích tái sử dụng bảng màu và bộ quân cờ SVG/PNG của Pygame Desktop UI
để kết xuất trạng thái `chess.Board` thành ảnh RGB `np.ndarray` dùng trong Notebook và Báo cáo.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import chess
import numpy as np


_LIGHT = (238, 238, 210)
_DARK = (118, 150, 86)
_MOVE_HIGHLIGHT = (186, 203, 68)
_BAD_HIGHLIGHT = (220, 85, 85)
_SQUARE_ACCENT = (246, 210, 82)
_ARROW_GOOD = (34, 160, 85)
_ARROW_CANDIDATE = (59, 130, 246)
_ARROW_BAD = (210, 50, 50)
_COORD_LIGHT = (118, 150, 86)
_COORD_DARK = (238, 238, 210)

_PIECE_FILENAMES: Dict[str, str] = {
    "K": "wK", "Q": "wQ", "R": "wR", "B": "wB", "N": "wN", "P": "wP",
    "k": "bK", "q": "bQ", "r": "bR", "b": "bB", "n": "bN", "p": "bP",
}

_PIECE_CACHE: Dict[Tuple[str, int], Dict[str, object]] = {}


def _ensure_pygame_headless():
    if "SDL_VIDEODRIVER" not in os.environ:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()
    return pygame


def _load_pieces(pygame, piece_set: str, sq_size: int) -> Dict[str, object]:
    cache_key = (piece_set, sq_size)
    if cache_key in _PIECE_CACHE:
        return _PIECE_CACHE[cache_key]

    piece_dir = Path(__file__).resolve().parent / "assets" / "piece" / piece_set
    target_px = max(16, int(sq_size * 0.88))
    images: Dict[str, object] = {}
    for symbol, fname in _PIECE_FILENAMES.items():
        for suffix in (".png", ".svg"):
            p = piece_dir / f"{fname}{suffix}"
            if p.exists():
                try:
                    raw = pygame.image.load(str(p))
                    images[symbol] = pygame.transform.smoothscale(raw, (target_px, target_px))
                    break
                except Exception:
                    continue
    _PIECE_CACHE[cache_key] = images
    return images


def _square_center(sq: int, sq_size: int, flipped: bool) -> Tuple[int, int]:
    file_idx = chess.square_file(sq)
    rank_idx = chess.square_rank(sq)
    col = (7 - file_idx) if flipped else file_idx
    row = rank_idx if flipped else (7 - rank_idx)
    return int((col + 0.5) * sq_size), int((row + 0.5) * sq_size)


def _draw_arrow(
    pygame,
    surface,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    width: int,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-3:
        return
    ux, uy = dx / length, dy / length
    head_len = max(10.0, width * 2.6)
    shaft_end = (int(end[0] - ux * head_len * 0.65), int(end[1] - uy * head_len * 0.65))
    pygame.draw.line(surface, color, start, shaft_end, width)

    px, py = -uy, ux
    tip = end
    left = (
        int(end[0] - ux * head_len + px * head_len * 0.55),
        int(end[1] - uy * head_len + py * head_len * 0.55),
    )
    right = (
        int(end[0] - ux * head_len - px * head_len * 0.55),
        int(end[1] - uy * head_len - py * head_len * 0.55),
    )
    pygame.draw.polygon(surface, color, [tip, left, right])


def render_board_image(
    board: chess.Board,
    move: Optional[chess.Move] = None,
    moves: Optional[Iterable[chess.Move]] = None,
    bad_move: Optional[chess.Move] = None,
    highlight_squares: Optional[Iterable[int]] = None,
    board_size: int = 384,
    piece_set: str = "cburnett",
    flipped: bool = False,
) -> np.ndarray:
    """
    Kết xuất bàn cờ `chess.Board` thành ảnh RGB `np.ndarray` shape `(H, W, 3)`
    sử dụng lại bảng màu và bộ quân cờ của `ChessPygameApp`.
    """
    pygame = _ensure_pygame_headless()
    sq_size = max(24, board_size // 8)
    actual_size = sq_size * 8
    surface = pygame.Surface((actual_size, actual_size))

    hl_set = set(highlight_squares) if highlight_squares is not None else set()
    candidate_moves = list(moves) if moves is not None else []
    move_sqs = {move.from_square, move.to_square} if move is not None else set()
    bad_sqs = {bad_move.from_square, bad_move.to_square} if bad_move is not None else set()

    font = pygame.font.SysFont("arial", max(10, sq_size // 5), bold=True)
    pieces = _load_pieces(pygame, piece_set, sq_size)

    for row in range(8):
        for col in range(8):
            file_idx = (7 - col) if flipped else col
            rank_idx = row if flipped else (7 - row)
            sq = chess.square(file_idx, rank_idx)
            is_light = (rank_idx + file_idx) % 2 == 1

            color = _LIGHT if is_light else _DARK
            if sq in hl_set:
                color = _SQUARE_ACCENT
            if sq in move_sqs:
                color = _MOVE_HIGHLIGHT
            if sq in bad_sqs:
                color = _BAD_HIGHLIGHT

            rect = pygame.Rect(col * sq_size, row * sq_size, sq_size, sq_size)
            pygame.draw.rect(surface, color, rect)

            # Tọa độ hàng/cột ở mép bàn cờ
            text_color = _COORD_LIGHT if is_light else _COORD_DARK
            if col == 0:
                lbl = font.render(str(rank_idx + 1), True, text_color)
                surface.blit(lbl, (rect.x + 2, rect.y + 2))
            if row == 7:
                lbl = font.render(chess.FILE_NAMES[file_idx], True, text_color)
                surface.blit(lbl, (rect.right - lbl.get_width() - 2, rect.bottom - lbl.get_height() - 1))

            piece = board.piece_at(sq)
            if piece is not None:
                sprite = pieces.get(piece.symbol())
                if sprite is not None:
                    px = rect.x + (sq_size - sprite.get_width()) // 2
                    py = rect.y + (sq_size - sprite.get_height()) // 2
                    surface.blit(sprite, (px, py))

    # Vẽ mũi tên nước đi (ứng viên xanh dương -> nước bị loại đỏ -> nước được chọn xanh lá)
    arrow_w = max(3, sq_size // 10)
    cand_w = max(2, sq_size // 14)
    for cm in candidate_moves:
        if move is not None and cm == move:
            continue
        _draw_arrow(
            pygame,
            surface,
            _square_center(cm.from_square, sq_size, flipped),
            _square_center(cm.to_square, sq_size, flipped),
            _ARROW_CANDIDATE,
            cand_w,
        )
    if bad_move is not None:
        _draw_arrow(
            pygame,
            surface,
            _square_center(bad_move.from_square, sq_size, flipped),
            _square_center(bad_move.to_square, sq_size, flipped),
            _ARROW_BAD,
            arrow_w,
        )
    if move is not None:
        _draw_arrow(
            pygame,
            surface,
            _square_center(move.from_square, sq_size, flipped),
            _square_center(move.to_square, sq_size, flipped),
            _ARROW_GOOD,
            arrow_w,
        )

    # pygame.surfarray.array3d trả về (W, H, 3) -> chuyển vị sang (H, W, 3) cho matplotlib
    arr = pygame.surfarray.array3d(surface)
    return np.transpose(arr, (1, 0, 2)).copy()
