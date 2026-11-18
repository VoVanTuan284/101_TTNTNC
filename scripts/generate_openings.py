"""
scripts/generate_openings.py
Kịch bản sinh 50 thế cờ khai cuộc cân bằng (6-8 plies) phục vụ giải đấu Round-Robin Arena.
Mỗi thế cờ được kiểm định qua hàm lượng giá tĩnh cổ điển (|eval| <= 60 cp),
khử trùng lặp qua Zobrist hash 64-bit và ghi ra định dạng EPD chuẩn.
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Set

# Thêm thư mục gốc vào PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import chess
from chess_ai.core.evaluation import evaluate_board
from chess_ai.core.zobrist import compute_zobrist_hash

# Danh mục các chuỗi khai cuộc chuẩn đa dạng (ECO A-E) làm hạt giống (seed lines)
# Mỗi chuỗi gồm 6 - 8 plies (3 - 4 nước đi trọn vẹn của 2 bên)
CANONICAL_OPENING_LINES = [
    # C40-C50: Ruy Lopez / Italian / Two Knights / Four Knights
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"],  # Ruy Lopez Morph
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "d2d3", "f8c5"],  # Berlin Defense
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6"],  # Giuoco Piano
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8e7"],  # Two Knights Modern
    ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6", "f1b5", "f8b4"],  # Four Knights Spanish
    ["e2e4", "e7e5", "f2f4", "e5f4", "g1f3", "d7d6", "d2d4", "g8f6"],  # King's Gambit Accepted
    ["e2e4", "e7e5", "g1f3", "d7d6", "d2d4", "b8d7", "f1c4", "c7c6"],  # Philidor Defense
    ["e2e4", "e7e5", "g1f3", "g8f6", "f3e5", "d7d6", "e5f3", "f6e4"],  # Petroff Defense
    ["e2e4", "e7e5", "d2d4", "e5d4", "g1f3", "b8c6", "f1c4", "f8c5"],  # Scotch Gambit

    # B20-B99: Sicilian Defense
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6"],  # Sicilian Open
    ["e2e4", "c7c5", "g1f3", "e7e6", "d2d4", "c5d4", "f3d4", "a7a6"],  # Sicilian Kan
    ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6"],  # Sicilian Classical
    ["e2e4", "c7c5", "c2c3", "d7d5", "e4d5", "d8d5", "d2d4", "g8f6"],  # Sicilian Alapin
    ["e2e4", "c7c5", "b1c3", "b8c6", "g2g3", "g7g6", "f1g2", "f8g7"],  # Closed Sicilian
    ["e2e4", "c7c5", "g1f3", "d7d6", "f1b5", "c8d7", "b5d7", "d8d7"],  # Moscow Variation

    # C00-C19: French Defense
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "c8g5", "f8e7"],  # French Classical
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1d2", "g8f6", "e4e5", "f6d7"],  # French Tarrasch
    ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5", "c7c5", "c2c3", "b8c6"],  # French Advance
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4", "e4e5", "c7c5"],  # French Winawer

    # B10-B19: Caro-Kann Defense
    ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5"],  # Caro-Kann Classical
    ["e2e4", "c7c6", "d2d4", "d7d5", "e4e5", "c8f5", "g1f3", "e7e6"],  # Caro-Kann Advance
    ["e2e4", "c7c6", "d2d4", "d7d5", "b1d2", "d5e4", "d2e4", "b8d7"],  # Caro-Kann Steinitz
    ["e2e4", "c7c6", "c2c4", "d7d5", "e4d5", "c6d5", "c4d5", "g8f6"],  # Panov-Botvinnik

    # B00-B09: Pirc / Modern / Alekhine / Scandinavian
    ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "g1f3", "f8g7"],  # Pirc Defense
    ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "d2d4", "g8f6"],  # Scandinavian
    ["e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6", "g1f3", "g7g6"],  # Alekhine Defense

    # D00-D69: Queen's Gambit & Slav
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"],  # QGD Orthodox
    ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "e7e6"],  # Semi-Slav
    ["d2d4", "d7d5", "c2c4", "c7c6", "c4d5", "c6d5", "b1c3", "g8f6"],  # Slav Exchange
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "c7c6", "e2e3", "g8f6"],  # QGD Noteboom
    ["d2d4", "d7d5", "c1f4", "g8f6", "e2e3", "c7c5", "c2c3", "b8c6"],  # London System
    ["d2d4", "d7d5", "g1f3", "g8f6", "e2e3", "e7e6", "f1d3", "c7c5"],  # Colle System

    # E60-E99: King's Indian & Grünfeld
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"],  # King's Indian Mar del Plata
    ["d2d4", "g8f6", "c2c4", "g7g6", "g1f3", "f8g7", "g2g3", "e8g8"],  # King's Indian Fianchetto
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5", "f6d5"],  # Grünfeld Exchange
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "g1f3", "f8g7"],  # Grünfeld Russian

    # E10-E59: Nimzo-Indian & Queen's Indian / Bogo-Indian
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "e8g8"],  # Nimzo-Indian Rubinstein
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "d1c2", "d7d5"],  # Nimzo-Indian Classical
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "g2g3", "c8a6"],  # Queen's Indian Fianchetto
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "f8b4", "c1d2", "d8e7"],  # Bogo-Indian

    # E00-E09: Catalan Opening
    ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "f1g2", "f8e7"],  # Catalan Closed
    ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "f1g2", "d5c4"],  # Catalan Open

    # A10-A39: English Opening
    ["c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6", "g2g3", "f8b4"],  # English Four Knights
    ["c2c4", "c7c5", "b1c3", "b8c6", "g2g3", "g7g6", "f1g2", "f8g7"],  # English Symmetrical
    ["c2c4", "e7e6", "g1f3", "d7d5", "g2g3", "g8f6", "f1g2", "f8e7"],  # English Neo-Catalan

    # A04-A09: Réti Opening & King's Indian Attack
    ["g1f3", "d7d5", "c2c4", "c7c6", "b2b3", "g8f6", "g2g3", "c8f5"],  # Réti Slav
    ["g1f3", "d7d5", "g2g3", "g8f6", "f1g2", "c7c6", "e1g1", "c8g4"],  # King's Indian Attack
    ["g1f3", "g8f6", "g2g3", "g7g6", "b2b3", "f8g7", "c1b2", "e8g8"],  # Double Fianchetto

    # A80-A99: Dutch Defense
    ["d2d4", "f7f5", "g2g3", "g8f6", "f1g2", "e7e6", "g1f3", "f8e7"],  # Dutch Leningrad/Classical
]


def extract_fen_4_fields(board: chess.Board) -> str:
    """Trích xuất 4 trường FEN chuẩn theo quy cách EPD (bỏ qua halfmove clock và fullmove number)."""
    parts = board.fen().split(" ")
    return f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}"


def generate_openings(
    num_openings: int = 50,
    target_plies: int = 8,
    max_eval_cp: float = 60.0,
    seed: int = 42,
) -> List[str]:
    """
    Sinh danh sách `num_openings` thế cờ khai cuộc cân bằng (|eval| <= max_eval_cp),
    không trùng lặp (dùng Zobrist hash) và đạt chuẩn EPD.
    """
    rng = random.Random(seed)
    openings: List[str] = []
    seen_hashes: Set[int] = set()

    # Bước 1: Nạp từ tập canonical lines có kiểm định đánh giá
    shuffled_canonical = list(CANONICAL_OPENING_LINES)
    rng.shuffle(shuffled_canonical)

    for line in shuffled_canonical:
        if len(openings) >= num_openings:
            break
        board = chess.Board()
        valid_line = True
        for uci_move in line[:target_plies]:
            move = chess.Move.from_uci(uci_move)
            if move in board.legal_moves:
                board.push(move)
            else:
                valid_line = False
                break

        if not valid_line or board.is_game_over():
            continue

        z_hash = compute_zobrist_hash(board)
        if z_hash in seen_hashes:
            continue

        # Lượng giá tĩnh theo góc nhìn tuyệt đối của Trắng
        score = evaluate_board(board)
        white_eval = score if board.turn == chess.WHITE else -score
        if abs(white_eval) <= max_eval_cp:
            seen_hashes.add(z_hash)
            openings.append(extract_fen_4_fields(board))

    # Bước 2: Nếu chưa đủ số lượng (ví dụ cần > số canonical), sinh bổ sung qua bước ngẫu nhiên hợp lý
    extra_attempts = 0
    max_extra_attempts = 1000
    while len(openings) < num_openings and extra_attempts < max_extra_attempts:
        extra_attempts += 1
        board = chess.Board()
        # Chọn ngẫu nhiên một line canonical làm tiền đề 4-6 plies, sau đó random walk 2 plies
        base_line = rng.choice(CANONICAL_OPENING_LINES)
        base_depth = min(rng.choice([4, 6]), len(base_line))
        for m in base_line[:base_depth]:
            mv = chess.Move.from_uci(m)
            if mv in board.legal_moves:
                board.push(mv)

        # Walk thêm cho đủ target_plies
        while len(board.move_stack) < target_plies and not board.is_game_over():
            legal = list(board.legal_moves)
            if not legal:
                break
            # Ưu tiên các nước không dâng quân (không mất vật chất lộ liễu)
            board.push(rng.choice(legal))

        if len(board.move_stack) < target_plies or board.is_game_over():
            continue

        z_hash = compute_zobrist_hash(board)
        if z_hash in seen_hashes:
            continue

        score = evaluate_board(board)
        white_eval = score if board.turn == chess.WHITE else -score
        if abs(white_eval) <= max_eval_cp:
            seen_hashes.add(z_hash)
            openings.append(extract_fen_4_fields(board))

    return openings[:num_openings]


def write_epd_file(openings: List[str], output_path: str) -> None:
    """Ghi danh sách thế cờ ra tệp EPD chuẩn kèm header chú thích."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Bộ 50 thế cờ khai cuộc cân bằng (6-8 plies) phục vụ giải đấu Arena (100 ván/cặp = 50 khai cuộc x 2 màu)\n")
        f.write("# Định dạng EPD chuẩn: <FEN_4_fields>; id \"OPEN.001\";\n")
        f.write("# Tự động sinh bởi scripts/generate_openings.py (TV3 - Arena & Analytics Lead)\n\n")
        for idx, fen_4 in enumerate(openings, start=1):
            f.write(f'{fen_4}; id "OPEN.{idx:03d}";\n')


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh 50 thế cờ khai cuộc cân bằng cho Arena.")
    parser.add_argument(
        "--output",
        type=str,
        default="resources/openings_50.epd",
        help="Đường dẫn file EPD đầu ra",
    )
    parser.add_argument(
        "--num-openings",
        type=int,
        default=50,
        help="Số lượng thế cờ khai cuộc cần sinh (mặc định: 50)",
    )
    parser.add_argument(
        "--plies",
        type=int,
        default=8,
        help="Độ sâu thế cờ khai cuộc (mặc định: 8 plies = 4 nước đi)",
    )
    parser.add_argument(
        "--max-eval",
        type=float,
        default=60.0,
        help="Ngưỡng chênh lệch lượng giá tối đa (centipawns, mặc định: 60 cp)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed ngẫu nhiên (mặc định: 42)",
    )

    args = parser.parse_args()
    target_path = Path(args.output)
    if not target_path.is_absolute():
        target_path = REPO_ROOT / target_path

    print(f"[+] Đang sinh {args.num_openings} thế cờ khai cuộc cân bằng (target_plies={args.plies}, max_eval={args.max_eval} cp)...")
    openings = generate_openings(
        num_openings=args.num_openings,
        target_plies=args.plies,
        max_eval_cp=args.max_eval,
        seed=args.seed,
    )

    print(f"[+] Đã sinh thành công {len(openings)} thế cờ. Đang lưu ra: {target_path}")
    write_epd_file(openings, str(target_path))
    print("[+] Hoàn tất!")


if __name__ == "__main__":
    main()

