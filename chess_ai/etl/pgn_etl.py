"""
chess_ai/etl/pgn_etl.py
Đường ống ETL đọc tệp Lichess Elite Database, lọc nhanh ở mức Header (ELO >= 2100,
TimeControl >= 180s), lấy mẫu cách quãng k in {3, 4}, khử trùng lặp thế cờ bằng Zobrist 64-bit,
chia tách Game-level split tất định qua MD5(Game ID) và ghi ra mảng nhị phân np.memmap.
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import zipfile

import chess
import chess.pgn
import numpy as np

from chess_ai.config import load_yaml_config
from chess_ai.core.board_tensor import board_to_uint8_planes, move_to_action_index
from chess_ai.core.evaluation import evaluate_board_tanh
from chess_ai.core.zobrist import compute_zobrist_hash


_RESULT_MAP: Dict[str, float] = {
    "1-0": 1.0,
    "0-1": -1.0,
    "1/2-1/2": 0.0,
}

_VALID_TERMINATIONS = {"Normal", "Time forfeit", ""}


def resolve_input_pgn_path(input_path: Optional[str] = None) -> Path:
    """Tìm đường dẫn tệp dữ liệu thô (.pgn, .pgn.zst hoặc .zip) trong data/raw/."""
    if input_path:
        p = Path(input_path)
        if p.exists():
            return p

    raw_dir = Path("data/raw")
    for pattern in ("*.pgn", "*.pgn.zst", "*.zst", "*.zip"):
        matches = sorted(raw_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu PGN tại '{input_path}' hoặc trong data/raw/.")


@contextmanager
def open_pgn_text_stream(filepath: Path) -> Iterator[io.TextIOBase]:
    """Mở luồng văn bản UTF-8 hỗ trợ cả .pgn, .pgn.zst và .zip."""
    suffix = filepath.suffix.lower()
    if suffix == ".pgn":
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            yield f
    elif suffix == ".zst":
        import zstandard as zstd

        with open(filepath, "rb") as raw_f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(raw_f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                yield text_stream
    elif suffix == ".zip":
        with zipfile.ZipFile(filepath, "r") as zf:
            pgn_names = [n for n in zf.namelist() if n.lower().endswith(".pgn")]
            if not pgn_names:
                raise ValueError(f"Không tìm thấy tệp .pgn bên trong {filepath}")
            with zf.open(pgn_names[0], "r") as raw_f:
                text_stream = io.TextIOWrapper(raw_f, encoding="utf-8", errors="replace")
                yield text_stream
    else:
        raise ValueError(f"Định dạng tệp không hỗ trợ: {filepath}")


def parse_estimated_time_sec(time_control: str) -> Optional[int]:
    """Tính T_est = base + 40 * increment (giây) từ chuỗi TimeControl PGN."""
    if not time_control or time_control in ("-", "?"):
        return None
    parts = time_control.split("+")
    try:
        base = int(parts[0])
        inc = int(parts[1]) if len(parts) > 1 else 0
        return base + 40 * inc
    except ValueError:
        return None


def deterministic_split(game_id: str, train_ratio: float = 0.80, val_ratio: float = 0.10) -> str:
    """Phân chia cấp độ ván đấu tất định bằng MD5(Game ID) % 100."""
    bucket = int(hashlib.md5(game_id.encode("utf-8")).hexdigest(), 16) % 100
    train_cut = int(round(train_ratio * 100))
    val_cut = train_cut + int(round(val_ratio * 100))
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "test"


def _check_headers(headers: chess.pgn.Headers, min_elo: int, min_time_sec: int) -> Tuple[bool, str]:
    """Kiểm tra nhanh 3 tiêu chí ở mức Header: Result/Termination, ELO và TimeControl."""
    result = headers.get("Result", "")
    if result not in _RESULT_MAP:
        return False, "invalid_result"

    term = headers.get("Termination", "")
    if term not in _VALID_TERMINATIONS:
        return False, "invalid_termination"

    try:
        w_elo = int(headers.get("WhiteElo", "0"))
        b_elo = int(headers.get("BlackElo", "0"))
    except ValueError:
        return False, "invalid_elo"
    if w_elo < min_elo or b_elo < min_elo:
        return False, "low_elo"

    t_est = parse_estimated_time_sec(headers.get("TimeControl", ""))
    if t_est is None or t_est < min_time_sec:
        return False, "low_time_control"

    return True, "ok"


def inspect_raw_pgn(input_path: Optional[str] = None, max_games: int = 10000) -> Dict[str, Any]:
    """
    Phân tích nhanh dữ liệu thô ở mức Header (EDA) phục vụ notebook analysis.ipynb.
    """
    path = resolve_input_pgn_path(input_path)
    cfg = load_yaml_config("default_config.yaml")["data_etl"]
    min_elo = int(cfg.get("min_elo", 2100))
    min_time_sec = int(cfg.get("min_base_time_sec", 180))

    scanned = 0
    passed_header = 0
    rejected_reasons: Dict[str, int] = {}
    results_count: Dict[str, int] = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "other": 0}
    white_elos: List[int] = []
    black_elos: List[int] = []
    est_times: List[int] = []

    with open_pgn_text_stream(path) as stream:
        while scanned < max_games:
            headers = chess.pgn.read_headers(stream)
            if headers is None:
                break
            scanned += 1
            res = headers.get("Result", "")
            results_count[res if res in ("1-0", "0-1", "1/2-1/2") else "other"] += 1

            try:
                white_elos.append(int(headers.get("WhiteElo", "0")))
                black_elos.append(int(headers.get("BlackElo", "0")))
            except ValueError:
                pass
            t_est = parse_estimated_time_sec(headers.get("TimeControl", ""))
            if t_est is not None:
                est_times.append(t_est)

            ok, reason = _check_headers(headers, min_elo, min_time_sec)
            if ok:
                passed_header += 1
            else:
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

    return {
        "file_path": str(path),
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "games_inspected": scanned,
        "header_passed": passed_header,
        "header_pass_rate": round(passed_header / scanned, 4) if scanned else 0.0,
        "rejected_reasons": rejected_reasons,
        "results_distribution": results_count,
        "white_elo_mean": round(float(np.mean(white_elos)), 1) if white_elos else 0.0,
        "black_elo_mean": round(float(np.mean(black_elos)), 1) if black_elos else 0.0,
        "white_elo_min": int(min(white_elos)) if white_elos else 0,
        "white_elo_max": int(max(white_elos)) if white_elos else 0,
        "est_time_median_sec": float(np.median(est_times)) if est_times else 0.0,
    }


class _SplitMemmapWriter:
    """Bộ đệm ghi nhị phân tuần tự ra .dat (RAM tiêu thụ cực thấp) tương thích np.memmap."""

    def __init__(self, out_dir: Path, split: str, flush_every: int = 4096) -> None:
        self.split = split
        self.flush_every = flush_every
        self.count = 0
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._values: List[float] = []

        self.f_states = open(out_dir / f"{split}_states.dat", "wb")
        self.f_actions = open(out_dir / f"{split}_actions.dat", "wb")
        self.f_values = open(out_dir / f"{split}_values.dat", "wb")

    def append(self, state: np.ndarray, action_idx: int, value: float) -> None:
        self._states.append(state)
        self._actions.append(action_idx)
        self._values.append(value)
        self.count += 1
        if len(self._actions) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._actions:
            return
        self.f_states.write(np.asarray(self._states, dtype=np.uint8).tobytes())
        self.f_actions.write(np.asarray(self._actions, dtype=np.int16).tobytes())
        self.f_values.write(np.asarray(self._values, dtype=np.float32).tobytes())
        self._states.clear()
        self._actions.clear()
        self._values.clear()

    def close(self) -> None:
        self.flush()
        self.f_states.close()
        self.f_actions.close()
        self.f_values.close()


def run_etl_pipeline(
    config_path: str = "configs/default_config.yaml",
    input_path: Optional[str] = None,
    output_dir: str = "data/processed",
    max_games: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Chạy toàn bộ quy trình ETL ngoại tuyến và xuất dataset_metadata.json."""
    t0 = time.perf_counter()
    cfg = load_yaml_config(config_path)
    etl_cfg = cfg["data_etl"]

    min_elo = int(etl_cfg.get("min_elo", 2100))
    min_time_sec = int(etl_cfg.get("min_base_time_sec", 180))
    min_plies = int(etl_cfg.get("min_plies", 30))
    early_draw_ply = int(etl_cfg.get("early_draw_ply_cutoff", 60))
    skip_opening_plies = int(etl_cfg.get("skip_opening_plies", 12))
    k_min = int(etl_cfg.get("subsample_k_min", 3))
    k_max = int(etl_cfg.get("subsample_k_max", 4))
    dedup_zobrist = bool(etl_cfg.get("deduplicate_zobrist", True))
    drop_underpromo = bool(etl_cfg.get("drop_underpromotions", True))
    # Pha trộn nhãn value: z_mix = mix * kết_cục_ván + (1 - mix) * tanh(static_eval / scale)
    # mix = 1.0 giữ nguyên nhãn thuần theo kết cục ván (hành vi cũ). mix < 1.0 giảm nhiễu nhãn.
    value_label_mix = float(etl_cfg.get("value_label_mix", 1.0))
    value_eval_scale = float(etl_cfg.get("value_eval_scale", 600.0))
    target_games = max_games if max_games is not None else int(etl_cfg.get("target_games", 50000))
    target_samples = max_samples if max_samples is not None else int(etl_cfg.get("target_samples", 1000000))

    ratios = etl_cfg.get("split_ratios", {"train": 0.80, "val": 0.10, "test": 0.10})
    train_ratio = float(ratios.get("train", 0.80))
    val_ratio = float(ratios.get("val", 0.10))

    rng = random.Random(int(cfg.get("random_seed", 42)))
    src_path = resolve_input_pgn_path(input_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    writers = {
        "train": _SplitMemmapWriter(out_path, "train"),
        "val": _SplitMemmapWriter(out_path, "val"),
        "test": _SplitMemmapWriter(out_path, "test"),
    }

    seen_zobrist: Set[int] = set()
    games_scanned = 0
    valid_games = 0
    total_samples = 0
    zobrist_dups_skipped = 0
    underpromo_skipped = 0
    rejected_reasons: Dict[str, int] = {}
    split_games: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    result_counts: Dict[str, int] = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

    try:
        with open_pgn_text_stream(src_path) as stream:
            while valid_games < target_games and total_samples < target_samples:
                game = chess.pgn.read_game(stream)
                if game is None:
                    break
                games_scanned += 1
                headers = game.headers

                ok, reason = _check_headers(headers, min_elo, min_time_sec)
                if not ok:
                    rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
                    continue

                moves = list(game.mainline_moves())
                num_plies = len(moves)
                if num_plies < min_plies:
                    rejected_reasons["short_game"] = rejected_reasons.get("short_game", 0) + 1
                    continue

                res_str = headers["Result"]
                abs_val = _RESULT_MAP[res_str]

                # Lọc hòa sớm trước 60 plies khi còn nhiều quân (> 24 quân)
                board = game.board()
                if res_str == "1/2-1/2" and num_plies < early_draw_ply:
                    temp_board = game.board()
                    for mv in moves:
                        temp_board.push(mv)
                    if temp_board.occupied.bit_count() > 24:
                        rejected_reasons["early_draw"] = rejected_reasons.get("early_draw", 0) + 1
                        continue

                game_id = headers.get("LichessURL") or headers.get("Site")
                if not game_id or game_id in ("?", "-"):
                    game_id = (
                        f"{headers.get('White', '')}_{headers.get('Black', '')}_"
                        f"{headers.get('UTCDate', '')}_{headers.get('UTCTime', '')}_{games_scanned}"
                    )
                split = deterministic_split(game_id, train_ratio=train_ratio, val_ratio=val_ratio)
                writer = writers[split]

                valid_games += 1
                split_games[split] += 1
                result_counts[res_str] += 1

                next_sample_ply = skip_opening_plies + rng.randint(0, k_max - k_min)
                for ply_idx, mv in enumerate(moves):
                    if total_samples >= target_samples:
                        break

                    if ply_idx >= next_sample_ply:
                        if drop_underpromo and mv.promotion not in (None, chess.QUEEN):
                            underpromo_skipped += 1
                        else:
                            z_key = compute_zobrist_hash(board)
                            if dedup_zobrist and z_key in seen_zobrist:
                                zobrist_dups_skipped += 1
                            else:
                                if dedup_zobrist:
                                    seen_zobrist.add(z_key)
                                state = board_to_uint8_planes(board)
                                action_idx = move_to_action_index(mv, board.turn)
                                rel_val = abs_val if board.turn == chess.WHITE else -abs_val
                                if value_label_mix < 1.0:
                                    static_val = evaluate_board_tanh(board, scale=value_eval_scale)
                                    rel_val = value_label_mix * rel_val + (1.0 - value_label_mix) * static_val
                                writer.append(state, action_idx, float(rel_val))
                                total_samples += 1

                        next_sample_ply = ply_idx + rng.randint(k_min, k_max)

                    board.push(mv)
    finally:
        for w in writers.values():
            w.close()

    elapsed_sec = time.perf_counter() - t0
    metadata: Dict[str, Any] = {
        "source_file": str(src_path),
        "elapsed_sec": round(elapsed_sec, 2),
        "games_scanned": games_scanned,
        "valid_games": valid_games,
        "rejected_reasons": rejected_reasons,
        "total_samples": total_samples,
        "zobrist_duplicates_skipped": zobrist_dups_skipped,
        "underpromotions_skipped": underpromo_skipped,
        "result_distribution": result_counts,
        "value_label_mix": value_label_mix,
        "value_eval_scale": value_eval_scale,
        "splits": {
            split: {
                "games": split_games[split],
                "samples": writers[split].count,
            }
            for split in ("train", "val", "test")
        },
        "tensor_shape": [18, 8, 8],
        "dtypes": {"states": "uint8", "actions": "int16", "values": "float32"},
    }

    with open(out_path / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy ETL Lichess Elite PGN -> np.memmap.")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Đường dẫn cấu hình YAML.")
    parser.add_argument("--input_path", default=None, help="Đường dẫn tệp .pgn/.zip/.pgn.zst trong data/raw/.")
    parser.add_argument("--output_dir", default="data/processed", help="Thư mục lưu tệp .dat và metadata.")
    parser.add_argument("--max-games", type=int, default=None, help="Giới hạn số ván hợp lệ trích xuất.")
    parser.add_argument("--max-samples", type=int, default=None, help="Giới hạn số mẫu tối đa.")
    args = parser.parse_args()

    meta = run_etl_pipeline(
        config_path=args.config,
        input_path=args.input_path,
        output_dir=args.output_dir,
        max_games=args.max_games,
        max_samples=args.max_samples,
    )
    print(f"ETL hoàn tất trong {meta['elapsed_sec']}s -> {args.output_dir}")
    print(
        f"Đã quét: {meta['games_scanned']} ván | Hợp lệ: {meta['valid_games']} ván | "
        f"Tổng mẫu: {meta['total_samples']} | Trùng Zobrist đã loại: {meta['zobrist_duplicates_skipped']}"
    )
    for split, info in meta["splits"].items():
        print(f"  - {split:5s}: {info['games']:6d} ván | {info['samples']:7d} mẫu")


if __name__ == "__main__":
    main()
