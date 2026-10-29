"""
chess_ai/etl/pgn_etl.py
Đường ống ETL đọc tệp Lichess Elite Database, lọc nhanh ở mức Header (ELO >= 2100,
TimeControl >= 180s), lấy mẫu cách quãng k in {3, 4}, khử trùng lặp thế cờ bằng Zobrist 64-bit,
chia tách Game-level split tất định qua MD5(Game ID) và ghi ra mảng nhị phân np.memmap.
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""


def run_etl_pipeline(
    config_path: str = "configs/default_config.yaml",
    input_path: str = "data/raw/lichess_elite.pgn.zst",
    output_dir: str = "data/processed",
) -> None:
    """Chạy toàn bộ quy trình ETL ngoại tuyến và xuất dataset_metadata.json."""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO (TV2): Parse CLI arguments và gọi run_etl_pipeline()
    pass

