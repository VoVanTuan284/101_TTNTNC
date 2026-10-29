"""
tests/conftest.py
Cấu hình pytest fixtures dùng chung cho toàn bộ 61 kịch bản kiểm thử tự động (TC01 - TC61).
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

from pathlib import Path
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_DIR = PROJECT_ROOT / "resources"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def wac_epd_path() -> Path:
    """Đường dẫn tới bộ 50 thế cờ chiến thuật chuẩn WAC-50."""
    return RESOURCES_DIR / "benchmark_positions.epd"


@pytest.fixture
def sample_pgn_path() -> Path:
    """Đường dẫn tới tệp PGN mẫu phục vụ kiểm thử đơn vị pipeline ETL ngoại tuyến."""
    return FIXTURES_DIR / "sample_games.pgn"

