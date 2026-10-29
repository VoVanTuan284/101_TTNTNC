"""
chess_ai/config.py
Bộ nạp cấu hình YAML tập trung (Single Source of Truth) cho toàn dự án.
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

from pathlib import Path
from typing import Any, Dict
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_yaml_config(filename: str = "default_config.yaml") -> Dict[str, Any]:
    """Đọc tệp cấu hình YAML từ thư mục configs/ hoặc đường dẫn chỉ định."""
    path = Path(filename)
    if not path.is_absolute() and not path.exists():
        path = CONFIGS_DIR / filename
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

