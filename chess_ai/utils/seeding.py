"""
chess_ai/utils/seeding.py
Đồng bộ hóa hạt giống ngẫu nhiên (random, numpy, torch) đảm bảo tái lập thực nghiệm 100%.
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Thiết lập seed đồng bộ cho Python random, NumPy và PyTorch (nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

