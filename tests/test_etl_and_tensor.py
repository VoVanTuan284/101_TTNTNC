"""
tests/test_etl_and_tensor.py
Kiểm thử biểu diễn Tensor 18 kênh (board_tensor.py), đường ống ETL (pgn_etl.py)
và PyTorch Memmap Dataset (memmap_dataset.py).
"""

import json
from pathlib import Path
import chess
import numpy as np
import pytest

from chess_ai.core.board_tensor import (
    action_index_to_move,
    board_to_tensor,
    board_to_uint8_planes,
    canonical_square,
    get_legal_action_mask,
    move_to_action_index,
)
from chess_ai.etl.memmap_dataset import ChessMemmapDataset
from chess_ai.etl.pgn_etl import (
    deterministic_split,
    inspect_raw_pgn,
    parse_estimated_time_sec,
    run_etl_pipeline,
)


def test_board_tensor_canonical_and_halfmove() -> None:
    board = chess.Board()
    board.halfmove_clock = 37
    t_uint8 = board_to_uint8_planes(board)
    t_f32 = board_to_tensor(board)

    assert t_uint8.shape == (18, 8, 8)
    assert t_uint8.dtype == np.uint8
    assert t_uint8[17, 0, 0] == 37
    assert t_f32.dtype == np.float32
    assert t_f32[17, 0, 0] == pytest.approx(0.37)

    # Đối xứng lật bàn cờ khi Đen đi
    assert canonical_square(chess.E2, chess.WHITE) == chess.E2
    assert canonical_square(chess.E7, chess.BLACK) == chess.E2


def test_action_encoding_and_promotion_decoding() -> None:
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    promo_q = chess.Move.from_uci("a7a8q")
    idx = move_to_action_index(promo_q, chess.WHITE)
    decoded = action_index_to_move(idx, board)
    assert decoded == promo_q

    mask = get_legal_action_mask(board)
    assert mask.shape == (4096,)
    assert mask[idx] == 1.0


def test_etl_helpers_and_pipeline(tmp_path: Path) -> None:
    assert parse_estimated_time_sec("180+0") == 180
    assert parse_estimated_time_sec("60+2") == 140
    assert deterministic_split("https://lichess.org/abc123") in ("train", "val", "test")

    sample_pgn = tmp_path / "sample.pgn"
    moves_str = (
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 "
        "6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 "
        "11. Nbd2 Bb7 12. Bc2 Re8 13. Nf1 Bf8 14. Ng3 g6 15. Bg5 h6 "
        "16. Bd2 Bg7 1-0"
    )
    sample_pgn.write_text(
        f'[Event "Rated Blitz game"]\n'
        f'[Site "https://lichess.org/testgame01"]\n'
        f'[White "GM_A"]\n[Black "GM_B"]\n'
        f'[Result "1-0"]\n[WhiteElo "2450"]\n[BlackElo "2420"]\n'
        f'[TimeControl "180+2"]\n[Termination "Normal"]\n\n{moves_str}\n',
        encoding="utf-8",
    )

    eda = inspect_raw_pgn(str(sample_pgn), max_games=10)
    assert eda["games_inspected"] == 1
    assert eda["header_passed"] == 1

    out_dir = tmp_path / "processed"
    meta = run_etl_pipeline(
        input_path=str(sample_pgn),
        output_dir=str(out_dir),
        max_games=10,
        max_samples=100,
    )
    assert meta["valid_games"] == 1
    assert meta["total_samples"] > 0
    assert (out_dir / "dataset_metadata.json").exists()

    # Tìm split chứa ván testgame01 và nạp qua ChessMemmapDataset
    split = deterministic_split("https://lichess.org/testgame01")
    ds = ChessMemmapDataset(str(out_dir), split=split)
    assert len(ds) == meta["splits"][split]["samples"]
    s, a, v = ds[0]
    assert s.shape == (18, 8, 8)
    assert 0 <= int(a) < 4096
    assert float(v) in (-1.0, 0.0, 1.0)


def test_level5_dual_resnet_loss_agent_and_resume(tmp_path: Path) -> None:
    import torch
    from chess_ai.agents.level5_supervised import Level5SupervisedAgent
    from chess_ai.core.chess_env import ChessEnv
    from chess_ai.models.dual_resnet import DualHeadResNet
    from chess_ai.models.loss import DualHeadLoss
    from chess_ai.training.train_supervised import train_supervised

    # 1. Kiểm tra kiến trúc DualHeadResNet & DualHeadLoss
    model = DualHeadResNet()
    criterion = DualHeadLoss()
    x = torch.randn(4, 18, 8, 8)
    target_a = torch.tensor([0, 12, 100, 4095], dtype=torch.long)
    target_v = torch.tensor([1.0, -1.0, 0.0, 0.5], dtype=torch.float32)

    p_logits, v_pred = model(x)
    assert p_logits.shape == (4, 4096)
    assert v_pred.shape == (4, 1)
    assert torch.all(v_pred >= -1.0) and torch.all(v_pred <= 1.0)

    loss, parts = criterion(p_logits, v_pred, target_a, target_v)
    assert loss.item() > 0.0
    assert "policy_loss" in parts and "value_loss" in parts

    # 2. Tạo tập memmap giả lập nhỏ (16 mẫu train, 8 mẫu val, 8 mẫu test) để test train_supervised & --resume
    data_dir = tmp_path / "memmap_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "splits": {
            "train": {"games": 2, "samples": 16},
            "val": {"games": 1, "samples": 8},
            "test": {"games": 1, "samples": 8},
        }
    }
    (data_dir / "dataset_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    for sp, n_s in (("train", 16), ("val", 8), ("test", 8)):
        st = np.memmap(data_dir / f"{sp}_states.dat", dtype=np.uint8, mode="w+", shape=(n_s, 18, 8, 8))
        ac = np.memmap(data_dir / f"{sp}_actions.dat", dtype=np.int16, mode="w+", shape=(n_s,))
        va = np.memmap(data_dir / f"{sp}_values.dat", dtype=np.float32, mode="w+", shape=(n_s,))
        st[:] = np.random.randint(0, 2, size=(n_s, 18, 8, 8), dtype=np.uint8)
        ac[:] = np.random.randint(0, 4096, size=(n_s,), dtype=np.int16)
        va[:] = np.random.choice([-1.0, 0.0, 1.0], size=(n_s,)).astype(np.float32)
        st.flush()
        ac.flush()
        va.flush()

    ckpt_dir = tmp_path / "ckpts"
    metrics_out = tmp_path / "sl_metrics.json"

    # Chạy 1 epoch đầu
    rep1 = train_supervised(
        data_dir=str(data_dir),
        checkpoint_dir=str(ckpt_dir),
        metrics_output=str(metrics_out),
        epochs_override=1,
        batch_size_override=8,
        num_workers_override=0,
        verbose=False,
    )
    assert rep1["epochs_completed"] == 1
    assert (ckpt_dir / "latest.pt").exists()
    assert (ckpt_dir / "best_model.pt").exists()

    # Kiểm tra Resume từ latest.pt chạy tiếp lên epoch 2
    rep2 = train_supervised(
        resume_path=str(ckpt_dir / "latest.pt"),
        data_dir=str(data_dir),
        checkpoint_dir=str(ckpt_dir),
        metrics_output=str(metrics_out),
        epochs_override=2,
        batch_size_override=8,
        num_workers_override=0,
        verbose=False,
    )
    assert rep2["epochs_completed"] == 2
    assert len(rep2["history"]) == 2

    # 3. Kiểm tra Level5SupervisedAgent nạp best_model.pt và chọn nước đi hợp lệ
    agent = Level5SupervisedAgent(checkpoint_path=str(ckpt_dir / "best_model.pt"))
    assert agent.checkpoint_loaded is True
    env = ChessEnv()
    move = agent.select_move(env)
    assert move in env.legal_actions()
    assert -1.0 <= agent.last_search_stats["eval"] <= 1.0
