"""
tests/test_level5_improvements.py
Kiểm thử các cải tiến Level 5: lật gương tăng cường dữ liệu, label smoothing,
value dropout (tương thích checkpoint cũ), và pha trộn nhãn value trong ETL.
"""

import json
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chess_ai.core.board_tensor import board_to_uint8_planes, move_to_action_index
from chess_ai.etl.memmap_dataset import ChessMemmapDataset, mirror_action_index
from chess_ai.etl.pgn_etl import run_etl_pipeline
from chess_ai.models.dual_resnet import DualHeadResNet
from chess_ai.models.loss import DualHeadLoss


def test_mirror_matches_real_flipped_board() -> None:
    """Lật gương mảng tensor + chỉ số hành động phải khớp với bàn cờ lật thật (khi hết quyền nhập thành)."""
    fen = "r3k2r/pp3ppp/2n1b3/3pP3/8/2N2N2/PPP2PPP/R3K2R w - d6 0 12"
    board = chess.Board(fen)
    board.castling_rights = 0
    mirrored = board.transform(chess.flip_horizontal)

    planes = board_to_uint8_planes(board)
    planes_m = board_to_uint8_planes(mirrored)
    assert np.array_equal(planes[:, :, ::-1], planes_m)

    for mv in board.legal_moves:
        mv_m = chess.Move(mv.from_square ^ 7, mv.to_square ^ 7, promotion=mv.promotion)
        idx = move_to_action_index(mv, board.turn)
        idx_m = move_to_action_index(mv_m, mirrored.turn)
        assert mirror_action_index(idx) == idx_m


def test_dataset_mirror_only_without_castling(tmp_path: Path) -> None:
    n = 4
    meta = {"splits": {"train": {"games": 1, "samples": n}, "val": {"games": 0, "samples": 0}, "test": {"games": 0, "samples": 0}}}
    (tmp_path / "dataset_metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    states = np.zeros((n, 18, 8, 8), dtype=np.uint8)
    states[:, 0, 1, 0] = 1  # một Tốt ở a2
    states[2:, 12] = 1      # mẫu 2,3 còn quyền nhập thành -> không được lật
    actions = np.full((n,), 8 * 64 + 16, dtype=np.int16)
    values = np.zeros((n,), dtype=np.float32)
    for name, arr in (("states", states), ("actions", actions), ("values", values)):
        mm = np.memmap(tmp_path / f"train_{name}.dat", dtype=arr.dtype, mode="w+", shape=arr.shape)
        mm[:] = arr
        mm.flush()

    ds = ChessMemmapDataset(str(tmp_path), split="train", augment_mirror=True)
    saw_flip = False
    for _ in range(40):
        s, a, _ = ds[0]
        if float(s[0, 1, 7]) == 1.0:
            saw_flip = True
            assert int(a) == mirror_action_index(8 * 64 + 16)
        else:
            assert float(s[0, 1, 0]) == 1.0
            assert int(a) == 8 * 64 + 16
    assert saw_flip

    for _ in range(20):
        s, a, _ = ds[2]  # còn quyền nhập thành: luôn giữ nguyên
        assert float(s[0, 1, 0]) == 1.0 and int(a) == 8 * 64 + 16


def test_label_smoothing_changes_policy_loss() -> None:
    logits = torch.randn(8, 4096)
    values = torch.zeros(8, 1)
    actions = torch.randint(0, 4096, (8,))
    base, _ = DualHeadLoss()(logits, values, actions, torch.zeros(8))
    smooth, _ = DualHeadLoss(label_smoothing=0.1)(logits, values, actions, torch.zeros(8))
    assert float(base) != pytest.approx(float(smooth))


def test_value_dropout_keeps_old_checkpoint_compatible() -> None:
    old_style = DualHeadResNet(num_res_blocks=1, filters=8)
    new_style = DualHeadResNet(num_res_blocks=1, filters=8, value_dropout=0.3)
    new_style.load_state_dict(old_style.state_dict())  # Dropout không có tham số
    new_style.eval()
    x = torch.randn(2, 18, 8, 8)
    p1, v1 = new_style(x)
    p2, v2 = new_style(x)
    assert torch.allclose(v1, v2)  # eval() tắt dropout -> tất định


def test_etl_value_label_mix(tmp_path: Path) -> None:
    pgn = tmp_path / "g.pgn"
    moves_str = (
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O "
        "9. h3 Nb8 10. d4 Nbd7 11. Nbd2 Bb7 12. Bc2 Re8 13. Nf1 Bf8 14. Ng3 g6 15. Bg5 h6 "
        "16. Bd2 Bg7 1-0"
    )
    pgn.write_text(
        '[Event "x"]\n[Site "https://lichess.org/mixgame01"]\n[White "A"]\n[Black "B"]\n'
        '[Result "1-0"]\n[WhiteElo "2450"]\n[BlackElo "2420"]\n[TimeControl "180+2"]\n'
        f'[Termination "Normal"]\n\n{moves_str}\n',
        encoding="utf-8",
    )
    out = tmp_path / "out"
    meta = run_etl_pipeline(input_path=str(pgn), output_dir=str(out), max_games=5, max_samples=50)
    assert meta["total_samples"] > 0
    assert 0.0 < meta["value_label_mix"] <= 1.0

    vals = []
    for sp in ("train", "val", "test"):
        n = meta["splits"][sp]["samples"]
        if n:
            vals.append(np.fromfile(out / f"{sp}_values.dat", dtype=np.float32, count=n))
    allv = np.concatenate(vals)
    assert np.all(np.abs(allv) <= 1.0 + 1e-6)
    if meta["value_label_mix"] < 1.0:
        assert np.any(np.abs(np.abs(allv) - 1.0) > 1e-4)  # đã pha trộn, không còn toàn ±1/0
