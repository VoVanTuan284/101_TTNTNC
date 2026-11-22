"""
chess_ai/training/train_supervised.py
Kịch bản huấn luyện có giám sát mạng Dual-Head ResNet (Cấp 5)
tích hợp Automatic Mixed Precision (AMP FP16) và CheckpointManager hỗ trợ --resume.
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from chess_ai.agents.level5_supervised import build_resnet_from_config
from chess_ai.config import load_yaml_config
from chess_ai.etl.memmap_dataset import ChessMemmapDataset
from chess_ai.models.dual_resnet import DualHeadResNet
from chess_ai.models.loss import DualHeadLoss
from chess_ai.utils.seeding import set_global_seed


@torch.no_grad()
def evaluate_supervised_split(
    model: DualHeadResNet,
    loader: DataLoader,
    criterion: DualHeadLoss,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Đánh giá Top-1, Top-3 Policy Accuracy, Value MSE và Loss trên tập Validation hoặc Test."""
    model.eval()
    total_samples = 0
    sum_total_loss = 0.0
    sum_policy_loss = 0.0
    sum_value_mse = 0.0
    top1_correct = 0
    top3_correct = 0
    zero_baseline_mse_sum = 0.0

    amp_enabled = use_amp and device.type == "cuda"

    for states, actions, values in loader:
        states = states.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        batch_sz = int(states.size(0))

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            policy_logits, value_pred = model(states)
            _, parts = criterion(policy_logits, value_pred, actions, values)

        sum_total_loss += parts["total_loss"] * batch_sz
        sum_policy_loss += parts["policy_loss"] * batch_sz
        sum_value_mse += parts["value_loss"] * batch_sz
        zero_baseline_mse_sum += float((values.view(-1) ** 2).sum().item())

        # Top-1 & Top-3 accuracy
        pred_top1 = torch.argmax(policy_logits, dim=-1)
        top1_correct += int((pred_top1 == actions).sum().item())
        if policy_logits.size(-1) >= 3:
            _, pred_top3 = torch.topk(policy_logits, k=3, dim=-1)
            top3_correct += int((pred_top3 == actions.unsqueeze(-1)).any(dim=-1).sum().item())

        total_samples += batch_sz

    denom = max(1, total_samples)
    return {
        "samples": total_samples,
        "total_loss": round(sum_total_loss / denom, 5),
        "policy_loss": round(sum_policy_loss / denom, 5),
        "value_mse": round(sum_value_mse / denom, 5),
        "zero_baseline_mse": round(zero_baseline_mse_sum / denom, 5),
        "top1_policy_acc": round(top1_correct / denom, 5),
        "top3_policy_acc": round(top3_correct / denom, 5),
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    model: DualHeadResNet,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    train_loss: float,
    val_policy_acc: float,
    val_value_mse: float,
    best_val_loss: float,
    model_config: Dict[str, Any],
    random_seed: int,
) -> None:
    """Lưu tệp checkpoint tuân thủ đầy đủ 13 trường trong configs/model_resnet.yaml."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else {},
        "scaler_state_dict": scaler.state_dict() if scaler is not None else {},
        "train_loss": float(train_loss),
        "val_policy_acc": float(val_policy_acc),
        "val_value_mse": float(val_value_mse),
        "best_val_loss": float(best_val_loss),
        "model_config": model_config,
        "random_seed": int(random_seed),
        "torch_version": str(torch.__version__),
    }
    torch.save(payload, path)


def train_supervised(
    config_path: str = "configs/model_resnet.yaml",
    resume_path: str = "",
    data_dir: str = "data/processed",
    checkpoint_dir: str = "checkpoints",
    metrics_output: str = "results/training/sl_training_metrics.json",
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    num_workers_override: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Thực thi vòng lặp huấn luyện có giám sát Dual-Head ResNet (Cấp 5),
    lưu checkpoints/latest.pt sau mỗi epoch (phục vụ --resume) và checkpoints/best_model.pt.
    """
    t_start = time.perf_counter()
    set_global_seed(seed)
    cfg = load_yaml_config(config_path)
    train_cfg = cfg.get("training", {})
    hw_cfg = cfg.get("hardware", {})
    loss_cfg = train_cfg.get("loss", {})

    epochs = int(epochs_override if epochs_override is not None else train_cfg.get("epochs", 20))
    batch_size = int(batch_size_override if batch_size_override is not None else train_cfg.get("batch_size", 256))
    lr = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    eta_min = float(train_cfg.get("lr_scheduler", {}).get("eta_min", 1e-5))
    use_amp = bool(hw_cfg.get("use_amp", True))
    num_workers = int(num_workers_override if num_workers_override is not None else hw_cfg.get("num_workers", 0))
    pin_memory = bool(hw_cfg.get("pin_memory", True)) and torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = use_amp and device.type == "cuda"

    # 1. Nạp Dataset từ np.memmap
    train_ds_full = ChessMemmapDataset(data_dir=data_dir, split="train")
    val_ds_full = ChessMemmapDataset(data_dir=data_dir, split="val")
    test_ds_full = ChessMemmapDataset(data_dir=data_dir, split="test")

    train_ds = (
        Subset(train_ds_full, range(min(len(train_ds_full), max_train_samples)))
        if max_train_samples is not None
        else train_ds_full
    )
    val_ds = (
        Subset(val_ds_full, range(min(len(val_ds_full), max_val_samples)))
        if max_val_samples is not None
        else val_ds_full
    )
    test_ds = (
        Subset(test_ds_full, range(min(len(test_ds_full), max_val_samples)))
        if max_val_samples is not None
        else test_ds_full
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 2. Khởi tạo Mạng, Loss, Optimizer AdamW, CosineAnnealingLR & GradScaler
    model = build_resnet_from_config(cfg).to(device)
    criterion = DualHeadLoss(
        policy_weight=float(loss_cfg.get("policy_weight", 1.0)),
        value_weight=float(loss_cfg.get("value_weight", 1.0)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=eta_min)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = ckpt_dir / "latest.pt"
    best_ckpt_path = ckpt_dir / "best_model.pt"

    metrics_path = Path(metrics_output)
    if resume_path:
        r_file = Path(resume_path)
        if not r_file.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint để resume tại: {r_file}")
        ckpt = torch.load(r_file, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] and amp_enabled:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        if metrics_path.exists():
            try:
                old_m = json.loads(metrics_path.read_text(encoding="utf-8"))
                history = [h for h in old_m.get("history", []) if int(h.get("epoch", 0)) < start_epoch]
            except Exception:
                pass
        if verbose:
            print(
                f"[Resume] Đã nạp checkpoint '{r_file}' (tiếp tục từ Epoch {start_epoch}/{epochs}, best_val_loss={best_val_loss:.4f}).",
                flush=True,
            )

    if verbose:
        print(
            f"Bắt đầu huấn luyện Cấp 5 (Dual-Head ResNet) trên thiết bị: {device} | "
            f"Train: {len(train_ds):,} mẫu | Val: {len(val_ds):,} mẫu | Batch: {batch_size} | Epochs: {start_epoch}->{epochs}",
            flush=True,
        )

    total_batches = max(1, len(train_loader))
    log_interval = max(1, total_batches // 20)

    for epoch in range(start_epoch, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        running_total = 0.0
        running_policy = 0.0
        running_value = 0.0
        seen = 0

        for b_idx, (states, actions, values) in enumerate(train_loader, start=1):
            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            bsz = int(states.size(0))

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                policy_logits, value_pred = model(states)
                loss, parts = criterion(policy_logits, value_pred, actions, values)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            seen += bsz
            running_total += parts["total_loss"] * bsz
            running_policy += parts["policy_loss"] * bsz
            running_value += parts["value_loss"] * bsz

            if verbose and (b_idx % log_interval == 0 or b_idx == total_batches):
                pct = int(b_idx * 100 / total_batches)
                avg_l = running_total / max(1, seen)
                print(
                    f"\r  [Epoch {epoch:02d}/{epochs:02d}] Batch {b_idx:4d}/{total_batches} ({pct:3d}%) | "
                    f"Train Loss: {avg_l:.4f} (P: {running_policy/max(1,seen):.4f}, V: {running_value/max(1,seen):.4f})",
                    end="",
                    flush=True,
                )

        scheduler.step()
        train_loss_ep = running_total / max(1, seen)
        train_p_loss_ep = running_policy / max(1, seen)
        train_v_loss_ep = running_value / max(1, seen)

        val_stats = evaluate_supervised_split(model, val_loader, criterion, device, use_amp=amp_enabled)
        val_loss = val_stats["total_loss"]
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # Lưu latest.pt sau mỗi epoch để có thể dừng và --resume bất cứ lúc nào
        save_checkpoint(
            path=latest_ckpt_path,
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loss=train_loss_ep,
            val_policy_acc=val_stats["top1_policy_acc"],
            val_value_mse=val_stats["value_mse"],
            best_val_loss=best_val_loss,
            model_config=cfg,
            random_seed=seed,
        )
        if is_best or not best_ckpt_path.exists():
            save_checkpoint(
                path=best_ckpt_path,
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                train_loss=train_loss_ep,
                val_policy_acc=val_stats["top1_policy_acc"],
                val_value_mse=val_stats["value_mse"],
                best_val_loss=best_val_loss,
                model_config=cfg,
                random_seed=seed,
            )

        ep_elapsed = round(time.perf_counter() - ep_t0, 2)
        ep_record = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": round(float(optimizer.param_groups[0]["lr"]), 7),
            "train_total_loss": round(train_loss_ep, 5),
            "train_policy_loss": round(train_p_loss_ep, 5),
            "train_value_loss": round(train_v_loss_ep, 5),
            "val_total_loss": val_stats["total_loss"],
            "val_policy_loss": val_stats["policy_loss"],
            "val_value_mse": val_stats["value_mse"],
            "val_top1_policy_acc": val_stats["top1_policy_acc"],
            "val_top3_policy_acc": val_stats["top3_policy_acc"],
            "is_best": is_best,
            "elapsed_sec": ep_elapsed,
        }
        history.append(ep_record)

        if verbose:
            star = " ★ [BEST]" if is_best else ""
            print(
                f"\r  [Epoch {epoch:02d}/{epochs:02d}] TrainLoss: {train_loss_ep:.4f} | "
                f"ValLoss: {val_loss:.4f} | Top1Acc: {val_stats['top1_policy_acc']*100:5.2f}% | "
                f"Top3Acc: {val_stats['top3_policy_acc']*100:5.2f}% | ValMSE: {val_stats['value_mse']:.4f} | "
                f"{ep_elapsed:.1f}s{star}",
                flush=True,
            )

        # Ghi cập nhật file metrics JSON ngay sau mỗi epoch
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        partial_report = {
            "experiment": "Level5_Supervised_DualHeadResNet",
            "device": str(device),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "batch_size": batch_size,
            "epochs_completed": epoch,
            "target_epochs": epochs,
            "best_val_loss": round(best_val_loss, 5),
            "best_checkpoint": str(best_ckpt_path),
            "latest_checkpoint": str(latest_ckpt_path),
            "history": history,
        }
        metrics_path.write_text(json.dumps(partial_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3. Đánh giá trên tập Test độc lập bằng best_model.pt
    if best_ckpt_path.exists():
        best_payload = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model_state_dict"])
    test_stats = evaluate_supervised_split(model, test_loader, criterion, device, use_amp=amp_enabled)

    final_report = {
        "experiment": "Level5_Supervised_DualHeadResNet",
        "device": str(device),
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "batch_size": batch_size,
        "epochs_completed": epochs,
        "best_val_loss": round(best_val_loss, 5),
        "best_checkpoint": str(best_ckpt_path),
        "latest_checkpoint": str(latest_ckpt_path),
        "test_evaluation": test_stats,
        "history": history,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(
            f"Hoàn tất huấn luyện Cấp 5! Test Top-1 Acc: {test_stats['top1_policy_acc']*100:.2f}% | "
            f"Top-3 Acc: {test_stats['top3_policy_acc']*100:.2f}% | "
            f"Test Value MSE: {test_stats['value_mse']:.4f} (vs Zero-Baseline MSE: {test_stats['zero_baseline_mse']:.4f})",
            flush=True,
        )
    return final_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện có giám sát mạng Dual-Head ResNet (Cấp 5).")
    parser.add_argument("--config", default="configs/model_resnet.yaml", help="Đường dẫn cấu hình YAML.")
    parser.add_argument("--data-dir", default="data/processed", help="Thư mục chứa dữ liệu np.memmap.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Thư mục lưu best_model.pt và latest.pt.")
    parser.add_argument("--resume", default="", help="Đường dẫn checkpoint (ví dụ: checkpoints/latest.pt) để huấn luyện tiếp.")
    parser.add_argument("--epochs", type=int, default=None, help="Ghi đè tổng số epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Ghi đè kích thước batch.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Giới hạn số mẫu train (hữu ích khi chạy trên CPU).")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Giới hạn số mẫu val/test.")
    parser.add_argument("--num-workers", type=int, default=0, help="Số tiến trình nạp dữ liệu DataLoader.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    train_supervised(
        config_path=args.config,
        resume_path=args.resume,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        num_workers_override=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

