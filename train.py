"""
Training script for Mask2Former + ResNet-101 on LoveDA.

Examples:
    python train.py
    python train.py --epochs 100 --full
    python train.py --backbone resnet18 --checkpoint checkpoints/best_resnet18.pth
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data.loveda_dataset import get_datasets
from losses import SetCriterion
from models import Mask2Former
from models.backbone import BACKBONE_SPECS
from utils import compute_miou, predictions_to_semantic_map


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(model: Mask2Former) -> AdamW:
    backbone_params = list(model.backbone.parameters())
    other_params = [
        p for p in model.parameters() if not any(p is bp for bp in backbone_params)
    ]
    return AdamW(
        [
            {"params": backbone_params, "lr": config.LR_BACKBONE},
            {"params": other_params, "lr": config.LR_DECODER},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, masks in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        all_outputs = model(images)
        loss, _ = criterion(all_outputs, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int = config.NUM_CLASSES):
    model.eval()
    all_preds, all_gts = [], []

    for images, masks in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device, non_blocking=True)
        all_outputs = model(images)
        last_output = all_outputs[-1]
        pred_map = predictions_to_semantic_map(
            last_output,
            image_size=(images.shape[2], images.shape[3]),
            num_classes=num_classes,
        )
        all_preds.append(pred_map.cpu())
        all_gts.append(masks)

    pred_cat = torch.cat(all_preds, dim=0)
    gt_cat = torch.cat(all_gts, dim=0)
    return compute_miou(pred_cat, gt_cat, num_classes=num_classes)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mask2Former on LoveDA")
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full training split instead of subset fraction",
    )
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=config.SUBSET_FRACTION,
        help="Subset fraction used when --full is not set",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=config.BACKBONE,
        choices=sorted(BACKBONE_SPECS.keys()),
        help="ResNet backbone",
    )
    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="Disable ImageNet pretrained backbone weights",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="DataLoader batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=config.NUM_WORKERS,
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Output checkpoint path; default is checkpoints/best_{backbone}.pth",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="",
        help="Training CSV log path; default is training_log_{backbone}.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    subset_fraction = 1.0 if args.full else args.subset_fraction
    train_ds, val_ds = get_datasets(
        root=config.DATA_ROOT,
        download=True,
        subset_fraction=subset_fraction,
        image_size=config.IMAGE_SIZE,
        seed=config.SEED,
    )
    print(f"Backbone: {args.backbone}")
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    pin_memory = device.type == "cuda"
    drop_last = len(train_ds) >= args.batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = Mask2Former(
        num_classes=config.NUM_CLASSES,
        num_queries=config.NUM_QUERIES,
        hidden_dim=config.HIDDEN_DIM,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        backbone_name=args.backbone,
        pretrained_backbone=not args.no_pretrained_backbone,
    ).to(device)

    criterion = SetCriterion(num_classes=config.NUM_CLASSES).to(device)
    optimizer = build_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path("checkpoints") / f"best_{args.backbone}.pth"
    )
    log_path = (
        Path(args.log_path)
        if args.log_path
        else Path(f"training_log_{args.backbone}.csv")
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.parent != Path("."):
        log_path.parent.mkdir(parents=True, exist_ok=True)

    best_miou = -1.0
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_miou", "backbone"])

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_miou = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"  train_loss={train_loss:.4f}  val_mIoU={val_miou:.4f}")

        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch, f"{train_loss:.6f}", f"{val_miou:.6f}", args.backbone]
            )

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(
                {
                    "epoch": epoch,
                    "backbone": args.backbone,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_miou": val_miou,
                },
                checkpoint_path,
            )
            print(f"  New best mIoU: {best_miou:.4f} -> checkpoint saved")

    print(f"\nTraining complete ({args.epochs} epoch(s)).")
    print(f"Best val mIoU: {best_miou:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
