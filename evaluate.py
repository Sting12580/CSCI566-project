"""
Evaluation script for Mask2Former + ResNet (default: ResNet-101) on LoveDA val split.

Examples:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_resnet101.pth
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data.loveda_dataset import get_datasets
from models import Mask2Former
from models.backbone import BACKBONE_SPECS
from utils import predictions_to_semantic_map


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int = config.NUM_CLASSES):
    model.eval()
    per_class_intersection = torch.zeros(num_classes)
    per_class_union = torch.zeros(num_classes)

    for images, masks in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        all_outputs = model(images)
        last_output = all_outputs[-1]
        pred_map = predictions_to_semantic_map(
            last_output,
            image_size=(images.shape[2], images.shape[3]),
            num_classes=num_classes,
        ).cpu()

        for cls_idx in range(num_classes):
            cls = cls_idx + 1  # 1-indexed labels in LoveDA
            pred_cls = pred_map == cls
            gt_cls = masks == cls
            per_class_intersection[cls_idx] += (pred_cls & gt_cls).sum().item()
            per_class_union[cls_idx] += (pred_cls | gt_cls).sum().item()

    print("\nPer-class IoU:")
    ious = []
    for i, name in enumerate(config.CLASS_NAMES):
        if per_class_union[i] == 0:
            print(f"  {name:12s}: N/A (not present)")
            continue
        iou = (per_class_intersection[i] / per_class_union[i]).item()
        ious.append(iou)
        print(f"  {name:12s}: {iou:.4f}")

    miou = sum(ious) / len(ious) if ious else 0.0
    print(f"\nmIoU: {miou:.4f}")
    return miou


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Mask2Former on LoveDA val")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint path; default is checkpoints/best_{backbone}.pth",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=config.BACKBONE,
        choices=sorted(BACKBONE_SPECS.keys()),
        help="Backbone to build before loading checkpoint",
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
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path("checkpoints") / f"best_{args.backbone}.pth"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device()
    print(f"Using device: {device}")

    _, val_ds = get_datasets(
        root=config.DATA_ROOT,
        download=False,
        subset_fraction=1.0,
        image_size=config.IMAGE_SIZE,
        seed=config.SEED,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_backbone = ckpt.get("backbone", args.backbone)

    model = Mask2Former(
        num_classes=config.NUM_CLASSES,
        num_queries=config.NUM_QUERIES,
        hidden_dim=config.HIDDEN_DIM,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        backbone_name=ckpt_backbone,
        pretrained_backbone=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    saved_epoch = ckpt.get("epoch", "?")
    saved_miou = ckpt.get("val_miou")
    saved_miou_str = f"{saved_miou:.4f}" if isinstance(saved_miou, (int, float)) else "N/A"
    print(
        f"Loaded checkpoint: {checkpoint_path} | epoch={saved_epoch} "
        f"| backbone={ckpt_backbone} | saved val mIoU={saved_miou_str}"
    )

    evaluate(model, val_loader, device)


if __name__ == "__main__":
    main()
