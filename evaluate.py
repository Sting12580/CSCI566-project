"""
Evaluation script for Mask2Former + ResNet-18 on LoveDA val split.

Usage:
    python evaluate.py [--checkpoint checkpoints/best.pth]
"""

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data.loveda_dataset import get_datasets
from models import Mask2Former
from utils import predictions_to_semantic_map, compute_miou


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int = config.NUM_CLASSES):
    model.eval()
    all_preds, all_gts = [], []
    per_class_intersection = torch.zeros(num_classes)
    per_class_union        = torch.zeros(num_classes)

    for images, masks in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        all_outputs = model(images)
        last_output = all_outputs[-1]
        pred_map = predictions_to_semantic_map(
            last_output,
            image_size=(images.shape[2], images.shape[3]),
            num_classes=num_classes,
        ).cpu()

        for cls_idx in range(num_classes):
            cls = cls_idx + 1  # 1-indexed
            pred_cls = (pred_map == cls)
            gt_cls   = (masks    == cls)
            per_class_intersection[cls_idx] += (pred_cls & gt_cls).sum().item()
            per_class_union[cls_idx]        += (pred_cls | gt_cls).sum().item()

    class_names = [
        "Background", "Building", "Road", "Water",
        "Barren", "Forest", "Agriculture",
    ]
    print("\nPer-class IoU:")
    ious = []
    for i, name in enumerate(class_names):
        if per_class_union[i] == 0:
            print(f"  {name:12s}: N/A (not present)")
            continue
        iou = (per_class_intersection[i] / per_class_union[i]).item()
        ious.append(iou)
        print(f"  {name:12s}: {iou:.4f}")

    miou = sum(ious) / len(ious) if ious else 0.0
    print(f"\nmIoU: {miou:.4f}")
    return miou


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mask2Former on LoveDA val")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth",
                        help="Path to model checkpoint")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    _, val_ds = get_datasets(root=config.DATA_ROOT, download=False)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = Mask2Former(
        num_classes=config.NUM_CLASSES,
        num_queries=config.NUM_QUERIES,
        hidden_dim=config.HIDDEN_DIM,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        pretrained_backbone=False,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val mIoU at save: {ckpt.get('val_miou', 'N/A'):.4f})")

    evaluate(model, val_loader, device)


if __name__ == "__main__":
    main()
