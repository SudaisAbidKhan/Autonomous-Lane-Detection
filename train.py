# =============================================================
#  train.py  –  U-Net Lane Detection Training Script
#
#  Usage (from Colab or terminal):
#    python train.py \
#        --data_root        "/content/data/data_road_224" \
#        --model_save_path  "/content/models/lane_best.pth" \
#        --epochs           50 \
#        --batch_size       8 \
#        --image_size       224 \
#        --lr               1e-4
# =============================================================

import os
import sys
import argparse
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# ═════════════════════════════════════════════════════════════
#  1.  Argument parser
# ═════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Train U-Net for lane detection')
    p.add_argument('--data_root',       type=str,   default='/content/data/data_road_224')
    p.add_argument('--model_save_path', type=str,   default='/content/models/lane_best.pth')
    p.add_argument('--epochs',          type=int,   default=50)
    p.add_argument('--batch_size',      type=int,   default=8)
    p.add_argument('--image_size',      type=int,   default=224)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--val_split',       type=float, default=0.15)
    p.add_argument('--num_workers',     type=int,   default=2)
    p.add_argument('--encoder',         type=str,   default='resnet34')
    p.add_argument('--mask_threshold',  type=float, default=0.5)
    return p.parse_args()


# ═════════════════════════════════════════════════════════════
#  2.  Dataset
# ═════════════════════════════════════════════════════════════

def mask_to_binary(mask_path: str) -> np.ndarray:
    """
    KITTI gt_image_2 masks use pink/magenta for road pixels.
    Convert to binary float32 array.
    """
    arr = np.array(Image.open(mask_path).convert('RGB'), dtype=np.uint8)
    # Pink/magenta: R > 150, G < 100, B > 150
    binary = (
        (arr[:, :, 0] > 150) &
        (arr[:, :, 1] < 100) &
        (arr[:, :, 2] > 150)
    ).astype(np.float32)
    return binary


class KITTILaneDataset(Dataset):
    """
    Loads KITTI road images and ground-truth lane masks.

    Expected structure:
        data_root/
        └── training/
            ├── image_2/      ← RGB driving images (.png)
            └── gt_image_2/   ← Annotated lane masks (.png)
    """

    def __init__(self, data_root: str, image_size: int, augment: bool = False):
        self.image_size = image_size
        self.augment    = augment

        img_dir  = os.path.join(data_root, 'training', 'image_2')
        mask_dir = os.path.join(data_root, 'training', 'gt_image_2')

        if not os.path.exists(img_dir):
            raise FileNotFoundError(f'Image directory not found: {img_dir}')
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f'Mask directory not found: {mask_dir}')

        # Match images to masks by filename prefix
        self.pairs = []
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            stem      = Path(fname).stem
            mask_name = f'{stem}_road_{stem}.png'    # KITTI naming: um_road_um_000000.png
            # fallback: look for any mask file that starts with the stem
            if not os.path.exists(os.path.join(mask_dir, mask_name)):
                candidates = [f for f in os.listdir(mask_dir) if f.startswith(stem)]
                mask_name  = candidates[0] if candidates else None
            if mask_name and os.path.exists(os.path.join(mask_dir, mask_name)):
                self.pairs.append((
                    os.path.join(img_dir,  fname),
                    os.path.join(mask_dir, mask_name),
                ))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f'No image-mask pairs found in {data_root}/training/\n'
                f'Check that image_2/ and gt_image_2/ exist and contain matching files.'
            )

        print(f'[Dataset] Found {len(self.pairs)} image-mask pairs')

        # ── Transforms ────────────────────────────────────────
        if augment:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=10, p=0.4),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std =(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std =(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        mask  = mask_to_binary(mask_path)

        aug   = self.transform(image=image, mask=mask)
        image = aug['image']                         # [3, H, W]  float32
        mask  = aug['mask'].unsqueeze(0).float()     # [1, H, W]  float32

        return image, mask


# ═════════════════════════════════════════════════════════════
#  3.  Model
# ═════════════════════════════════════════════════════════════

def build_model(encoder: str, device: torch.device):
    model = smp.Unet(
        encoder_name    = encoder,
        encoder_weights = 'imagenet',
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[Model] U-Net ({encoder} encoder) — {n_params/1e6:.1f}M trainable params')
    return model


# ═════════════════════════════════════════════════════════════
#  4.  Loss
# ═════════════════════════════════════════════════════════════

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce        = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce  = self.bce(logits, targets)
        prob = torch.sigmoid(logits)
        smooth = 1e-6
        inter  = (prob * targets).sum(dim=(2, 3))
        dice   = 1 - (2 * inter + smooth) / (prob.sum(dim=(2, 3)) +
                                              targets.sum(dim=(2, 3)) + smooth)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice.mean()


# ═════════════════════════════════════════════════════════════
#  5.  Metrics
# ═════════════════════════════════════════════════════════════

def compute_metrics(logits, targets, threshold=0.5):
    preds  = (torch.sigmoid(logits) > threshold).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou  = (tp + 1e-6)     / (tp + fp + fn + 1e-6)
    return dice, iou


# ═════════════════════════════════════════════════════════════
#  6.  Train / Val loops
# ═════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, total_dice, total_iou = 0., 0., 0.

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
            logits = model(images)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        dice, iou = compute_metrics(logits.detach(), masks)
        total_loss += loss.item()
        total_dice += dice
        total_iou  += iou

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou = 0., 0., 0.

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss   = criterion(logits, masks)
        dice, iou = compute_metrics(logits, masks)
        total_loss += loss.item()
        total_dice += dice
        total_iou  += iou

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ═════════════════════════════════════════════════════════════
#  7.  Plot & save history
# ═════════════════════════════════════════════════════════════

def save_history_plot(history: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        ('train_loss', 'val_loss', 'LOSS',  'Loss'),
        ('train_dice', 'val_dice', 'DICE',  'Dice'),
        ('train_iou',  'val_iou',  'IOU',   'IoU'),
    ]
    for ax, (tr_key, val_key, title, ylabel) in zip(axes, metrics):
        ax.plot(epochs, history[tr_key],  label='Train')
        ax.plot(epochs, history[val_key], label='Val')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'[Plot] Saved → {path}')


# ═════════════════════════════════════════════════════════════
#  8.  Main
# ═════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n[Config] Device     : {device}')
    print(f'[Config] Data root  : {args.data_root}')
    print(f'[Config] Save path  : {args.model_save_path}')
    print(f'[Config] Epochs     : {args.epochs}')
    print(f'[Config] Batch size : {args.batch_size}')
    print(f'[Config] Image size : {args.image_size}')
    print(f'[Config] LR         : {args.lr}\n')

    # ── Dataset & split ───────────────────────────────────────
    full_dataset = KITTILaneDataset(args.data_root, args.image_size, augment=False)
    val_size     = max(1, int(len(full_dataset) * args.val_split))
    train_size   = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Re-wrap train split with augmentation
    train_ds_aug = KITTILaneDataset(args.data_root, args.image_size, augment=True)
    train_indices = train_ds.indices
    train_ds_aug  = torch.utils.data.Subset(train_ds_aug, train_indices)

    train_loader = DataLoader(train_ds_aug, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,       batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    print(f'[Split] Train: {train_size}  |  Val: {val_size}')

    # ── Model, loss, optimizer ────────────────────────────────
    model     = build_model(args.encoder, device)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # ── Save dirs ──────────────────────────────────────────────
    model_dir = os.path.dirname(args.model_save_path)
    os.makedirs(model_dir or '.', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # ── Training loop ─────────────────────────────────────────
    history = {'train_loss':[], 'val_loss':[],
               'train_dice':[], 'val_dice':[],
               'train_iou': [], 'val_iou': []}
    best_val_dice = 0.0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()

        tr_loss, tr_dice, tr_iou = train_epoch(model, train_loader, optimizer,
                                               criterion, device, scaler)
        vl_loss, vl_dice, vl_iou = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_dice'].append(tr_dice)
        history['val_dice'].append(vl_dice)
        history['train_iou'].append(tr_iou)
        history['val_iou'].append(vl_iou)

        elapsed = time.time() - t_epoch
        print(
            f'Epoch {epoch:3d}/{args.epochs}  '
            f'| Loss {tr_loss:.4f}/{vl_loss:.4f}  '
            f'| Dice {tr_dice:.4f}/{vl_dice:.4f}  '
            f'| IoU {tr_iou:.4f}/{vl_iou:.4f}  '
            f'| {elapsed:.1f}s'
        )

        # Save best model
        if vl_dice > best_val_dice:
            best_val_dice = vl_dice
            torch.save(model.state_dict(), args.model_save_path)
            print(f'  ✓ Best model saved  (val Dice={best_val_dice:.4f})')

    total_time = time.time() - t_start
    print(f'\n[Done] Training complete in {total_time/60:.1f} min')
    print(f'[Done] Best Val Dice : {best_val_dice:.4f}')
    print(f'[Done] Model saved   : {args.model_save_path}')

    # ── Save config & plot ─────────────────────────────────────
    config = {
        'data_root':       args.data_root,
        'model_save_path': args.model_save_path,
        'epochs':          args.epochs,
        'batch_size':      args.batch_size,
        'image_size':      args.image_size,
        'lr':              args.lr,
        'encoder':         args.encoder,
        'best_val_dice':   round(best_val_dice, 4),
        'device':          str(device),
    }
    with open('logs/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print('[Done] Config saved → logs/config.yaml')

    save_history_plot(history, 'logs')


if __name__ == '__main__':
    main()