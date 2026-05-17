# =============================================================
#  inference.py  –  Model loading + prediction logic
# =============================================================

import os
import sys
import io
import base64
import logging

import numpy as np
from PIL import Image

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Make ai/ importable ───────────────────────────────────────
AI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ai')
sys.path.insert(0, os.path.abspath(AI_DIR))

import config
from model import UNet
from utils import get_device

logger = logging.getLogger(__name__)

# ── Inference transform ────────────────────────────────────────
_TRANSFORM = A.Compose([
    A.Resize(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std =(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])


# ═════════════════════════════════════════════════════════════
#  1.  Model loader
# ═════════════════════════════════════════════════════════════

def load_model_once(model_path: str = None, device: torch.device = None):
    if device is None:
        device = get_device()

    path = model_path or config.MODEL_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model weights not found at: {path}\n"
            f"Train the model first using the Colab notebook."
        )

    model = UNet(
        in_channels     = config.IMAGE_CHANNELS,
        out_channels    = config.MASK_CHANNELS,
        encoder_name    = config.ENCODER_NAME,
        encoder_weights = None,   # weights come from .pth file
    )
    state_dict = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {path}  ({model.count_parameters():,} params)")
    return model, device


# ═════════════════════════════════════════════════════════════
#  2.  Image helpers
# ═════════════════════════════════════════════════════════════

def _bytes_to_rgb_array(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    return np.array(img, dtype=np.uint8)


def _bytes_to_mask_array(file_bytes: bytes) -> np.ndarray:
    """
    Convert KITTI ground-truth mask to binary float32.
    KITTI masks use pink/magenta for road — convert to binary.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    # KITTI road pixels: R > 150, G < 100, B > 150 (pink/magenta)
    road_mask = (
        (arr[:, :, 0] > 150) &
        (arr[:, :, 1] < 100) &
        (arr[:, :, 2] > 150)
    ).astype(np.float32)
    return road_mask


def _preprocess(image_np: np.ndarray) -> torch.Tensor:
    aug    = _TRANSFORM(image=image_np)
    tensor = aug['image']
    return tensor.unsqueeze(0)   # [1, 3, H, W]


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    t    = tensor.cpu().permute(1, 2, 0).numpy()
    t    = t * std + mean
    return np.clip(t * 255, 0, 255).astype(np.uint8)


# ═════════════════════════════════════════════════════════════
#  3.  Post-processing helpers
# ═════════════════════════════════════════════════════════════

def _build_overlay(image_np: np.ndarray,
                   mask_np: np.ndarray,
                   alpha: float = 0.45,
                   color=(0, 255, 100)) -> np.ndarray:   # green for lanes
    overlay = image_np.copy()
    roi = mask_np == 1
    if roi.any():
        overlay[roi] = (
            (1 - alpha) * overlay[roi].astype(np.float32) +
            alpha * np.array(color, dtype=np.float32)
        ).astype(np.uint8)
    return overlay


def _build_prob_map_image(prob_np: np.ndarray) -> np.ndarray:
    p = prob_np
    r = np.clip(p * 3,     0, 1)
    g = np.clip(p * 3 - 1, 0, 1)
    b = np.clip(p * 3 - 2, 0, 1)
    heatmap = np.stack([r, g, b], axis=-1)
    return (heatmap * 255).astype(np.uint8)


def _resize_mask_to_original(mask_np, orig_h, orig_w):
    pil = Image.fromarray((mask_np * 255).astype(np.uint8))
    pil = pil.resize((orig_w, orig_h), Image.NEAREST)
    return (np.array(pil) > 127).astype(np.float32)


def _resize_prob_to_original(prob_np, orig_h, orig_w):
    pil = Image.fromarray((prob_np * 255).astype(np.uint8))
    pil = pil.resize((orig_w, orig_h), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


def _ndarray_to_b64(arr: np.ndarray, mode: str = 'RGB') -> str:
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ═════════════════════════════════════════════════════════════
#  4.  Metrics
# ═════════════════════════════════════════════════════════════

def _compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    p  = pred.flatten()
    t  = target.flatten()
    tp = float((p * t).sum())
    fp = float((p * (1 - t)).sum())
    fn = float(((1 - p) * t).sum())
    tn = float(((1 - p) * (1 - t)).sum())

    dice     = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou      = (tp + 1e-6)     / (tp + fp + fn + 1e-6)
    accuracy = (tp + tn)       / (tp + tn + fp + fn + 1e-6)

    return {
        'dice'    : round(float(dice),     4),
        'iou'     : round(float(iou),      4),
        'accuracy': round(float(accuracy), 4),
    }


# ═════════════════════════════════════════════════════════════
#  5.  Core prediction functions
# ═════════════════════════════════════════════════════════════

def run_prediction(image_bytes: bytes,
                   model,
                   device: torch.device,
                   threshold: float = config.MASK_THRESHOLD) -> dict:

    image_np       = _bytes_to_rgb_array(image_bytes)
    orig_h, orig_w = image_np.shape[:2]
    tensor         = _preprocess(image_np).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits)
        preds  = (probs > threshold).float()

    prob_map_256  = probs[0, 0].cpu().numpy()
    pred_mask_256 = preds[0, 0].cpu().numpy()

    pred_mask = _resize_mask_to_original(pred_mask_256, orig_h, orig_w)
    prob_map  = _resize_prob_to_original(prob_map_256,  orig_h, orig_w)

    original_display = _denormalize(tensor[0].cpu())
    original_display = np.array(
        Image.fromarray(original_display).resize((orig_w, orig_h), Image.BILINEAR)
    )

    overlay      = _build_overlay(original_display, pred_mask)
    prob_map_img = _build_prob_map_image(prob_map)

    lane_coverage = float(pred_mask.sum() / pred_mask.size * 100)

    return {
        'pred_mask_b64': _ndarray_to_b64((pred_mask * 255).astype(np.uint8), mode='L'),
        'overlay_b64'  : _ndarray_to_b64(overlay,          mode='RGB'),
        'original_b64' : _ndarray_to_b64(original_display, mode='RGB'),
        'prob_map_b64' : _ndarray_to_b64(prob_map_img,     mode='RGB'),
        'lane_coverage': lane_coverage,
    }


def run_prediction_with_mask(image_bytes: bytes,
                              mask_bytes: bytes,
                              model,
                              device: torch.device,
                              threshold: float = config.MASK_THRESHOLD) -> dict:

    result = run_prediction(image_bytes, model, device, threshold)

    gt_np = _bytes_to_mask_array(mask_bytes)
    orig_size = Image.open(io.BytesIO(image_bytes)).size   # (W, H)
    gt_resized = _resize_mask_to_original(gt_np, orig_size[1], orig_size[0])

    pred_np = np.array(
        Image.open(io.BytesIO(base64.b64decode(result['pred_mask_b64']))).convert('L'),
        dtype=np.float32
    )
    pred_np = (pred_np > 127).astype(np.float32)

    if pred_np.shape != gt_resized.shape:
        pred_pil = Image.fromarray((pred_np * 255).astype(np.uint8))
        pred_pil = pred_pil.resize(
            (gt_resized.shape[1], gt_resized.shape[0]), Image.NEAREST
        )
        pred_np = (np.array(pred_pil) > 127).astype(np.float32)

    metrics = _compute_metrics(pred_np, gt_resized)

    result['gt_mask_b64'] = _ndarray_to_b64(
        (gt_resized * 255).astype(np.uint8), mode='L'
    )
    result['metrics'] = metrics

    logger.info(f"Metrics → Dice={metrics['dice']}  "
                f"IoU={metrics['iou']}  "
                f"Accuracy={metrics['accuracy']}")
    return result
