# =============================================================
#  config.py  –  Shared configuration for training & inference
# =============================================================

import os

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'lane_best.pth')

# ── Dataset structure ──────────────────────────────────────────
# training/image_2   → input images
# training/gt_image_2 → ground truth masks
TRAIN_IMAGES_DIR = 'training/image_2'
TRAIN_MASKS_DIR  = 'training/gt_image_2'

# ── Image dimensions ───────────────────────────────────────────
IMAGE_HEIGHT   = 224
IMAGE_WIDTH    = 224
IMAGE_CHANNELS = 3   # RGB input
MASK_CHANNELS  = 1   # Binary mask output

# ── Model architecture ─────────────────────────────────────────
ENCODER_NAME    = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'   # pretrained encoder for better results
FEATURES        = [64, 128, 256, 512]
DROPOUT         = 0.1

# ── Inference ──────────────────────────────────────────────────
# Pixel is "lane" if sigmoid probability > this value
MASK_THRESHOLD = 0.5
