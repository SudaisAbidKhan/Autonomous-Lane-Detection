# =============================================================
#  predict.py  –  Lane Detection Inference Script
#
#  Single image:
#    python predict.py \
#        --image_path "/path/to/image.png" \
#        --model_path "/content/models/lane_best.pth" \
#        --output_path "/content/output/prediction.png"
#
#  Flask API server:
#    python predict.py \
#        --serve \
#        --model_path "/content/models/lane_best.pth" \
#        --port 5000
# =============================================================

import os
import sys
import argparse
import io
import base64

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# ═════════════════════════════════════════════════════════════
#  1.  Config / Constants
# ═════════════════════════════════════════════════════════════

IMAGE_SIZE     = 224
MASK_THRESHOLD = 0.5
MEAN           = (0.485, 0.456, 0.406)
STD            = (0.229, 0.224, 0.225)
OVERLAY_COLOR  = (0, 220, 90)   # green for lanes

TRANSFORM = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])


# ═════════════════════════════════════════════════════════════
#  2.  Model loader
# ═════════════════════════════════════════════════════════════

def load_model(model_path: str, device: torch.device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'Model not found: {model_path}\n'
            'Run the training notebook first.'
        )
    model = smp.Unet(
        encoder_name    = 'resnet34',
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    )
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f'[INFO] Model loaded → {model_path}')
    return model


# ═════════════════════════════════════════════════════════════
#  3.  Helpers
# ═════════════════════════════════════════════════════════════

def preprocess(image_np: np.ndarray) -> torch.Tensor:
    aug = TRANSFORM(image=image_np)
    return aug['image'].unsqueeze(0)   # [1, 3, H, W]


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    t = tensor.cpu().permute(1, 2, 0).numpy()
    t = t * np.array(STD) + np.array(MEAN)
    return np.clip(t * 255, 0, 255).astype(np.uint8)


def build_overlay(image_np: np.ndarray, mask_np: np.ndarray,
                  alpha: float = 0.45, color=OVERLAY_COLOR) -> np.ndarray:
    overlay = image_np.copy()
    roi = mask_np.astype(bool)
    if roi.any():
        overlay[roi] = (
            (1 - alpha) * overlay[roi].astype(np.float32) +
            alpha * np.array(color, dtype=np.float32)
        ).astype(np.uint8)
    return overlay


def build_heatmap(prob_np: np.ndarray) -> np.ndarray:
    r = np.clip(prob_np * 3,     0, 1)
    g = np.clip(prob_np * 3 - 1, 0, 1)
    b = np.clip(prob_np * 3 - 2, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def ndarray_to_b64(arr: np.ndarray, mode: str = 'RGB') -> str:
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def mask_to_binary_kitti(mask_bytes: bytes) -> np.ndarray:
    """Convert KITTI pink/magenta ground-truth mask to binary."""
    arr = np.array(Image.open(io.BytesIO(mask_bytes)).convert('RGB'), dtype=np.uint8)
    return (
        (arr[:, :, 0] > 150) &
        (arr[:, :, 1] < 100) &
        (arr[:, :, 2] > 150)
    ).astype(np.float32)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    p  = pred.flatten(); t = target.flatten()
    tp = float((p * t).sum())
    fp = float((p * (1 - t)).sum())
    fn = float(((1 - p) * t).sum())
    tn = float(((1 - p) * (1 - t)).sum())
    return {
        'dice'    : round((2*tp+1e-6)/(2*tp+fp+fn+1e-6),   4),
        'iou'     : round((tp+1e-6)/(tp+fp+fn+1e-6),       4),
        'accuracy': round((tp+tn)/(tp+tn+fp+fn+1e-6),      4),
    }


# ═════════════════════════════════════════════════════════════
#  4.  Core prediction
# ═════════════════════════════════════════════════════════════

def predict(image_np: np.ndarray, model, device: torch.device,
            threshold: float = MASK_THRESHOLD) -> dict:
    orig_h, orig_w = image_np.shape[:2]
    tensor = preprocess(image_np).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits)
        preds  = (probs > threshold).float()

    # Resize back to original dims
    prob_256 = probs[0, 0].cpu().numpy()
    pred_256 = preds[0, 0].cpu().numpy()

    prob_orig = np.array(
        Image.fromarray((prob_256 * 255).astype(np.uint8))
             .resize((orig_w, orig_h), Image.BILINEAR),
        dtype=np.float32
    ) / 255.0
    pred_orig = (np.array(
        Image.fromarray((pred_256 * 255).astype(np.uint8))
             .resize((orig_w, orig_h), Image.NEAREST),
        dtype=np.float32
    ) > 127).astype(np.float32)

    display = denormalize(tensor[0].cpu())
    display = np.array(
        Image.fromarray(display).resize((orig_w, orig_h), Image.BILINEAR)
    )

    overlay = build_overlay(display, pred_orig)
    heatmap = build_heatmap(prob_orig)
    coverage = float(pred_orig.sum() / pred_orig.size * 100)

    return {
        'display'      : display,
        'pred_mask'    : pred_orig,
        'overlay'      : overlay,
        'heatmap'      : heatmap,
        'prob_map'     : prob_orig,
        'lane_coverage': coverage,
    }


# ═════════════════════════════════════════════════════════════
#  5.  Single image CLI mode
# ═════════════════════════════════════════════════════════════

def run_single(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(args.model_path, device)

    image_np = np.array(Image.open(args.image_path).convert('RGB'), dtype=np.uint8)
    result   = predict(image_np, model, device)

    # Save visualization
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [
        (result['display'], 'Original Image',   None),
        (result['pred_mask'],'Predicted Mask',  'gray'),
        (result['overlay'],  'Lane Overlay',    None),
    ]
    for ax, (img, title, cmap) in zip(axes, pairs):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axis('off')

    plt.suptitle(
        f'Lane Coverage: {result["lane_coverage"]:.2f}%  |  Threshold: {MASK_THRESHOLD}',
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=120, bbox_inches='tight')
    plt.show()

    print(f'[INFO] Visualization saved → {args.output_path}')
    print(f'[DONE] Segmentation complete  |  Lane coverage: {result["lane_coverage"]:.2f}%')


# ═════════════════════════════════════════════════════════════
#  6.  Flask API server mode
# ═════════════════════════════════════════════════════════════

def run_server(args):
    from flask import Flask, request, jsonify
    from flask_cors import CORS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(args.model_path, device)

    app = Flask(__name__)
    CORS(app)

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok', 'model_loaded': True, 'device': str(device)})

    @app.route('/predict', methods=['POST'])
    def predict_route():
        if 'file' not in request.files:
            return jsonify({'error': "No 'file' field"}), 400
        file  = request.files['file']
        image_np = np.array(Image.open(file.stream).convert('RGB'), dtype=np.uint8)

        import time
        t0     = time.perf_counter()
        result = predict(image_np, model, device)
        ms     = (time.perf_counter() - t0) * 1000

        return jsonify({
            'success'       : True,
            'pred_mask_b64' : ndarray_to_b64((result['pred_mask']*255).astype(np.uint8), 'L'),
            'overlay_b64'   : ndarray_to_b64(result['overlay'],  'RGB'),
            'original_b64'  : ndarray_to_b64(result['display'],  'RGB'),
            'prob_map_b64'  : ndarray_to_b64(result['heatmap'],  'RGB'),
            'lane_coverage' : round(result['lane_coverage'], 4),
            'inference_ms'  : round(ms, 2),
        })

    @app.route('/predict-with-mask', methods=['POST'])
    def predict_with_mask():
        if 'image' not in request.files or 'mask' not in request.files:
            return jsonify({'error': "Need 'image' and 'mask' fields"}), 400

        image_np = np.array(Image.open(request.files['image'].stream).convert('RGB'))
        mask_bytes = request.files['mask'].stream.read()

        import time
        t0     = time.perf_counter()
        result = predict(image_np, model, device)
        ms     = (time.perf_counter() - t0) * 1000

        gt = mask_to_binary_kitti(mask_bytes)
        gt = np.array(Image.fromarray((gt*255).astype(np.uint8))
                          .resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST),
                      dtype=np.float32) / 255.0

        pred = result['pred_mask']
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray((pred*255).astype(np.uint8))
                                .resize((gt.shape[1], gt.shape[0]), Image.NEAREST),
                            dtype=np.float32) / 255.0

        return jsonify({
            'success'       : True,
            'pred_mask_b64' : ndarray_to_b64((result['pred_mask']*255).astype(np.uint8), 'L'),
            'overlay_b64'   : ndarray_to_b64(result['overlay'],  'RGB'),
            'original_b64'  : ndarray_to_b64(result['display'],  'RGB'),
            'prob_map_b64'  : ndarray_to_b64(result['heatmap'],  'RGB'),
            'gt_mask_b64'   : ndarray_to_b64((gt*255).astype(np.uint8), 'L'),
            'lane_coverage' : round(result['lane_coverage'], 4),
            'inference_ms'  : round(ms, 2),
            'metrics'       : compute_metrics(pred, gt),
        })

    print(f'\n{"="*50}')
    print(f'  Lane Detection API  →  http://localhost:{args.port}')
    print(f'{"="*50}\n')
    app.run(host='0.0.0.0', port=args.port, debug=False)


# ═════════════════════════════════════════════════════════════
#  7.  Entry point
# ═════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Lane Detection Inference')
    p.add_argument('--image_path',  type=str, default=None)
    p.add_argument('--model_path',  type=str, default='/content/models/lane_best.pth')
    p.add_argument('--output_path', type=str, default='/content/output/prediction.png')
    p.add_argument('--serve',       action='store_true', help='Run as Flask API server')
    p.add_argument('--port',        type=int, default=5000)
    p.add_argument('--no_ngrok',    action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.serve:
        run_server(args)
    else:
        if not args.image_path:
            print('ERROR: Provide --image_path or use --serve flag')
            sys.exit(1)
        run_single(args)
