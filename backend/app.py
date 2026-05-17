# =============================================================
#  app.py  –  Flask API Server for U-Net Lane Detection
#
#  Routes:
#    GET  /health              → server + model status check
#    POST /predict             → upload driving image, get lane mask
#    POST /predict-with-mask   → upload image + ground truth, get metrics
#    GET  /model-info          → model architecture details
#
#  Usage:
#    pip install -r requirements.txt
#    python app.py
#    Server runs on http://localhost:5000
# =============================================================

import os
import sys
import uuid
import time
import logging
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── Make ai/ importable from backend/ ────────────────────────
AI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ai')
sys.path.insert(0, os.path.abspath(AI_DIR))

import config
from inference import (
    run_prediction,
    run_prediction_with_mask,
    load_model_once,
)

# ── App setup ─────────────────────────────────────────────────
app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
        ]
    }
})

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s  [%(levelname)s]  %(message)s',
    datefmt = '%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── Upload folder ─────────────────────────────────────────────
UPLOAD_FOLDER      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024   # 16 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER']     = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# ── Load model once at startup ────────────────────────────────
logger.info("Loading U-Net Lane Detection model …")
MODEL, DEVICE = load_model_once()
logger.info(f"Model ready on {DEVICE} ✓")


# ── Helpers ───────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def unique_filename(original: str) -> str:
    ext = Path(original).suffix
    return f"{uuid.uuid4().hex}{ext}"


def error_response(message: str, status: int = 400):
    return jsonify({"success": False, "error": message}), status


# ── Routes ────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "success"      : True,
        "status"       : "ok",
        "model_loaded" : MODEL is not None,
        "device"       : str(DEVICE),
        "model_path"   : config.MODEL_PATH,
        "timestamp"    : time.time(),
    }), 200


@app.route('/model-info', methods=['GET'])
def model_info():
    if MODEL is None:
        return error_response("Model not loaded", 503)

    return jsonify({
        "success"          : True,
        "architecture"     : "U-Net (ResNet34 encoder)",
        "input_shape"      : [config.IMAGE_CHANNELS,
                              config.IMAGE_HEIGHT,
                              config.IMAGE_WIDTH],
        "output_shape"     : [config.MASK_CHANNELS,
                              config.IMAGE_HEIGHT,
                              config.IMAGE_WIDTH],
        "trainable_params" : MODEL.count_parameters(),
        "mask_threshold"   : config.MASK_THRESHOLD,
        "dataset"          : "KITTI Road/Lane Detection (Kaggle)",
        "task"             : "Binary Lane Segmentation",
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts a single driving image and returns lane segmentation.

    Request (multipart/form-data):
        file : image file (.png / .jpg / .jpeg)

    Response (JSON):
        {
          "success"         : true,
          "pred_mask_b64"   : "<base64 PNG>",
          "overlay_b64"     : "<base64 PNG>",
          "original_b64"    : "<base64 PNG>",
          "prob_map_b64"    : "<base64 PNG>",
          "lane_coverage"   : 12.34,
          "inference_ms"    : 45.2,
        }
    """
    if 'file' not in request.files:
        return error_response("No file part in request. Send image as 'file' field.")

    file = request.files['file']

    if file.filename == '':
        return error_response("No file selected.")

    if not allowed_file(file.filename):
        return error_response(
            f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        image_bytes = file.read()
    except Exception as e:
        return error_response("Failed to read uploaded file.", 500)

    try:
        save_name = unique_filename(secure_filename(file.filename))
        with open(os.path.join(UPLOAD_FOLDER, save_name), 'wb') as f:
            f.write(image_bytes)
    except Exception as e:
        logger.warning(f"Could not save upload: {e}")

    try:
        t_start      = time.perf_counter()
        result       = run_prediction(image_bytes, MODEL, DEVICE)
        inference_ms = (time.perf_counter() - t_start) * 1000
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return error_response(f"Inference failed: {str(e)}", 500)

    logger.info(f"Prediction done in {inference_ms:.1f} ms  |  "
                f"Lane coverage: {result['lane_coverage']:.2f}%")

    return jsonify({
        "success"       : True,
        "pred_mask_b64" : result["pred_mask_b64"],
        "overlay_b64"   : result["overlay_b64"],
        "original_b64"  : result["original_b64"],
        "prob_map_b64"  : result["prob_map_b64"],
        "lane_coverage" : round(result["lane_coverage"], 4),
        "inference_ms"  : round(inference_ms, 2),
    }), 200


@app.route('/predict-with-mask', methods=['POST'])
def predict_with_mask():
    """
    Accepts a driving image AND a ground-truth mask.
    Returns prediction + Dice / IoU / Accuracy metrics.

    Request (multipart/form-data):
        image : driving image file
        mask  : ground-truth mask file

    Response (JSON):
        { ...same as /predict..., "metrics": { dice, iou, accuracy } }
    """
    if 'image' not in request.files:
        return error_response("Missing 'image' field in request.")
    if 'mask' not in request.files:
        return error_response("Missing 'mask' field in request.")

    image_file = request.files['image']
    mask_file  = request.files['mask']

    for f, label in [(image_file, 'image'), (mask_file, 'mask')]:
        if f.filename == '':
            return error_response(f"No {label} file selected.")
        if not allowed_file(f.filename):
            return error_response(f"{label} file type not allowed.")

    try:
        image_bytes = image_file.read()
        mask_bytes  = mask_file.read()
    except Exception as e:
        return error_response(f"Failed to read files: {e}", 500)

    try:
        t_start      = time.perf_counter()
        result       = run_prediction_with_mask(image_bytes, mask_bytes, MODEL, DEVICE)
        inference_ms = (time.perf_counter() - t_start) * 1000
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return error_response(f"Inference failed: {str(e)}", 500)

    logger.info(
        f"Prediction+metrics done in {inference_ms:.1f} ms  |  "
        f"Dice={result['metrics']['dice']:.4f}  "
        f"IoU={result['metrics']['iou']:.4f}"
    )

    return jsonify({
        "success"       : True,
        "pred_mask_b64" : result["pred_mask_b64"],
        "overlay_b64"   : result["overlay_b64"],
        "original_b64"  : result["original_b64"],
        "prob_map_b64"  : result["prob_map_b64"],
        "gt_mask_b64"   : result["gt_mask_b64"],
        "lane_coverage" : round(result["lane_coverage"], 4),
        "inference_ms"  : round(inference_ms, 2),
        "metrics"       : result["metrics"],
    }), 200


# ── Error handlers ────────────────────────────────────────────

@app.errorhandler(413)
def file_too_large(e):
    return error_response(
        f"File too large. Maximum size is {MAX_CONTENT_LENGTH // (1024*1024)} MB.", 413
    )

@app.errorhandler(404)
def not_found(e):
    return error_response("Endpoint not found.", 404)

@app.errorhandler(500)
def server_error(e):
    return error_response("Internal server error.", 500)


# ── Entry point ───────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  U-Net Lane Detection — Flask API")
    print("="*55)
    print(f"  Model  : {config.MODEL_PATH}")
    print(f"  Device : {DEVICE}")
    print(f"  URL    : http://localhost:5000")
    print("="*55 + "\n")

    app.run(
        host  = '0.0.0.0',
        port  = 5000,
        debug = False,
    )
