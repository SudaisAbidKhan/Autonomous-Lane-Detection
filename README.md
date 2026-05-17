# Autonomous Lane Detection System

## Project Overview

This is a comprehensive **Deep Learning-based Lane Detection System** designed to identify and segment road lanes in autonomous driving scenarios. The system uses a U-Net semantic segmentation model trained on the KITTI Road Detection dataset and provides a complete full-stack web application for inference and visualization.

## System Architecture

The project is structured as a three-tier application:

### 1. **AI/ML Component** (`ai/` directory)

- **Deep Learning Model**: U-Net architecture with ResNet34 encoder
- **Framework**: PyTorch 2.2+ with segmentation-models-pytorch library
- **Input**: RGB driving images (224×224 pixels)
- **Output**: Binary lane segmentation masks
- **Model Weights**: Pre-trained on KITTI road detection dataset
- **Training Script**: Configurable hyperparameters for fine-tuning
- **Architecture Details**:
  - Encoder: ResNet34 with ImageNet pretraining
  - Decoder: Progressive upsampling with skip connections
  - Activation: Sigmoid for binary classification (lane vs non-lane)
  - Dropout: 0.1 for regularization

### 2. **Backend API Server** (`backend/` directory)

- **Framework**: Flask 3.0.3 with Flask-CORS
- **Server**: Gunicorn for production deployment
- **Port**: 5000 (development)

**Available Endpoints:**

- `GET /health` - Server and model status verification
- `POST /predict` - Single image lane detection
- `POST /predict-with-mask` - Prediction with ground truth comparison and metrics
- `GET /model-info` - Model architecture and parameter information

**Features:**

- CORS enabled for localhost development (ports 3000, 5173)
- File upload handling with security validation
- Maximum file size: 16MB
- Supported formats: PNG, JPG, JPEG, BMP
- Efficient model loading and caching at startup
- Structured logging for debugging

### 3. **Frontend Web Interface** (`frontend/` directory)

- **Framework**: React 19.2.6 with Vite bundler
- **Build Tool**: Vite 8.0.12 for fast development
- **Routing**: React Router v7.15 for multi-page navigation

**Pages:**

- **Home**: Main interface for image upload and prediction
- **About**: Project information and technical details

**Components:**

- Upload Panel - Drag-and-drop or click file selection
- Result Panel - Visual display of predicted lane masks
- Metrics Card - Statistical evaluation when ground truth is available
- Header - Navigation and status indicators
- Loader - Loading state feedback

**Services:**

- API Module - Centralized backend communication
- Health Check - Server connectivity verification
- Model Info Retrieval - Architecture details display

## Project Features

### Image Processing & Augmentation

- Image resizing to 224×224 pixels
- Normalization with ImageNet statistics (mean: 0.485, 0.456, 0.406 | std: 0.229, 0.224, 0.225)
- Albumentations library for data augmentation during training
- Binary mask conversion from KITTI format (pink/magenta lane pixels)

### Model Training Capabilities

- Configurable training hyperparameters:
  - Batch size, learning rate, number of epochs
  - Validation split ratio
  - Image resolution and encoder architecture
  - Mask threshold for binary classification
- Data augmentation for improved generalization
- Train/validation split with DataLoader
- Loss tracking and model checkpointing
- Support for CPU and GPU training (CUDA 12.1 compatible)

### Inference Optimization

- Single model instance loaded at server startup
- Device detection (CPU/GPU) for optimal performance
- Batch processing capability through API
- Base64 image encoding for API responses
- Threshold-based lane detection (default: 0.5)

## Dataset Information

**KITTI Road Detection Dataset:**

- Training images: `training/image_2/` - RGB driving scenes
- Ground truth masks: `training/gt_image_2/` - Annotated lane segmentation
- Lane pixels marked in pink/magenta (RGB: R>150, G<100, B>150)
- Collected from autonomous driving perspectives
- Variable image dimensions (resized to 224×224 for model input)

## Technology Stack

### Dependencies Overview

- **Deep Learning**: PyTorch 2.2+, TorchVision 0.17+, Segmentation Models PyTorch 0.3+
- **Web Framework**: Flask 3.0.3, Flask-CORS 4.0.1, Gunicorn 22.0.0
- **Image Processing**: Pillow 10.4.0, NumPy 1.26.4, Albumentations 1.4.10
- **Frontend**: React 19.2.6, React Router 7.15.0, Vite 8.0.12
- **Utilities**: Python-Dotenv 1.0.1, Werkzeug 3.0.3

## Usage Workflows

### Training Workflow

Use the training script with command-line arguments to train or fine-tune the model on your data. Configurable parameters include dataset path, model save location, training epochs, batch size, and learning rate.

### Inference Workflow (Backend API)

1. Start the Flask server
2. Upload an image via POST request to `/predict`
3. Receive predicted lane mask as base64-encoded image
4. Optional: Compare with ground truth for performance metrics

### Interactive Web Workflow

1. Launch the frontend development server
2. Upload driving images through the web interface
3. View real-time lane detection results
4. Optionally upload corresponding masks for metric evaluation
5. Navigate to About page for technical details

## Project Structure Summary

```
├── ai/                          # Machine learning core
│   ├── config.py               # Centralized configuration
│   ├── model.py                # U-Net architecture
│   ├── utils.py                # Helper utilities
│   └── lane_best.pth           # Pre-trained model weights
├── backend/                     # Flask API server
│   ├── app.py                  # Main server application
│   ├── inference.py            # Prediction logic
│   ├── requirements.txt         # Python dependencies
│   └── uploads/                # Temporary file storage
├── frontend/                    # React web application
│   ├── src/
│   │   ├── App.jsx            # Root component
│   │   ├── pages/             # Page components
│   │   ├── components/        # Reusable UI components
│   │   └── services/          # API communication
│   ├── package.json           # NPM dependencies
│   ├── vite.config.js         # Build configuration
│   └── public/                # Static assets
├── train.py                    # Training script
├── predict.py                  # Standalone prediction utility
├── lane_detection.ipynb        # Jupyter notebook (analysis/demo)
└── README.md                   # This file
```

## Performance Specifications

- **Model Parameters**: Trainable with configurable architecture
- **Input Resolution**: 224×224 RGB images
- **Output Resolution**: Binary segmentation mask (224×224)
- **Inference Speed**: Real-time on GPU; near real-time on CPU
- **Supported File Size**: Up to 16MB per image
- **Batch Processing**: Available through API for multiple images

## Key Design Decisions

1. **U-Net Architecture**: Chosen for semantic segmentation with encoder-decoder structure preserving spatial information through skip connections
2. **ResNet34 Encoder**: Balance between model complexity and performance; ImageNet pretraining for transfer learning
3. **Binary Classification**: Lane/non-lane segmentation approach over multi-class detection
4. **224×224 Resolution**: Optimal balance between computational efficiency and feature preservation
5. **Flask Backend**: Lightweight, flexible API framework suitable for ML model serving
6. **React Frontend**: Modern, responsive UI with real-time user feedback
7. **CORS Configuration**: Supports development environments and local testing

## Project Context

This is a comprehensive deep learning assignment focused on autonomous driving perception. The system demonstrates practical implementation of semantic segmentation for lane detection, combining model training, API development, and interactive web interface design. It integrates cutting-edge computer vision techniques with production-level software engineering practices.

---

**Last Updated**: May 2026  
**Status**: Full-stack application with integrated training, inference, and visualization pipeline
