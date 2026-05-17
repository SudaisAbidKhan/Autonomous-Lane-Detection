# Autonomous Lane Detection System

## Project Overview

This is a comprehensive **Deep Learning-based Lane Detection System** designed to identify and segment road lanes in autonomous driving scenarios. The system uses a U-Net semantic segmentation model trained on the KITTI Road Detection dataset and provides a complete full-stack web application for inference and visualization.

## System Architecture

The project is structured as a three-tier application:

### 1. **AI/ML Component**

The core deep learning model is built on a U-Net architecture with a ResNet34 encoder, leveraging PyTorch 2.2+ and the segmentation-models-pytorch library. The system accepts RGB driving images at 224×224 pixel resolution and produces binary lane segmentation masks as output. The model weights are pre-trained on the KITTI road detection dataset with configurable hyperparameters for fine-tuning and transfer learning. The architecture features a ResNet34 encoder with ImageNet pretraining for robust feature extraction, a progressive upsampling decoder with skip connections to preserve spatial information, sigmoid activation for binary classification between lane and non-lane pixels, and dropout regularization set at 0.1 to prevent overfitting.

### 2. **Backend API Server**

The backend is powered by Flask 3.0.3 with Flask-CORS support and Gunicorn for production deployment, running on port 5000 during development. The API provides four main endpoints: a GET /health endpoint for server and model status verification, a POST /predict endpoint for single image lane detection, a POST /predict-with-mask endpoint for prediction with ground truth comparison and performance metrics, and a GET /model-info endpoint for retrieving model architecture details and parameter information. The server implements CORS configuration for localhost development on ports 3000 and 5173, provides comprehensive file upload handling with built-in security validation, enforces a maximum file size limit of 16MB, supports multiple image formats including PNG, JPG, JPEG, and BMP, efficiently loads and caches the model at startup to optimize inference performance, and includes structured logging for comprehensive debugging and monitoring.

### 3. **Frontend Web Interface**

The frontend is built with React 19.2.6 and Vite 8.0.12 bundler for fast development cycles, featuring React Router v7.15 for seamless multi-page navigation. The application includes two main pages: the Home page serves as the primary interface for image upload and lane prediction, while the About page provides project information and technical details. The interface consists of several reusable components including an Upload Panel supporting both drag-and-drop and click file selection, a Result Panel for visual display of predicted lane masks, a Metrics Card for statistical evaluation when ground truth masks are provided, a Header component for navigation and status indicators, and a Loader component for loading state feedback. The frontend communicates with the backend through a dedicated API module that handles centralized backend communication, server connectivity verification through health checks, and retrieval of model architecture details.

## Project Features

### Image Processing & Augmentation

The system implements comprehensive image processing to ensure consistent model input. Images are resized to a uniform 224×224 pixel dimension and normalized using ImageNet statistics with mean values of 0.485, 0.456, and 0.406 across RGB channels and standard deviation of 0.229, 0.224, and 0.225. During training, the Albumentations library provides robust data augmentation techniques to improve generalization and model robustness. Ground truth masks from the KITTI dataset are converted from their native format, where lane pixels are marked in pink/magenta colors (RGB values with R>150, G<100, B>150), into binary format for supervised learning.

### Model Training Capabilities

The training pipeline offers extensive configurability for optimizing model performance. Hyperparameters including batch size, learning rate, number of epochs, validation split ratio, image resolution, and encoder architecture can be adjusted through command-line arguments. The system implements data augmentation for improved generalization, performs automatic train/validation splitting using PyTorch's DataLoader, tracks loss metrics throughout training, and supports automatic model checkpointing to save the best weights. The framework is compatible with both CPU-only setups and GPU acceleration with CUDA 12.1 support, allowing users to choose the appropriate computational resources based on their hardware availability.

### Inference Optimization

The inference pipeline is optimized for production performance through several key strategies. The model is instantiated once during server startup and remains loaded in memory, eliminating redundant initialization overhead for each prediction request. The system automatically detects available hardware and routes computation to GPU when available or CPU when necessary, ensuring optimal performance across different deployments. Batch processing capability through the API allows simultaneous processing of multiple images. Predictions are returned as base64-encoded images for efficient network transmission, and lane detection uses a configurable threshold of 0.5 to convert continuous sigmoid outputs into binary lane masks.

## Dataset Information

The project utilizes the KITTI Road Detection Dataset, a comprehensive collection of autonomous driving data. Training images are sourced from the training/image_2 directory containing RGB driving scenes captured from real-world autonomous vehicle perspectives. Ground truth masks are provided in the training/gt_image_2 directory with carefully annotated lane segmentation. Lane pixels are distinctively marked in pink/magenta colors using the RGB encoding scheme where R>150, G<100, B>150. The dataset contains images with variable original dimensions, which are standardized to 224×224 pixels during preprocessing to match the model's input requirements. This dataset provides realistic and challenging scenarios for training robust lane detection models.

## Technology Stack

The deep learning infrastructure is built on PyTorch 2.2+ with TorchVision 0.17+ and Segmentation Models PyTorch 0.3+ for semantic segmentation capabilities. The web framework relies on Flask 3.0.3, Flask-CORS 4.0.1 for cross-origin requests, and Gunicorn 22.0.0 for production-grade server deployment. Image processing is handled through Pillow 10.4.0, NumPy 1.26.4 for numerical computations, and Albumentations 1.4.10 for data augmentation. The frontend is built with React 19.2.6 for the user interface, React Router 7.15.0 for application routing, and Vite 8.0.12 as the modern build bundler. Supporting utilities include Python-Dotenv 1.0.1 for environment variable management and Werkzeug 3.0.3 for WSGI utilities and HTTP functionality.

## Usage Workflows

### Training Workflow

The training workflow allows users to prepare and train the model on their own dataset. Users can execute the training script with command-line arguments to specify critical parameters such as the dataset path, model save location, number of training epochs, batch size, and learning rate. The configurable nature of the training pipeline enables both initial model training on new datasets and fine-tuning of the pre-trained weights to adapt to specific use cases or improve performance on specialized domain data.

### Inference Workflow (Backend API)

The backend API provides a streamlined inference pipeline for lane detection. Users begin by starting the Flask server, then submit driving images via POST requests to the /predict endpoint. The server processes the image through the loaded model and returns the predicted lane mask encoded as a base64 string in the response. For evaluation purposes, users can leverage the /predict-with-mask endpoint to simultaneously provide ground truth masks alongside predictions, receiving detailed performance metrics comparing predicted output against the ground truth.

### Interactive Web Workflow

The interactive web application provides an intuitive interface for lane detection tasks. Users launch the frontend development server and interact with the upload panel to submit driving images through either drag-and-drop or traditional file selection. The system displays lane detection results in real-time within the Result Panel component. For advanced analysis, users can optionally upload corresponding mask images to trigger metric evaluation, which compares predictions against ground truth. The application's navigation interface enables seamless access to the About page for reviewing technical documentation and project details.

## Performance Specifications

The model architecture is fully trainable with configurable components allowing adaptation to specific requirements. The system standardizes input to 224×224 pixel RGB images and produces binary segmentation masks at the same resolution. Inference performance varies with hardware configuration, achieving real-time processing speeds on GPU-equipped systems and near real-time performance on CPU-only configurations. The API enforces a maximum file size limit of 16MB per image to manage memory and network bandwidth effectively. The backend supports batch processing capabilities through its API interface, enabling efficient processing of multiple images in succession.

## Key Design Decisions

The project employs U-Net architecture for semantic segmentation because of its proven effectiveness in preserving spatial information through an encoder-decoder structure with skip connections, making it ideal for pixel-level prediction tasks. ResNet34 was chosen as the encoder backbone to provide an optimal balance between model complexity and inference performance, with ImageNet pretraining enabling effective transfer learning from large-scale visual datasets. The implementation focuses on binary classification between lane and non-lane pixels rather than multi-class detection, simplifying the problem while maintaining practical effectiveness for autonomous driving applications. The 224×224 pixel resolution represents a carefully chosen compromise between computational efficiency and feature preservation, allowing reasonably fast inference while maintaining sufficient detail for lane detection. Flask was selected for the backend API due to its lightweight design and flexibility in serving machine learning models, making it suitable for rapid development and deployment scenarios. React was chosen for the frontend to deliver a modern, responsive user interface with efficient updates and smooth user interactions. The CORS configuration supports localhost development on multiple ports, facilitating parallel development of frontend and backend components on the same machine.

## Project Context

This is a comprehensive deep learning assignment focused on autonomous driving perception. The system demonstrates practical implementation of semantic segmentation for lane detection, combining model training, API development, and interactive web interface design. It integrates cutting-edge computer vision techniques with production-level software engineering practices.

---

**Last Updated**: May 2026  
**Status**: Full-stack application with integrated training, inference, and visualization pipeline
