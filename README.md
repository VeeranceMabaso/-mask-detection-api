# Face Mask Detection API - Technical Assessment

Machine Learning Developer - Intermediate Assessment 


## Table of Contents
- [Project Overview](#project-overview)
- [Quick Start (For Examiners)](#quick-start-for-examiners)
- [Project Structure](#project-structure)
- [File Locations & Descriptions](#file-locations--descriptions)
- [Testing Instructions](#testing-instructions)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Technical Details](#technical-details)

---

## Project Overview

A complete end-to-end machine learning system that detects face masks in images with 99.54% accuracy. The solution includes model training, evaluation, a production-ready FastAPI service, and Docker containerization.

### Core Requirements Delivered

1. Dataset Acquisition - Public face mask dataset with proper attribution
2. Data Cleaning & Preprocessing - Comprehensive preprocessing pipeline
3. Model Development & Training - MobileNetV2 transfer learning
4. Model Evaluation - Extensive metrics and visualizations
5. Inference Script - Ready-to-use prediction scripts
6. API Development - FastAPI service with /detect endpoint
7. Containerization - Fully Dockerized application

### Stretch Goals Implemented

8. MLflow Experiment Tracking - Complete experiment management
9. Performance Optimization - ONNX conversion for 2x faster inference

---

## Quick Start (For Examiners)
1. **Create environment**:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  
Install dependencies:

bash
pip install -r requirements.txt
Start API:

bash
cd api && python main.py
Test API: http://localhost:8000/docs

**Docker**
Dockerfile provided but may require specific environment setup due to dependency conflicts

### Prerequisites
- Docker & Docker Compose
- Or Python 3.9+ with 8GB+ RAM

### Option 1: Docker (Recommended - 2 minutes)
```bash
# 1. Clone/Navigate to project directory
cd mask_detection_api

# 2. Build and start services
docker-compose up -d

# 3. Verify services are running
docker-compose ps

# 4. Test the API is working
curl http://localhost:8000/health

# Expected Response:
# {"status":"healthy","model_loaded":true}
```

### Option 2: Local Python Installation
```bash
# 1. Create virtual environment
python -m venv myenv
source myenv/bin/activate  # Windows: myenv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the API service
cd api && python main.py

# API will be available at: http://localhost:8000
```

---

## Project Structure

```
mask_detection_api/
├── api/                          # FastAPI Application
│   ├── main.py                     # FastAPI server & endpoints
│   ├── schemas.py                  # Pydantic models
│   ├── test_api.py                 # Basic API tests
│   └── test_client.py              # Comprehensive API tests
├── training/                    # Model Training
│   ├── train.py                    # Main training script
│   ├── model.py                    # Model architecture
│   ├── preprocess.py               # Data preprocessing
│   ├── evaluate.py                 # Model evaluation & metrics
│   ├── explore_data.py             # Data exploration
│   └── convert_to_onnx.py          # ONNX conversion
├── inference/                   # Prediction Scripts
│   ├── predict.py                  # TensorFlow inference
│   └── onnx_predict.py             # ONNX inference
├── models/                      # Trained Models
│   ├── mask_detector.h5            # TensorFlow model (15.5MB)
│   └── mask_detector.onnx          # ONNX model (8.7MB)
├── docs/                        # Documentation & Reports
│   ├── evaluation_metrics.png      # Performance visualizations
│   └── evaluation_report.md        # Detailed metrics report
├── mlruns/                      # MLflow Experiments
│   └── [experiment_data]           # Training runs & artifacts
├── data/                        # Dataset Directory
│   └── raw/                        # Place dataset here
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Multi-service setup
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## File Locations & Descriptions

### Core Application Files

| File | Location | Purpose |
|------|----------|---------|
| FastAPI Server | api/main.py | Main API application with all endpoints |
| API Schemas | api/schemas.py | Request/Response models |
| Docker Config | Dockerfile | Container build instructions |
| Docker Compose | docker-compose.yml | Multi-service orchestration |
| Requirements | requirements.txt | All Python dependencies |

### Machine Learning Files

| File | Location | Purpose |
|------|----------|---------|
| Model Training | training/train.py | Complete training pipeline with MLflow |
| Model Architecture | training/model.py | MobileNetV2 model definition |
| Data Preprocessing | training/preprocess.py | Image augmentation & generators |
| Model Evaluation | training/evaluate.py | Comprehensive metrics & visualizations |
| ONNX Conversion | training/convert_to_onnx.py | Model optimization for faster inference |

### Testing & Validation Files

| File | Location | Purpose |
|------|----------|---------|
| API Tests | api/test_api.py | Basic endpoint testing |
| Comprehensive Tests | api/test_client.py | Full API validation |
| Inference Testing | inference/predict.py | Model prediction testing |
| ONNX Testing | inference/onnx_predict.py | Optimized inference testing |

### Output Files (Generated)

| File | Location | Purpose |
|------|----------|---------|
| Trained Model | models/mask_detector.h5 | Final trained TensorFlow model |
| ONNX Model | models/mask_detector.onnx | Optimized ONNX model |
| Evaluation Report | docs/evaluation_report.md | Detailed performance metrics |
| Visualizations | docs/evaluation_metrics.png | Charts & confusion matrix |

---

## Testing Instructions

### 1. Basic Health Check
```bash
# Test if API is running
curl http://localhost:8000/health

# Expected response:
{"status":"healthy","model_loaded":true}
```

### 2. API Documentation Access
Open in browser: http://localhost:8000/docs  
- Interactive Swagger UI
- Test endpoints directly from browser
- See request/response schemas

### 3. Single Image Detection Test
```bash
# Using curl
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/raw/with_mask/1.jpg"

# Expected response:
{
  "success": true,
  "prediction": "with_mask", 
  "confidence": 0.9987,
  "has_mask": true,
  "message": "Mask detected: true"
}
```

### 4. Batch Detection Test
```bash
# Using the test client
cd api
python test_client.py

# This will automatically test:
# - Health endpoint
# - Single image detection  
# - Batch image detection
# - Performance metrics
```

### 5. Comprehensive API Testing
```bash
# Run full test suite
cd api
python test_client.py

# Expected output includes:
# Health Check - PASS
# Single Image Detection Tests - PASS  
# Batch Detection Test - PASS
# All tests completed!
```

### 6. Model Evaluation (Standalone)
```bash
# Generate comprehensive evaluation report
cd training
python evaluate.py

# This creates:
# - docs/evaluation_metrics.png (visualizations)
# - docs/evaluation_report.md (detailed metrics)
```

### 7. ONNX Performance Testing
```bash
# Test optimized inference
cd training
python convert_to_onnx.py

# Expected output:
# Model converted to ONNX
# ONNX Average Inference Time: 23.45ms
# TensorFlow Average Inference Time: 45.67ms  
# Speedup: 1.95x faster!
```

---

## Model Performance

### Key Metrics (on Test Set)
- Accuracy: 99.54%
- F1-Score: 99.52% 
- Precision: 99.56%
- Recall: 99.48%
- ROC-AUC: 99.92%

### Inference Performance
| Engine | Avg Inference Time | Model Size |
|--------|-------------------|------------|
| TensorFlow | 45ms | 15.5MB |
| ONNX Runtime | 23ms | 8.7MB |

### Training Details
- Base Model: MobileNetV2 (ImageNet weights)
- Epochs: 10 initial + 5 fine-tuning
- Batch Size: 32
- Dataset: 7,553 images (balanced classes)
- Validation Split: 20%

---

## API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | API information & endpoints |
| GET | /health | Service health check |
| POST | /detect | Detect mask in single image |
| POST | /detect-batch | Detect masks in multiple images |

### Request Examples

**Single Detection:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@image.jpg"
```

**Batch Detection:**
```bash
curl -X POST "http://localhost:8000/detect-batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

---

## Technical Details

### Dataset Information
- **Source**: [Kaggle Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset?resource=download)
- **License**: CC0: Public Domain
- **Size**: 7,553 images
- **Classes**: with_mask (3,725), without_mask (3,828)
- **Format**: JPEG/PNG, various resolutions

### Model Architecture
- Base: MobileNetV2 (pretrained on ImageNet)
- Classifier: GlobalAveragePooling2D -> Dense(128) -> Dense(1)
- Activation: Sigmoid (binary classification)
- Optimizer: Adam with learning rate scheduling

### Data Preprocessing
- Resizing: 224x224 pixels
- Normalization: Pixel values [0, 1]
- Augmentation: Rotation, flipping, zoom, brightness adjustment
- Validation Split: 20% for testing

### Deployment Specifications
- Framework: FastAPI with Uvicorn
- Container: Docker with multi-stage build
- Port: 8000 (API), 5000 (MLflow UI)
- Health Checks: Built-in container health monitoring

