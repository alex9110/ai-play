# Integer Factorization ML Project

A complete full-stack machine learning system for integer factorization using PyTorch, FastAPI, and React. This project demonstrates a production-ready ML service with real-time training progress, model versioning, and a modern web interface.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Architecture](#model-architecture)
- [Training Guide](#training-guide)
- [Model Versioning](#model-versioning)
- [Troubleshooting](#troubleshooting)

## 🎯 Project Overview

This project implements a neural network that learns to predict factors of integers. The system consists of:

- **Backend**: FastAPI server with PyTorch ML model
- **Frontend**: React + Vite + TypeScript with real-time training visualization
- **ML Pipeline**: Modular training system with embeddings, checkpoints, and logging
- **Infrastructure**: Dockerized deployment with hot-reload

## 🏗️ Architecture

### Project Structure

```
.
├── backend/
│   ├── core/                    # Core ML modules
│   │   ├── model.py            # PyTorch model with embeddings
│   │   ├── dataset.py          # Dataset generation and loading
│   │   ├── trainer.py          # Training pipeline with logging
│   │   └── inference.py        # Model loading and prediction
│   ├── api/                     # API routes
│   │   ├── routes_predict.py   # Prediction endpoints
│   │   ├── routes_training.py  # Training endpoints with SSE
│   │   ├── routes_dataset.py   # Dataset generation endpoints
│   │   └── routes_model.py     # Model info endpoints
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration management
│   ├── models/                 # Saved model checkpoints
│   ├── logs/                   # Training logs (CSV, JSON, plots)
│   ├── data/                   # Dataset files
│   ├── requirements.txt        # Python dependencies
│   └── Dockerfile              # Backend container
├── frontend/
│   ├── src/
│   │   ├── components/         # React components
│   │   │   ├── NumberInput.tsx
│   │   │   ├── ResultCard.tsx
│   │   │   ├── LoaderSpinner.tsx
│   │   │   └── TrainingProgress.tsx
│   │   ├── App.tsx            # Main application
│   │   ├── api.ts             # API client
│   │   └── main.tsx           # Entry point
│   ├── package.json           # Node dependencies
│   └── Dockerfile             # Frontend container
├── docker-compose.yml         # Docker orchestration
└── README.md                  # This file
```

### System Architecture Diagram

```
┌─────────────────┐
│   React Frontend │
│   (Port 3000)    │
└────────┬─────────┘
         │ HTTP/SSE
         ▼
┌─────────────────┐
│  FastAPI Backend │
│   (Port 8000)    │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────┐
│ PyTorch│ │ Dataset  │
│ Model  │ │ Generator│
└────────┘ └──────────┘
```

## ✨ Features

### Backend
- ✅ **Modular Architecture**: Clean separation of concerns (core, api, config)
- ✅ **Embedding-based Model**: Uses embeddings instead of one-hot encoding
- ✅ **Dynamic Input Support**: Handles variable-length number sequences
- ✅ **Model Versioning**: Automatic versioning (model_v1.pt, model_v2.pt, ...)
- ✅ **Training Logs**: CSV and JSON logs with loss per epoch
- ✅ **Checkpointing**: Saves model after each epoch
- ✅ **SSE Support**: Real-time training progress via Server-Sent Events
- ✅ **Validation Split**: Automatic 80/20 train/validation split
- ✅ **Automatic Dataset Regeneration**: On-demand dataset generation

### Frontend
- ✅ **React + Vite + TypeScript**: Modern, fast development setup
- ✅ **TailwindCSS**: Beautiful, responsive UI
- ✅ **Real-time Training Progress**: SSE integration for live updates
- ✅ **Loss Visualization**: Chart.js graphs for training metrics
- ✅ **Mobile-responsive**: Works on all device sizes
- ✅ **Component-based**: Reusable, clean components

## 🚀 Quick Start

### Using Docker (Recommended)

1. **Clone and navigate to the project**:
   ```bash
   cd experements
   ```

2. **Start the services**:
   ```bash
   docker-compose up --build
   ```

3. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

4. **Generate dataset and train**:
   - Use the web UI, or
   - Use curl commands (see [Usage](#usage))

## 📦 Installation

### Prerequisites

- **Docker & Docker Compose** (for containerized deployment)
- **OR** Python 3.10+ and Node.js 20+ (for local development)

### Local Development Setup

#### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

#### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

   The app will open at http://localhost:3000

## 💻 Usage

### 1. Generate Dataset

**Via Web UI**: Click "Generate Dataset" in the Training section

**Via API**:
```bash
curl -X POST http://localhost:8000/dataset/generate \
  -H "Content-Type: application/json" \
  -d '{
    "num_samples": 10000,
    "min_val": 1000,
    "max_val": 9999
  }'
```

### 2. Train the Model

**Via Web UI**: 
1. Set training parameters (epochs, batch size, learning rate)
2. Optionally check "Resume from checkpoint"
3. Click "Start Training"
4. Watch real-time progress with loss graphs

**Via API**:
```bash
curl -X POST http://localhost:8000/train/start \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.001,
    "resume": false
  }'
```

**Monitor Training Progress (SSE)**:
```bash
curl http://localhost:8000/train/progress
```

### 3. Predict Factors

**Via Web UI**: Enter a number and click "Factorize"

**Via API**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"n": 2021}'
```

**Response**:
```json
{
  "n": 2021,
  "factorA": 43,
  "factorB": 47,
  "raw": [42.8, 47.1]
}
```

## 🔌 API Endpoints

### Health Check
```http
GET /health
```

### Prediction
```http
POST /predict
Content-Type: application/json

{
  "n": 2021
}
```

### Training
```http
POST /train/start
Content-Type: application/json

{
  "epochs": 50,
  "batch_size": 64,
  "learning_rate": 0.001,
  "resume": false
}
```

### Training Progress (SSE)
```http
GET /train/progress
```

Returns Server-Sent Events stream with:
- `epoch`: Current epoch number
- `train_loss`: Training loss
- `val_loss`: Validation loss
- `is_best`: Whether this is the best model so far

### Training Status
```http
GET /train/status
```

### Dataset Generation
```http
POST /dataset/generate
Content-Type: application/json

{
  "num_samples": 10000,
  "min_val": 1000,
  "max_val": 9999
}
```

### Dataset Info
```http
GET /dataset/info
```

### Model Info
```http
GET /model/info
```

Returns model metadata:
- Version number
- Total parameters
- File size
- Last training time
- Best validation loss
- Model configuration

## 🧠 Model Architecture

### Architecture Overview

The model uses a **Multi-Layer Perceptron (MLP) with Embeddings**:

```
Input (digits) → Embedding Layer → Flatten → MLP → Output (2 factors)
```

### Key Components

1. **Embedding Layer**: 
   - Converts digit indices (0-9) to dense vectors
   - Embedding dimension: 32
   - Better generalization than one-hot encoding

2. **MLP Layers**:
   - Hidden dimensions: [256, 512, 256]
   - Activation: ReLU
   - Dropout: 0.2

3. **Output Layer**:
   - 2 neurons (factor A and factor B)
   - No activation (regression)

### Input Encoding

- Numbers are encoded as sequences of digit indices
- Maximum 10 digits supported
- Left-padded with zeros for shorter numbers
- Example: `2021` → `[0, 0, 0, 0, 0, 0, 2, 0, 2, 1]` (for max_digits=10)

### Training

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Validation Split**: 80% train, 20% validation
- **Checkpointing**: Saves after each epoch
- **Best Model**: Automatically saved when validation loss improves

## 🎓 Training Guide

### Initial Training

1. **Generate Dataset**:
   ```bash
   curl -X POST http://localhost:8000/dataset/generate \
     -H "Content-Type: application/json" \
     -d '{"num_samples": 10000}'
   ```

2. **Start Training**:
   ```bash
   curl -X POST http://localhost:8000/train/start \
     -H "Content-Type: application/json" \
     -d '{
       "epochs": 50,
       "batch_size": 64,
       "learning_rate": 0.001
     }'
   ```

3. **Monitor Progress**: 
   - Use the web UI for real-time graphs
   - Or check logs in `backend/logs/training_log.csv`

### Resume Training

To continue training from the latest checkpoint:

```bash
curl -X POST http://localhost:8000/train/start \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 100,
    "resume": true
  }'
```

### Training Parameters

- **`epochs`**: Number of training epochs (1-1000)
- **`batch_size`**: Batch size (1-512)
- **`learning_rate`**: Learning rate (0.0001-1.0)
- **`resume`**: Resume from latest checkpoint (boolean)

### Training Logs

Training logs are saved in `backend/logs/`:

- **`training_log.csv`**: CSV format with timestamp, epoch, train_loss, val_loss
- **`training_log.json`**: Full training history with metadata
- **`loss.png`**: Plot of training and validation loss curves

### Example Training Session

```bash
# 1. Generate dataset
curl -X POST http://localhost:8000/dataset/generate \
  -d '{"num_samples": 20000}'

# 2. Train for 50 epochs
curl -X POST http://localhost:8000/train/start \
  -d '{"epochs": 50, "batch_size": 64}'

# 3. Check training status
curl http://localhost:8000/train/status

# 4. Resume for 50 more epochs
curl -X POST http://localhost:8000/train/start \
  -d '{"epochs": 50, "resume": true}'
```

## 📊 Model Versioning

### Automatic Versioning

Models are automatically versioned:
- `model_v1.pt`: First trained model
- `model_v2.pt`: Second trained model
- `model_v3.pt`: Third trained model
- etc.

Each version is saved when training completes.

### Checkpoints

Epoch checkpoints are saved as:
- `checkpoint_epoch_1.pt`
- `checkpoint_epoch_2.pt`
- etc.

### Model Information

Get information about the current model:

```bash
curl http://localhost:8000/model/info
```

Response includes:
- Version number
- Total parameters
- File size
- Last training time
- Best validation loss
- Model configuration

### Using Specific Model Versions

The system automatically uses the latest model version. To use a specific version, modify `core/inference.py` or update the model loading logic.

## 🔧 Troubleshooting

### Backend Issues

**Problem**: Model not found
- **Solution**: Generate dataset and train the model first
- **Check**: `backend/models/` directory exists and contains model files

**Problem**: Port already in use
- **Solution**: Change port in `docker-compose.yml` or use `--port` flag
- **Check**: `lsof -i :8000` to see what's using the port

**Problem**: CUDA errors
- **Solution**: The model defaults to CPU. Install PyTorch with CUDA support if needed
- **Check**: `torch.cuda.is_available()` in Python

**Problem**: Import errors
- **Solution**: Ensure you're running from the backend directory or using Docker
- **Check**: Python path includes backend directory

**Problem**: Training fails with "Dataset not found"
- **Solution**: Generate dataset first using `/dataset/generate` endpoint
- **Check**: `backend/data/dataset.json` exists

### Frontend Issues

**Problem**: API connection failed
- **Solution**: Ensure backend is running and check `VITE_API_URL` in `.env`
- **Check**: Backend health endpoint: `curl http://localhost:8000/health`

**Problem**: Build errors
- **Solution**: Run `npm install` again and check Node.js version (requires 20+)
- **Check**: `node --version` and `npm --version`

**Problem**: SSE not working
- **Solution**: Check browser console for errors, ensure backend supports SSE
- **Check**: Network tab in browser dev tools

### Docker Issues

**Problem**: Build fails
- **Solution**: Ensure Docker has enough resources allocated (4GB+ RAM recommended)
- **Check**: `docker system df` to see disk usage

**Problem**: Volume permissions
- **Solution**: Check file permissions for `models/`, `logs/`, and `data/` directories
- **Check**: `ls -la backend/models/`

**Problem**: Container won't start
- **Solution**: Check logs: `docker-compose logs backend` or `docker-compose logs frontend`
- **Check**: Port conflicts: `netstat -an | grep 8000` or `netstat -an | grep 3000`

**Problem**: Hot-reload not working
- **Solution**: Ensure volumes are mounted correctly in `docker-compose.yml`
- **Check**: File changes should trigger reload in container logs

### Training Issues

**Problem**: Training loss not decreasing
- **Solution**: Try adjusting learning rate, increase dataset size, or train for more epochs
- **Check**: Review loss plots in `backend/logs/loss.png`

**Problem**: Out of memory
- **Solution**: Reduce batch size or dataset size
- **Check**: Monitor memory usage during training

**Problem**: Training progress not updating
- **Solution**: Check SSE connection in browser dev tools
- **Check**: Backend logs for errors

## 📝 Example curl Commands

### Complete Workflow

```bash
# 1. Check health
curl http://localhost:8000/health

# 2. Generate dataset
curl -X POST http://localhost:8000/dataset/generate \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 10000}'

# 3. Check dataset info
curl http://localhost:8000/dataset/info

# 4. Start training
curl -X POST http://localhost:8000/train/start \
  -H "Content-Type: application/json" \
  -d '{"epochs": 50, "batch_size": 64}'

# 5. Check training status
curl http://localhost:8000/train/status

# 6. Get model info
curl http://localhost:8000/model/info

# 7. Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"n": 2021}'
```

## 🎨 Frontend Components

### NumberInput
Input component for entering numbers to factorize.

### ResultCard
Displays prediction results with confidence metrics.

### LoaderSpinner
Reusable loading spinner component.

### TrainingProgress
Real-time training progress with:
- SSE integration
- Loss visualization (Chart.js)
- Current epoch display
- Train/validation loss metrics

## 🔐 Environment Variables

### Backend
- `API_HOST`: API host (default: `0.0.0.0`)
- `API_PORT`: API port (default: `8000`)
- `CORS_ORIGINS`: Comma-separated list of allowed origins (default: `*`)

### Frontend
- `VITE_API_URL`: Backend API URL (default: `http://localhost:8000`)

## 📚 Additional Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **PyTorch Docs**: https://pytorch.org/docs/
- **React Docs**: https://react.dev/
- **Vite Docs**: https://vitejs.dev/
- **Chart.js Docs**: https://www.chartjs.org/

## 🤝 Contributing

This is an experimental project for educational purposes. Feel free to:
- Experiment with different model architectures
- Try different training strategies
- Improve the UI/UX
- Add new features

## 📄 License

This is an experimental project for educational purposes.

---

**Built with**: Python 3.10+, FastAPI, PyTorch, React, TypeScript, Vite, TailwindCSS, Docker
