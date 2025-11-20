"""
Configuration management for the factorization ML service.
"""
from pathlib import Path
from typing import Optional
import os
import math

# Base paths - handle both running from project root and backend directory
if Path(__file__).parent.name == "backend":
    # Running from project root
    BASE_DIR = Path(__file__).parent.parent
    BACKEND_DIR = BASE_DIR / "backend"
else:
    # Running from backend directory
    BACKEND_DIR = Path(__file__).parent
    BASE_DIR = BACKEND_DIR.parent

MODELS_DIR = BACKEND_DIR / "models"
LOGS_DIR = BACKEND_DIR / "logs"
DATASET_DIR = BACKEND_DIR / "data"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
class ModelConfig:
    """Model architecture configuration."""
    # Embedding configuration
    VOCAB_SIZE = 10  # Digits 0-9
    EMBEDDING_DIM = 32
    MAX_DIGITS = 10  # Support up to 10-digit numbers
    
    # Network architecture
    HIDDEN_DIMS = [256, 512, 256]
    DROPOUT = 0.2
    
    # Training defaults
    DEFAULT_EPOCHS = 50
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_VALIDATION_SPLIT = 0.2

# Dataset configuration
class DatasetConfig:
    """Dataset generation configuration."""
    DEFAULT_NUM_SAMPLES = 10000
    DEFAULT_MIN_VAL = 1000
    DEFAULT_MAX_VAL = 9999
    DEFAULT_DATASET_PATH = DATASET_DIR / "dataset.json"
    
    @staticmethod
    def get_max_factor(max_val: int = None) -> int:
        """Calculate maximum factor class index."""
        if max_val is None:
            max_val = DatasetConfig.DEFAULT_MAX_VAL
        return int(math.sqrt(max_val))
    
    @staticmethod
    def get_num_classes(max_val: int = None) -> int:
        """Calculate number of classification classes."""
        return DatasetConfig.get_max_factor(max_val)

# Training configuration
class TrainingConfig:
    """Training pipeline configuration."""
    CHECKPOINT_DIR = MODELS_DIR
    LOG_DIR = LOGS_DIR
    SAVE_EVERY_EPOCH = True
    PLOT_LOSS = True

# API configuration
class APIConfig:
    """API server configuration."""
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", 8000))
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

