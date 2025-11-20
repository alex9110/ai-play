"""
PyTorch model for integer factorization prediction using embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from config import ModelConfig


class FactorizationMLP(nn.Module):
    """
    Multi-Layer Perceptron for predicting factors of an integer.
    Uses embeddings instead of one-hot encoding for better generalization.
    
    Input: Sequence of digits (dynamic size support)
    Output: Two floats representing the predicted factors
    """
    
    def __init__(
        self,
        vocab_size: int = ModelConfig.VOCAB_SIZE,
        embedding_dim: int = ModelConfig.EMBEDDING_DIM,
        hidden_dims: List[int] = None,
        dropout: float = ModelConfig.DROPOUT,
        max_digits: int = ModelConfig.MAX_DIGITS
    ):
        """
        Args:
            vocab_size: Number of unique digits (10 for 0-9)
            embedding_dim: Dimension of digit embeddings
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            max_digits: Maximum number of digits to support
        """
        super(FactorizationMLP, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = ModelConfig.HIDDEN_DIMS
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_digits = max_digits
        
        # Embedding layer for digits
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Build MLP layers
        # Input: max_digits * embedding_dim (flattened embeddings)
        input_dim = max_digits * embedding_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer: 2 factors
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, max_digits) with digit indices
            
        Returns:
            Tensor of shape (batch_size, 2) with predicted factors
        """
        # x shape: (batch_size, max_digits)
        # Embed digits
        embedded = self.embedding(x)  # (batch_size, max_digits, embedding_dim)
        
        # Flatten embeddings
        batch_size = embedded.size(0)
        flattened = embedded.view(batch_size, -1)  # (batch_size, max_digits * embedding_dim)
        
        # Pass through MLP
        output = self.network(flattened)  # (batch_size, 2)
        
        return output


def encode_number(n: int, max_digits: int = ModelConfig.MAX_DIGITS) -> torch.Tensor:
    """
    Encode an integer as a sequence of digit indices.
    
    Args:
        n: Integer to encode
        max_digits: Maximum number of digits to encode
        
    Returns:
        Tensor of shape (max_digits,) with digit indices (left-padded with 0)
    """
    num_str = str(n).zfill(max_digits)
    digits = [int(digit_char) for digit_char in num_str]
    return torch.tensor(digits, dtype=torch.long)


def decode_factors(prediction: torch.Tensor) -> Tuple[int, int]:
    """
    Decode model prediction to integer factors.
    
    Args:
        prediction: Tensor of shape (2,) with predicted factor values
        
    Returns:
        Tuple of (factor_a, factor_b) as integers
    """
    factor_a = max(1, int(round(prediction[0].item())))
    factor_b = max(1, int(round(prediction[1].item())))
    return factor_a, factor_b


def get_model_version() -> int:
    """
    Get the next model version number.
    
    Returns:
        Next version number (e.g., if model_v1.pt exists, returns 2)
    """
    from config import MODELS_DIR
    
    existing_versions = []
    for model_file in MODELS_DIR.glob("model_v*.pt"):
        try:
            version = int(model_file.stem.split("_v")[1])
            existing_versions.append(version)
        except (ValueError, IndexError):
            continue
    
    if not existing_versions:
        return 1
    
    return max(existing_versions) + 1


def get_latest_model_path() -> str:
    """
    Get the path to the latest model version.
    
    Returns:
        Path to latest model, or None if no models exist
    """
    from config import MODELS_DIR
    
    existing_versions = []
    for model_file in MODELS_DIR.glob("model_v*.pt"):
        try:
            version = int(model_file.stem.split("_v")[1])
            existing_versions.append((version, model_file))
        except (ValueError, IndexError):
            continue
    
    if not existing_versions:
        return None
    
    latest = max(existing_versions, key=lambda x: x[0])
    return str(latest[1])

