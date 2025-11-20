"""
PyTorch model for integer factorization prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizationMLP(nn.Module):
    """
    Multi-Layer Perceptron for predicting factors of an integer.
    
    Input: Integer encoded as one-hot digits or embedding
    Output: Two floats representing the predicted factors
    """
    
    def __init__(self, input_dim: int = 40, hidden_dims: list = [128, 256, 128], dropout: float = 0.2):
        """
        Args:
            input_dim: Dimension of input encoding (4 digits * 10 possible values = 40 for one-hot)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(FactorizationMLP, self).__init__()
        
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
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, 2) with predicted factors
        """
        return self.network(x)


def encode_number(n: int, max_digits: int = 4) -> torch.Tensor:
    """
    Encode an integer as one-hot digits.
    
    Args:
        n: Integer to encode
        max_digits: Maximum number of digits to encode
        
    Returns:
        One-hot encoded tensor of shape (max_digits * 10,)
    """
    digits = []
    num_str = str(n).zfill(max_digits)
    
    for digit_char in num_str:
        digit = int(digit_char)
        one_hot = [0.0] * 10
        one_hot[digit] = 1.0
        digits.extend(one_hot)
    
    return torch.tensor(digits, dtype=torch.float32)


def decode_factors(prediction: torch.Tensor) -> tuple:
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

