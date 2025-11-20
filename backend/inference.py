"""
Inference utilities for loading and using the trained model.
"""
import torch
from pathlib import Path
from typing import Optional, Tuple

from model import FactorizationMLP, encode_number, decode_factors


def load_model(checkpoint_path: str = "models/model.pt", device: Optional[torch.device] = None) -> FactorizationMLP:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on (default: auto-detect)
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FactorizationMLP(input_dim=40, hidden_dims=[128, 256, 128], dropout=0.2)
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(device)
    else:
        # If no checkpoint exists, return untrained model
        model = model.to(device)
        model.eval()
    
    return model


def predict(model: FactorizationMLP, n: int, device: Optional[torch.device] = None) -> Tuple[int, int, list]:
    """
    Predict factors for a given integer.
    
    Args:
        model: Trained model
        n: Integer to factorize
        device: Device to run inference on
        
    Returns:
        Tuple of (factor_a, factor_b, raw_prediction)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    with torch.no_grad():
        # Encode input
        x = encode_number(n).unsqueeze(0).to(device)
        
        # Predict
        prediction = model(x)
        
        # Decode
        raw = prediction[0].cpu().tolist()
        factor_a, factor_b = decode_factors(prediction[0])
        
        return factor_a, factor_b, raw

