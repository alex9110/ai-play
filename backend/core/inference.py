"""
Model inference utilities.
"""
import torch
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from datetime import datetime

from config import ModelConfig, MODELS_DIR
from core.model import FactorizationMLP, encode_number, decode_factors, get_latest_model_path


def load_model(model_path: Optional[str] = None, device: torch.device = None) -> FactorizationMLP:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint. If None, loads latest version.
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_path is None:
        model_path = get_latest_model_path()
        if model_path is None:
            raise FileNotFoundError("No model found. Please train a model first.")
    
    # Create model instance
    model = FactorizationMLP(
        vocab_size=ModelConfig.VOCAB_SIZE,
        embedding_dim=ModelConfig.EMBEDDING_DIM,
        hidden_dims=ModelConfig.HIDDEN_DIMS,
        dropout=ModelConfig.DROPOUT,
        max_digits=ModelConfig.MAX_DIGITS
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def predict(
    model: FactorizationMLP,
    n: int,
    device: torch.device = None
) -> Tuple[int, int, List[float]]:
    """
    Predict factors for a given integer.
    
    Args:
        model: Trained model
        n: Integer to factorize
        device: Device to run inference on
        
    Returns:
        Tuple of (factor_a, factor_b, raw_output)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Encode input
    input_tensor = encode_number(n).unsqueeze(0).to(device)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        raw_output = output[0].cpu().tolist()  # Remove batch dimension
    
    # Decode factors
    factor_a, factor_b = decode_factors(output[0])
    
    return factor_a, factor_b, raw_output


def get_model_info(model_path: Optional[str] = None) -> Dict:
    """
    Get information about a model.
    
    Args:
        model_path: Path to model. If None, uses latest.
        
    Returns:
        Dictionary with model information
    """
    if model_path is None:
        model_path = get_latest_model_path()
        if model_path is None:
            return {
                'exists': False,
                'message': 'No model found'
            }
    
    path = Path(model_path)
    if not path.exists():
        return {
            'exists': False,
            'message': 'Model file not found'
        }
    
    # Load checkpoint to get metadata
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Calculate model size
        model = FactorizationMLP()
        model.load_state_dict(checkpoint['model_state_dict'])
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get file size
        file_size = path.stat().st_size
        
        # Get modification time
        mod_time = datetime.fromtimestamp(path.stat().st_mtime)
        
        return {
            'exists': True,
            'path': str(path),
            'version': checkpoint.get('version', 'unknown'),
            'epoch': checkpoint.get('epoch', 'unknown'),
            'best_val_loss': checkpoint.get('best_val_loss', 'unknown'),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'file_size_bytes': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'last_trained': mod_time.isoformat(),
            'model_config': {
                'vocab_size': ModelConfig.VOCAB_SIZE,
                'embedding_dim': ModelConfig.EMBEDDING_DIM,
                'hidden_dims': ModelConfig.HIDDEN_DIMS,
                'max_digits': ModelConfig.MAX_DIGITS
            }
        }
    except Exception as e:
        return {
            'exists': True,
            'path': str(path),
            'error': str(e)
        }


def get_model_weights_stats(model_path: Optional[str] = None) -> Dict:
    """
    Get statistics about model weights for visualization.
    
    Args:
        model_path: Path to model. If None, uses latest.
        
    Returns:
        Dictionary with weight statistics for each layer
    """
    if model_path is None:
        model_path = get_latest_model_path()
        if model_path is None:
            return {
                'exists': False,
                'message': 'No model found'
            }
    
    path = Path(model_path)
    if not path.exists():
        return {
            'exists': False,
            'message': 'Model file not found'
        }
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model and load weights
        model = FactorizationMLP()
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Collect statistics for each layer
        layers_stats = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                weights = param.data.cpu().flatten()
                
                stats = {
                    'name': name,
                    'shape': list(param.shape),
                    'num_params': param.numel(),
                    'mean': float(weights.mean().item()),
                    'std': float(weights.std().item()),
                    'min': float(weights.min().item()),
                    'max': float(weights.max().item()),
                    'median': float(torch.median(weights).item()),
                    'q25': float(torch.quantile(weights, 0.25).item()),
                    'q75': float(torch.quantile(weights, 0.75).item())
                }
                layers_stats.append(stats)
        
        return {
            'exists': True,
            'model_path': str(path),
            'version': checkpoint.get('version', 'unknown'),
            'epoch': checkpoint.get('epoch', 'unknown'),
            'layers': layers_stats,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    except Exception as e:
        return {
            'exists': False,
            'error': str(e)
        }

