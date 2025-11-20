"""
Model inference utilities.
"""
import torch
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from datetime import datetime

from config import ModelConfig, MODELS_DIR, DatasetConfig
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
    num_classes = DatasetConfig.get_num_classes()
    model = FactorizationMLP(
        vocab_size=ModelConfig.VOCAB_SIZE,
        embedding_dim=ModelConfig.EMBEDDING_DIM,
        hidden_dims=ModelConfig.HIDDEN_DIMS,
        dropout=ModelConfig.DROPOUT,
        max_digits=ModelConfig.MAX_DIGITS,
        num_classes=num_classes
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
) -> Tuple[int, int, List[float], List[float]]:
    """
    Predict factors for a given integer using classification.
    
    Args:
        model: Trained model
        n: Integer to factorize
        device: Device to run inference on
        
    Returns:
        Tuple of (factor_a, factor_b, logits, probabilities)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Encode input
    input_tensor = encode_number(n).unsqueeze(0).to(device)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        logits = model(input_tensor)  # (1, num_classes)
        logits_1d = logits[0]  # Remove batch dimension
        
        # Get probabilities
        from core.model import get_class_probabilities
        probabilities = get_class_probabilities(logits_1d)
        
        logits_list = logits_1d.cpu().tolist()
        probabilities_list = probabilities.cpu().tolist()
    
    # Decode factors
    factor_a, factor_b = decode_factors(logits_1d, n)
    
    return factor_a, factor_b, logits_list, probabilities_list


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
        state_dict = checkpoint.get('model_state_dict', {})
        
        # Detect if this is an old regression model by checking output layer size
        # Old models have output layer with shape [2, hidden_dim] (out_features=2)
        # New models have output layer with shape [num_classes, hidden_dim] (out_features=num_classes)
        is_old_model = False
        num_classes = DatasetConfig.get_num_classes()
        
        # Find the output layer (last Linear layer in network Sequential)
        # Sequential layers are numbered: 0=Linear, 1=ReLU, 2=Dropout, 3=Linear, etc.
        # The last Linear layer will be the output layer
        output_layer_keys = [k for k in state_dict.keys() 
                           if 'network' in k and 'weight' in k and 'embedding' not in k]
        if output_layer_keys:
            # Find the layer with highest index (last in Sequential)
            # Keys are like 'network.0.weight', 'network.3.weight', etc.
            def get_layer_index(key):
                try:
                    # Extract number after 'network.'
                    parts = key.split('.')
                    if len(parts) >= 2:
                        return int(parts[1])
                except:
                    return -1
                return -1
            
            output_layer_keys.sort(key=get_layer_index, reverse=True)
            last_layer_key = output_layer_keys[0]
            output_shape = state_dict[last_layer_key].shape
            # For Linear layer: weight shape is [out_features, in_features]
            # Check if output dimension is 2 (old regression) instead of num_classes
            if len(output_shape) >= 2:
                out_features = output_shape[0]
                if out_features == 2:
                    is_old_model = True
                    print(f"[DEBUG] Detected old model: output layer size is {out_features}, expected {num_classes}")
                elif out_features == num_classes:
                    is_old_model = False
                    print(f"[DEBUG] Detected new model: output layer size is {out_features} (correct)")
                else:
                    # Unexpected size
                    print(f"[WARNING] Model output layer has unexpected size: {out_features}, expected {num_classes}")
        
        # Calculate model size
        num_classes = DatasetConfig.get_num_classes()
        model = FactorizationMLP(num_classes=num_classes)
        
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            # Model architecture mismatch - likely old regression model
            # Still return basic info from checkpoint
            file_size = path.stat().st_size
            mod_time = datetime.fromtimestamp(path.stat().st_mtime)
            error_msg = str(e)
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            
            return {
                'exists': True,
                'path': str(path),
                'version': checkpoint.get('version', 'unknown'),
                'epoch': checkpoint.get('epoch', 'unknown'),
                'best_val_loss': checkpoint.get('best_val_loss', 'unknown'),
                'file_size_bytes': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'last_trained': mod_time.isoformat(),
                'error': 'Model architecture mismatch (old regression model). Please train a new model.',
                'is_old_model': True,
                'model_config': {
                    'vocab_size': ModelConfig.VOCAB_SIZE,
                    'embedding_dim': ModelConfig.EMBEDDING_DIM,
                    'hidden_dims': ModelConfig.HIDDEN_DIMS,
                    'max_digits': ModelConfig.MAX_DIGITS
                }
            }
        
        # Double-check: verify the actual loaded model's output layer size
        # This catches cases where strict=False allowed partial loading
        actual_output_size = None
        for name, param in model.named_parameters():
            if 'network' in name and 'weight' in name and 'embedding' not in name:
                # Find the last Linear layer by checking all network layers
                parts = name.split('.')
                if len(parts) >= 2:
                    try:
                        layer_idx = int(parts[1])
                        # The last Linear layer will have the highest index that's a multiple of 3
                        # (Linear layers are at indices 0, 3, 6, 9, ...)
                        if layer_idx % 3 == 0:
                            if actual_output_size is None or layer_idx > actual_output_size[1]:
                                actual_output_size = (param.shape[0], layer_idx)
                    except:
                        pass
        
        # If we found the output layer, check its size
        if actual_output_size is not None:
            actual_out_features = actual_output_size[0]
            if actual_out_features == 2:
                # Definitely an old model
                is_old_model = True
                print(f"[DEBUG] Confirmed old model after loading: output size is {actual_out_features}")
            elif actual_out_features == num_classes:
                # Definitely a new model - override any previous detection
                is_old_model = False
                print(f"[DEBUG] Confirmed new model after loading: output size is {actual_out_features} (correct)")
        
        # If detected as old model, return error
        if is_old_model:
            file_size = path.stat().st_size
            mod_time = datetime.fromtimestamp(path.stat().st_mtime)
            return {
                'exists': True,
                'path': str(path),
                'version': checkpoint.get('version', 'unknown'),
                'epoch': checkpoint.get('epoch', 'unknown'),
                'best_val_loss': checkpoint.get('best_val_loss', 'unknown'),
                'file_size_bytes': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'last_trained': mod_time.isoformat(),
                'error': 'Model architecture mismatch (old regression model). Please train a new model.',
                'is_old_model': True,
                'model_config': {
                    'vocab_size': ModelConfig.VOCAB_SIZE,
                    'embedding_dim': ModelConfig.EMBEDDING_DIM,
                    'hidden_dims': ModelConfig.HIDDEN_DIMS,
                    'max_digits': ModelConfig.MAX_DIGITS
                }
            }
        
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
            'is_old_model': False,  # Explicitly mark as new model
            'model_config': {
                'vocab_size': ModelConfig.VOCAB_SIZE,
                'embedding_dim': ModelConfig.EMBEDDING_DIM,
                'hidden_dims': ModelConfig.HIDDEN_DIMS,
                'max_digits': ModelConfig.MAX_DIGITS
            }
        }
    except Exception as e:
        import traceback
        return {
            'exists': True,
            'path': str(path),
            'error': f'Failed to load model: {str(e)}'
        }


def delete_model(model_path: Optional[str] = None) -> Dict:
    """
    Delete a model file.
    
    Args:
        model_path: Path to model. If None, deletes latest model.
        
    Returns:
        Dictionary with deletion status
    """
    if model_path is None:
        model_path = get_latest_model_path()
        if model_path is None:
            return {
                'success': False,
                'message': 'No model found to delete'
            }
    
    path = Path(model_path)
    if not path.exists():
        return {
            'success': False,
            'message': 'Model file not found'
        }
    
    try:
        # Also try to delete related checkpoint files if they exist
        model_name = path.stem  # e.g., "model_v1"
        model_dir = path.parent
        
        # Extract version number from model name (e.g., "model_v1" -> 1)
        model_version = None
        try:
            if '_v' in model_name:
                model_version = int(model_name.split('_v')[1])
        except (ValueError, IndexError):
            pass
        
        # Delete the main model file
        path.unlink()
        deleted_files = [str(path)]
        
        # Try to delete related checkpoint files (checkpoint_epoch_*.pt)
        if model_version is not None:
            for checkpoint_file in model_dir.glob("checkpoint_epoch_*.pt"):
                # Check if checkpoint belongs to this model version
                try:
                    checkpoint = torch.load(checkpoint_file, map_location='cpu')
                    checkpoint_version = checkpoint.get('version', None)
                    if checkpoint_version == model_version:
                        checkpoint_file.unlink()
                        deleted_files.append(str(checkpoint_file))
                except Exception:
                    # If we can't read checkpoint, skip it
                    pass
        
        return {
            'success': True,
            'message': f'Model deleted successfully',
            'deleted_files': deleted_files
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'Failed to delete model: {str(e)}',
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
        state_dict = checkpoint.get('model_state_dict', {})
        
        # Detect if this is an old regression model by checking output layer size
        # Old models have output layer with shape [hidden_dim, 2]
        # New models have output layer with shape [hidden_dim, num_classes]
        is_old_model = False
        num_classes = DatasetConfig.get_num_classes()
        
        # Find the output layer (last Linear layer in network Sequential)
        # Sequential layers are numbered: 0=Linear, 1=ReLU, 2=Dropout, 3=Linear, etc.
        # The last Linear layer will be the output layer
        output_layer_keys = [k for k in state_dict.keys() 
                           if 'network' in k and 'weight' in k and 'embedding' not in k]
        if output_layer_keys:
            # Find the layer with highest index (last in Sequential)
            # Keys are like 'network.0.weight', 'network.3.weight', etc.
            def get_layer_index(key):
                try:
                    # Extract number after 'network.'
                    parts = key.split('.')
                    if len(parts) >= 2:
                        return int(parts[1])
                except:
                    return -1
                return -1
            
            output_layer_keys.sort(key=get_layer_index, reverse=True)
            last_layer_key = output_layer_keys[0]
            output_shape = state_dict[last_layer_key].shape
            # Check if output dimension is 2 (old regression) instead of num_classes
            if len(output_shape) >= 2 and output_shape[0] == 2:
                is_old_model = True
        
        # Create model and load weights
        num_classes = DatasetConfig.get_num_classes()
        model = FactorizationMLP(num_classes=num_classes)
        
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            # Model architecture mismatch - likely old regression model
            error_msg = str(e)
            # Truncate long error messages
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            
            return {
                'exists': False,
                'message': 'This model was trained with the old regression architecture and is incompatible with the new classification system. Please train a new model.',
                'error': error_msg,
                'is_old_model': True
            }
        
        # Double-check: verify the actual loaded model's output layer size
        # This catches cases where strict=False allowed partial loading
        actual_output_size = None
        for name, param in model.named_parameters():
            if 'network' in name and 'weight' in name and 'embedding' not in name:
                # Find the last Linear layer by checking all network layers
                parts = name.split('.')
                if len(parts) >= 2:
                    try:
                        layer_idx = int(parts[1])
                        # The last Linear layer will have the highest index that's a multiple of 3
                        # (Linear layers are at indices 0, 3, 6, 9, ...)
                        if layer_idx % 3 == 0:
                            if actual_output_size is None or layer_idx > actual_output_size[1]:
                                actual_output_size = (param.shape[0], layer_idx)
                    except:
                        pass
        
        # If we found the output layer, check its size
        if actual_output_size is not None:
            actual_out_features = actual_output_size[0]
            if actual_out_features == 2:
                # Definitely an old model - override detection
                is_old_model = True
                print(f"[DEBUG] Confirmed old model after loading: output size is {actual_out_features}")
            elif actual_out_features == num_classes:
                # Definitely a new model - override any previous detection
                is_old_model = False
                print(f"[DEBUG] Confirmed new model after loading: output size is {actual_out_features} (correct)")
        
        # If detected as old model, return error
        if is_old_model:
            return {
                'exists': False,
                'message': 'This model was trained with the old regression architecture and is incompatible with the new classification system. Please train a new model.',
                'is_old_model': True
            }
        
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
        
        if not layers_stats:
            return {
                'exists': False,
                'message': 'No trainable parameters found in model'
            }
        
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
        import traceback
        error_details = traceback.format_exc()
        return {
            'exists': False,
            'message': f'Failed to load model statistics: {str(e)}',
            'error': str(e)
        }

