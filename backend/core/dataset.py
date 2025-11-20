"""
Dataset generation and factorization utilities.
"""
import json
import random
import math
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from config import DatasetConfig, ModelConfig
from core.model import encode_number
import torch
from torch.utils.data import Dataset


def factorize(n: int) -> Optional[Tuple[int, int]]:
    """
    Find two factors of n using trial division.
    
    Args:
        n: Integer to factorize
        
    Returns:
        Tuple of (factor_a, factor_b) if found, None otherwise
    """
    if n < 2:
        return None
    
    # Check for small factors first
    if n % 2 == 0:
        return (2, n // 2)
    
    # Try factors up to sqrt(n)
    sqrt_n = int(math.sqrt(n))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return (i, n // i)
    
    # If no factors found, it's prime
    return None


def generate_dataset(
    num_samples: int = DatasetConfig.DEFAULT_NUM_SAMPLES,
    min_val: int = DatasetConfig.DEFAULT_MIN_VAL,
    max_val: int = DatasetConfig.DEFAULT_MAX_VAL
) -> List[Dict]:
    """
    Generate a dataset of numbers and their factors.
    
    Args:
        num_samples: Number of samples to generate
        min_val: Minimum number value
        max_val: Maximum number value
        
    Returns:
        List of dictionaries with 'n', 'factor_a', 'factor_b' keys
    """
    dataset = []
    attempts = 0
    max_attempts = num_samples * 10
    
    while len(dataset) < num_samples and attempts < max_attempts:
        attempts += 1
        n = random.randint(min_val, max_val)
        
        factors = factorize(n)
        if factors:
            factor_a, factor_b = factors
            dataset.append({
                'n': n,
                'factor_a': factor_a,
                'factor_b': factor_b
            })
    
    return dataset


def save_dataset(dataset: List[Dict], filepath: str = None):
    """Save dataset to JSON file."""
    if filepath is None:
        filepath = DatasetConfig.DEFAULT_DATASET_PATH
    
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(dataset, f, indent=2)


def load_dataset(filepath: str = None) -> List[Dict]:
    """Load dataset from JSON file."""
    if filepath is None:
        filepath = DatasetConfig.DEFAULT_DATASET_PATH
    
    path = Path(filepath)
    if not path.exists():
        return []
    
    with open(path, 'r') as f:
        return json.load(f)


class FactorizationDataset(Dataset):
    """
    PyTorch Dataset for factorization data.
    """
    
    def __init__(self, data: List[Dict], max_digits: int = ModelConfig.MAX_DIGITS):
        """
        Args:
            data: List of dicts with 'n', 'factor_a', 'factor_b' keys
            max_digits: Maximum number of digits to encode
        """
        self.data = data
        self.max_digits = max_digits
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (input_tensor, target_tensor)
            - input_tensor: Encoded number (max_digits,)
            - target_tensor: Factors (2,)
        """
        sample = self.data[idx]
        n = sample['n']
        factor_a = sample['factor_a']
        factor_b = sample['factor_b']
        
        # Encode input number
        input_tensor = encode_number(n, self.max_digits)
        
        # Create target tensor
        target_tensor = torch.tensor([float(factor_a), float(factor_b)], dtype=torch.float32)
        
        return input_tensor, target_tensor


if __name__ == "__main__":
    # Generate and save default dataset
    print("Generating dataset...")
    dataset = generate_dataset(num_samples=10000)
    save_dataset(dataset)
    print(f"Generated {len(dataset)} samples")
    print(f"Example: {dataset[0]}")

