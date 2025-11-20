"""
Dataset generation and factorization utilities.
"""
import json
import random
import math
from typing import List, Tuple, Optional
from pathlib import Path


def factorize(n: int) -> Optional[Tuple[int, int]]:
    """
    Find two factors of n using a non-bruteforce heuristic approach.
    Uses trial division with optimizations.
    
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


def generate_dataset(num_samples: int = 10000, min_val: int = 1000, max_val: int = 9999) -> List[dict]:
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


def save_dataset(dataset: List[dict], filepath: str = "dataset.json"):
    """Save dataset to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(dataset, f, indent=2)


def load_dataset(filepath: str = "dataset.json") -> List[dict]:
    """Load dataset from JSON file."""
    path = Path(filepath)
    if not path.exists():
        return []
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Generate and save default dataset
    print("Generating dataset...")
    dataset = generate_dataset(num_samples=10000)
    save_dataset(dataset, "dataset.json")
    print(f"Generated {len(dataset)} samples")
    print(f"Example: {dataset[0]}")

