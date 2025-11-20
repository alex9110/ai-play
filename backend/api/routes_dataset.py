"""
Dataset generation API routes.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from core.dataset import generate_dataset, save_dataset, load_dataset
from config import DatasetConfig

router = APIRouter(prefix="/dataset", tags=["dataset"])


class GenerateDatasetRequest(BaseModel):
    num_samples: int = Field(
        default=DatasetConfig.DEFAULT_NUM_SAMPLES,
        ge=100,
        le=100000,
        description="Number of samples to generate"
    )
    min_val: int = Field(
        default=DatasetConfig.DEFAULT_MIN_VAL,
        ge=1,
        description="Minimum number value"
    )
    max_val: int = Field(
        default=DatasetConfig.DEFAULT_MAX_VAL,
        ge=1,
        description="Maximum number value"
    )
    dataset_path: Optional[str] = Field(
        default=None,
        description="Path to save dataset (default: data/dataset.json)"
    )


class GenerateDatasetResponse(BaseModel):
    message: str
    num_samples: int
    min_val: int
    max_val: int
    dataset_path: str


@router.post("/generate", response_model=GenerateDatasetResponse)
async def generate_dataset_endpoint(request: GenerateDatasetRequest):
    """
    Generate a new dataset of numbers and their factors.
    """
    # Validation: min_val must be <= max_val
    if request.min_val > request.max_val:
        raise HTTPException(
            status_code=400,
            detail="min_val must be less than or equal to max_val"
        )
    
    try:
        dataset = generate_dataset(
            num_samples=request.num_samples,
            min_val=request.min_val,
            max_val=request.max_val
        )
        
        if request.dataset_path is None:
            dataset_path = DatasetConfig.DEFAULT_DATASET_PATH
        else:
            dataset_path = request.dataset_path
        
        save_dataset(dataset, dataset_path)
        
        return GenerateDatasetResponse(
            message=f"Generated {len(dataset)} samples",
            num_samples=len(dataset),
            min_val=request.min_val,
            max_val=request.max_val,
            dataset_path=str(dataset_path)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset generation error: {str(e)}"
        )


@router.get("/info")
async def get_dataset_info():
    """
    Get information about the current dataset.
    """
    try:
        dataset = load_dataset()
        
        if not dataset:
            return {
                'exists': False,
                'message': 'No dataset found. Please generate a dataset first.'
            }
        
        # Calculate statistics
        numbers = [sample['n'] for sample in dataset]
        factors_a = [sample['factor_a'] for sample in dataset]
        factors_b = [sample['factor_b'] for sample in dataset]
        
        return {
            'exists': True,
            'num_samples': len(dataset),
            'min_number': min(numbers),
            'max_number': max(numbers),
            'avg_number': sum(numbers) / len(numbers),
            'min_factor_a': min(factors_a),
            'max_factor_a': max(factors_a),
            'min_factor_b': min(factors_b),
            'max_factor_b': max(factors_b),
            'example': dataset[0] if dataset else None
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting dataset info: {str(e)}"
        )

