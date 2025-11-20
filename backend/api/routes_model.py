"""
Model information API routes.
"""
from fastapi import APIRouter
from core.inference import get_model_info, get_model_weights_stats
from typing import Optional

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info")
async def get_model_info_endpoint():
    """
    Get information about the current model.
    Returns model size, version, last training time, etc.
    """
    return get_model_info()


@router.get("/weights/stats")
async def get_model_weights_stats_endpoint(model_path: Optional[str] = None):
    """
    Get statistics about model weights for visualization.
    Returns distribution statistics for each layer.
    """
    return get_model_weights_stats(model_path)

