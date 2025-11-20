"""
Model information API routes.
"""
from fastapi import APIRouter
from core.inference import get_model_info

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info")
async def get_model_info_endpoint():
    """
    Get information about the current model.
    Returns model size, version, last training time, etc.
    """
    return get_model_info()

