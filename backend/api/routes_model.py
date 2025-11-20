"""
Model information API routes.
"""
from fastapi import APIRouter, HTTPException
from core.inference import get_model_info, get_model_weights_stats, delete_model
from typing import Optional
from pydantic import BaseModel

router = APIRouter(prefix="/model", tags=["model"])


class DeleteModelRequest(BaseModel):
    model_path: Optional[str] = None


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


@router.delete("/delete")
async def delete_model_endpoint(model_path: Optional[str] = None):
    """
    Delete a model file.
    If model_path is not provided, deletes the latest model.
    """
    try:
        result = delete_model(model_path)
        if not result.get('success', False):
            raise HTTPException(status_code=400, detail=result.get('message', 'Failed to delete model'))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")

