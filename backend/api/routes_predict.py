"""
Prediction API routes.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import torch

from core.inference import load_model, predict
from config import ModelConfig

router = APIRouter(prefix="/predict", tags=["prediction"])

# Global model instance
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model():
    """Lazy load model."""
    global _model
    if _model is None:
        try:
            _model = load_model(device=_device)
        except FileNotFoundError:
            _model = None
    return _model


class PredictRequest(BaseModel):
    n: int = Field(..., ge=1, description="Integer to factorize")


class PredictResponse(BaseModel):
    n: int
    factorA: int
    factorB: int
    logits: list[float]
    probabilities: list[float]


@router.post("", response_model=PredictResponse)
async def predict_factors(request: PredictRequest):
    """
    Predict factors for a given integer.
    """
    try:
        model = get_model()
        if model is None:
            raise HTTPException(
                status_code=404,
                detail="Model not found. Please train a model first."
            )
        
        factor_a, factor_b, logits, probabilities = predict(model, request.n, _device)
        
        return PredictResponse(
            n=request.n,
            factorA=factor_a,
            factorB=factor_b,
            logits=logits,
            probabilities=probabilities
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

