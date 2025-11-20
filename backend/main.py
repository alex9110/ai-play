"""
FastAPI main application with router mounts.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes_predict import router as predict_router
from api.routes_training import router as training_router
from api.routes_dataset import router as dataset_router
from api.routes_model import router as model_router
from config import APIConfig

app = FastAPI(
    title="Integer Factorization ML Service",
    description="Full-stack ML service for integer factorization using PyTorch",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=APIConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(predict_router)
app.include_router(training_router)
app.include_router(dataset_router)
app.include_router(model_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "factorization-ml"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        reload=True
    )
