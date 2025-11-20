"""
Training API routes with SSE support.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict
import torch
import json
import asyncio
import queue
from pathlib import Path

from core.model import FactorizationMLP, get_latest_model_path
from core.dataset import FactorizationDataset, load_dataset
from core.trainer import Trainer
from core.inference import get_model_info
from config import ModelConfig, DatasetConfig, TrainingConfig
from torch.utils.data import DataLoader

router = APIRouter(prefix="/train", tags=["training"])

# Global training state
_training_state = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'train_loss': 0.0,
    'val_loss': 0.0,
    'progress_callback_queue': None
}


class TrainRequest(BaseModel):
    epochs: int = Field(default=ModelConfig.DEFAULT_EPOCHS, ge=1, le=1000)
    batch_size: int = Field(default=ModelConfig.DEFAULT_BATCH_SIZE, ge=1, le=512)
    learning_rate: float = Field(default=ModelConfig.DEFAULT_LEARNING_RATE, gt=0)
    resume: bool = Field(default=False, description="Resume from latest checkpoint")
    dataset_path: Optional[str] = None


class TrainResponse(BaseModel):
    message: str
    training_started: bool


def progress_callback(metrics: Dict):
    """Callback function for training progress updates."""
    global _training_state
    _training_state.update({
        'current_epoch': metrics['epoch'],
        'train_loss': metrics['train_loss'],
        'val_loss': metrics['val_loss']
    })
    
    # Send to SSE queue if available (thread-safe queue.Queue)
    progress_queue = _training_state.get('progress_callback_queue')
    if progress_queue is not None:
        try:
            progress_queue.put_nowait(metrics)
        except queue.Full:
            # Queue is full, skip this update
            pass
        except AttributeError:
            # Queue doesn't have put_nowait (shouldn't happen with queue.Queue)
            pass


def training_worker(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    resume: bool,
    dataset_path: Optional[str]
):
    """Background worker for training (synchronous function)."""
    global _training_state
    
    try:
        print(f"[Training] Starting training worker: {epochs} epochs")
        _training_state['is_training'] = True
        _training_state['total_epochs'] = epochs
        _training_state['current_epoch'] = 0
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load dataset
        if dataset_path is None:
            dataset_path = DatasetConfig.DEFAULT_DATASET_PATH
        
        dataset = load_dataset(dataset_path)
        if not dataset:
            raise ValueError("Dataset not found. Please generate dataset first.")
        
        # Split dataset (80/20)
        split_idx = int((1 - ModelConfig.DEFAULT_VALIDATION_SPLIT) * len(dataset))
        train_data = dataset[:split_idx]
        val_data = dataset[split_idx:]
        
        # Create data loaders
        train_dataset = FactorizationDataset(train_data)
        val_dataset = FactorizationDataset(val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create or load model
        model = FactorizationMLP(
            vocab_size=ModelConfig.VOCAB_SIZE,
            embedding_dim=ModelConfig.EMBEDDING_DIM,
            hidden_dims=ModelConfig.HIDDEN_DIMS,
            dropout=ModelConfig.DROPOUT,
            max_digits=ModelConfig.MAX_DIGITS
        )
        
        # Resume from checkpoint if requested
        resume_from = None
        start_epoch = 0
        if resume:
            latest_model = get_latest_model_path()
            if latest_model:
                resume_from = latest_model
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=learning_rate,
            checkpoint_dir=TrainingConfig.CHECKPOINT_DIR,
            log_dir=TrainingConfig.LOG_DIR,
            progress_callback=progress_callback
        )
        
        # Train
        print(f"[Training] Beginning training loop...")
        metrics = trainer.train(
            num_epochs=epochs,
            resume_from=resume_from,
            start_epoch=start_epoch
        )
        
        print(f"[Training] Training completed successfully")
        _training_state['is_training'] = False
        _training_state['current_epoch'] = epochs
        
    except Exception as e:
        print(f"[Training] Error during training: {str(e)}")
        _training_state['is_training'] = False
        _training_state['error'] = str(e)
        import traceback
        traceback.print_exc()
        raise


@router.post("/start", response_model=TrainResponse)
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Start training the model. Training runs in background and progress can be
    streamed via GET /train/progress.
    """
    global _training_state
    
    if _training_state['is_training']:
        raise HTTPException(
            status_code=409,
            detail="Training is already in progress"
        )
    
    # Reset training state
    _training_state['current_epoch'] = 0
    _training_state['train_loss'] = 0.0
    _training_state['val_loss'] = 0.0
    _training_state['error'] = None
    
    # Create SSE queue for progress updates (use thread-safe queue.Queue)
    _training_state['progress_callback_queue'] = queue.Queue(maxsize=100)
    
    # Start training in background (use run_in_executor for sync function)
    import concurrent.futures
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        training_worker,
        request.epochs,
        request.batch_size,
        request.learning_rate,
        request.resume,
        request.dataset_path
    )
    
    return TrainResponse(
        message=f"Training started for {request.epochs} epochs",
        training_started=True
    )


@router.get("/progress")
async def get_training_progress():
    """
    Stream training progress via Server-Sent Events (SSE).
    """
    async def event_generator():
        global _training_state
        
        # Send initial connection message
        initial_data = {
            'status': 'connected',
            'is_training': _training_state['is_training'],
            'current_epoch': _training_state['current_epoch'],
            'total_epochs': _training_state['total_epochs']
        }
        yield f"data: {json.dumps(initial_data)}\n\n"
        
        # Ensure queue exists
        if _training_state['progress_callback_queue'] is None:
            _training_state['progress_callback_queue'] = queue.Queue(maxsize=100)
        
        progress_queue = _training_state['progress_callback_queue']
        
        while _training_state['is_training']:
            try:
                # Try to get progress update from queue
                # Use executor to run blocking queue.get in thread pool
                loop = asyncio.get_event_loop()
                try:
                    # Wait for item with timeout (runs in thread pool)
                    metrics = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: progress_queue.get(timeout=1.0)),
                        timeout=2.0
                    )
                    
                    data = {
                        'epoch': metrics['epoch'],
                        'train_loss': metrics['train_loss'],
                        'val_loss': metrics['val_loss'],
                        'is_best': metrics.get('is_best', False),
                        'total_epochs': _training_state['total_epochs']
                    }
                    
                    yield f"data: {json.dumps(data)}\n\n"
                    
                except (asyncio.TimeoutError, queue.Empty):
                    # No update available, send heartbeat
                    heartbeat_data = {
                        'status': 'training',
                        'current_epoch': _training_state['current_epoch'],
                        'total_epochs': _training_state['total_epochs']
                    }
                    yield f"data: {json.dumps(heartbeat_data)}\n\n"
                
            except Exception as e:
                error_data = {
                    'status': 'error',
                    'error': str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                break
        
        # Send final status
        yield f"data: {json.dumps({'status': 'completed'})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/status")
async def get_training_status():
    """Get current training status."""
    return {
        'is_training': _training_state['is_training'],
        'current_epoch': _training_state['current_epoch'],
        'total_epochs': _training_state['total_epochs'],
        'train_loss': _training_state.get('train_loss', 0.0),
        'val_loss': _training_state.get('val_loss', 0.0),
        'error': _training_state.get('error', None)
    }

