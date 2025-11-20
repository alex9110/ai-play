"""
Training pipeline for the factorization model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable
from pathlib import Path
import json
import csv
from datetime import datetime
import matplotlib.pyplot as plt

from config import ModelConfig, TrainingConfig, MODELS_DIR, LOGS_DIR
from core.model import FactorizationMLP, get_model_version


class Trainer:
    """
    Trainer class for the factorization model.
    """
    
    def __init__(
        self,
        model: FactorizationMLP,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = ModelConfig.DEFAULT_LEARNING_RATE,
        checkpoint_dir: Path = TrainingConfig.CHECKPOINT_DIR,
        log_dir: Path = TrainingConfig.LOG_DIR,
        progress_callback: Optional[Callable[[Dict], None]] = None
    ):
        """
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate for optimizer
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            progress_callback: Optional callback function for progress updates
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.progress_callback = progress_callback
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }
        
        # Best validation loss
        self.best_val_loss = float('inf')
        self.current_version = get_model_version()
        
        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
            'best_val_loss': self.best_val_loss,
            'version': self.current_version
        }
        
        # Save epoch checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / f"model_v{self.current_version}.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.current_version = checkpoint.get('version', self.current_version)
        
        # Restore history if available
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        return checkpoint.get('epoch', 0)
    
    def save_logs(self, epoch: int):
        """Save training logs to CSV and JSON."""
        timestamp = datetime.now().isoformat()
        
        # CSV log
        csv_path = self.log_dir / "training_log.csv"
        file_exists = csv_path.exists()
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'epoch', 'train_loss', 'val_loss'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'timestamp': timestamp,
                'epoch': epoch,
                'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
                'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None
            })
        
        # JSON log (full history)
        json_path = self.log_dir / "training_log.json"
        log_data = {
            'version': self.current_version,
            'last_updated': timestamp,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.history['epochs'])
        }
        
        with open(json_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def plot_loss(self):
        """Plot and save loss curves."""
        if len(self.history['epochs']) == 0:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['epochs'], self.history['train_loss'], label='Train Loss', marker='o')
        plt.plot(self.history['epochs'], self.history['val_loss'], label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plot_path = self.log_dir / "loss.png"
        plt.savefig(plot_path)
        plt.close()
    
    def train(
        self,
        num_epochs: int,
        resume_from: Optional[str] = None,
        start_epoch: int = 0
    ) -> Dict:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
            start_epoch: Starting epoch number (if resuming)
            
        Returns:
            Dictionary with training metrics
        """
        # Load checkpoint if resuming
        if resume_from and Path(resume_from).exists():
            start_epoch = self.load_checkpoint(resume_from)
            print(f"Resuming training from epoch {start_epoch}")
        
        # Training loop
        for epoch in range(start_epoch, start_epoch + num_epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update history
            self.history['epochs'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            if TrainingConfig.SAVE_EVERY_EPOCH:
                self.save_checkpoint(epoch + 1, is_best=is_best)
            
            # Save logs
            self.save_logs(epoch + 1)
            
            # Plot loss
            if TrainingConfig.PLOT_LOSS:
                self.plot_loss()
            
            # Progress callback
            if self.progress_callback:
                self.progress_callback({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'is_best': is_best
                })
            
            print(f"Epoch {epoch + 1}/{start_epoch + num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Final metrics
        metrics = {
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.history['epochs']),
            'model_version': self.current_version
        }
        
        return metrics

