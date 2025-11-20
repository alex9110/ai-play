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
        progress_callback: Optional[Callable[[Dict], None]] = None,
        alpha: float = 0.05
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
            alpha: Weight for product consistency loss (default: 0.05)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.progress_callback = progress_callback
        self.alpha = alpha
        
        # Loss functions
        self.criterion = nn.MSELoss()  # For factor loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {
            'train_loss_factors': [],
            'train_loss_product': [],
            'train_loss_total': [],
            'val_loss_factors': [],
            'val_loss_product': [],
            'val_loss_total': [],
            'epochs': []
        }
        
        # Best validation loss
        self.best_val_loss = float('inf')
        self.current_version = get_model_version()
        
        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with loss_factors, loss_product, loss_total
        """
        self.model.train()
        total_loss_factors = 0.0
        total_loss_product = 0.0
        total_loss_total = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            # Handle both old format (inputs, targets) and new format (inputs, targets, n_values)
            if len(batch) == 2:
                inputs, targets = batch
                n_values = None
            else:
                inputs, targets, n_values = batch
                n_values = n_values.to(self.device)
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Factor loss: MSE between predicted and true factors
            loss_factors = self.criterion(outputs, targets)
            
            # Product loss: MSE between predicted product and actual n
            if n_values is not None:
                # Compute predicted product: a_pred * b_pred
                product_pred = outputs[:, 0] * outputs[:, 1]
                # Reshape n_values to match batch size if needed
                if n_values.dim() == 0:
                    n_values = n_values.unsqueeze(0).expand(outputs.size(0))
                elif n_values.size(0) != outputs.size(0):
                    n_values = n_values[:outputs.size(0)]
                loss_product = self.criterion(product_pred, n_values)
            else:
                # Fallback: compute n from targets for backward compatibility
                n_computed = targets[:, 0] * targets[:, 1]
                product_pred = outputs[:, 0] * outputs[:, 1]
                loss_product = self.criterion(product_pred, n_computed)
            
            # Composite loss
            loss_total = loss_factors + self.alpha * loss_product
            
            # Backward pass
            loss_total.backward()
            self.optimizer.step()
            
            total_loss_factors += loss_factors.item()
            total_loss_product += loss_product.item()
            total_loss_total += loss_total.item()
            num_batches += 1
        
        return {
            'loss_factors': total_loss_factors / num_batches if num_batches > 0 else 0.0,
            'loss_product': total_loss_product / num_batches if num_batches > 0 else 0.0,
            'loss_total': total_loss_total / num_batches if num_batches > 0 else 0.0
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary with loss_factors, loss_product, loss_total
        """
        self.model.eval()
        total_loss_factors = 0.0
        total_loss_product = 0.0
        total_loss_total = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Handle both old format (inputs, targets) and new format (inputs, targets, n_values)
                if len(batch) == 2:
                    inputs, targets = batch
                    n_values = None
                else:
                    inputs, targets, n_values = batch
                    n_values = n_values.to(self.device)
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                
                # Factor loss: MSE between predicted and true factors
                loss_factors = self.criterion(outputs, targets)
                
                # Product loss: MSE between predicted product and actual n
                if n_values is not None:
                    # Compute predicted product: a_pred * b_pred
                    product_pred = outputs[:, 0] * outputs[:, 1]
                    # Reshape n_values to match batch size if needed
                    if n_values.dim() == 0:
                        n_values = n_values.unsqueeze(0).expand(outputs.size(0))
                    elif n_values.size(0) != outputs.size(0):
                        n_values = n_values[:outputs.size(0)]
                    loss_product = self.criterion(product_pred, n_values)
                else:
                    # Fallback: compute n from targets for backward compatibility
                    n_computed = targets[:, 0] * targets[:, 1]
                    product_pred = outputs[:, 0] * outputs[:, 1]
                    loss_product = self.criterion(product_pred, n_computed)
                
                # Composite loss
                loss_total = loss_factors + self.alpha * loss_product
                
                total_loss_factors += loss_factors.item()
                total_loss_product += loss_product.item()
                total_loss_total += loss_total.item()
                num_batches += 1
        
        return {
            'loss_factors': total_loss_factors / num_batches if num_batches > 0 else 0.0,
            'loss_product': total_loss_product / num_batches if num_batches > 0 else 0.0,
            'loss_total': total_loss_total / num_batches if num_batches > 0 else 0.0
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.history['train_loss_total'][-1] if self.history['train_loss_total'] else None,
            'val_loss': self.history['val_loss_total'][-1] if self.history['val_loss_total'] else None,
            'best_val_loss': self.best_val_loss,
            'version': self.current_version,
            'alpha': self.alpha
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
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'epoch', 
                'train_loss_factors', 'train_loss_product', 'train_loss_total',
                'val_loss_factors', 'val_loss_product', 'val_loss_total'
            ])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'timestamp': timestamp,
                'epoch': epoch,
                'train_loss_factors': self.history['train_loss_factors'][-1] if self.history['train_loss_factors'] else None,
                'train_loss_product': self.history['train_loss_product'][-1] if self.history['train_loss_product'] else None,
                'train_loss_total': self.history['train_loss_total'][-1] if self.history['train_loss_total'] else None,
                'val_loss_factors': self.history['val_loss_factors'][-1] if self.history['val_loss_factors'] else None,
                'val_loss_product': self.history['val_loss_product'][-1] if self.history['val_loss_product'] else None,
                'val_loss_total': self.history['val_loss_total'][-1] if self.history['val_loss_total'] else None
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
        
        plt.figure(figsize=(14, 8))
        
        # Plot total losses
        plt.subplot(2, 1, 1)
        plt.plot(self.history['epochs'], self.history['train_loss_total'], label='Train Total Loss', marker='o')
        plt.plot(self.history['epochs'], self.history['val_loss_total'], label='Val Total Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Training and Validation Total Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot component losses
        plt.subplot(2, 1, 2)
        plt.plot(self.history['epochs'], self.history['train_loss_factors'], label='Train Factors Loss', marker='o', linestyle='--')
        plt.plot(self.history['epochs'], self.history['train_loss_product'], label='Train Product Loss', marker='o', linestyle=':')
        plt.plot(self.history['epochs'], self.history['val_loss_factors'], label='Val Factors Loss', marker='s', linestyle='--')
        plt.plot(self.history['epochs'], self.history['val_loss_product'], label='Val Product Loss', marker='s', linestyle=':')
        plt.xlabel('Epoch')
        plt.ylabel('Component Loss')
        plt.title('Training and Validation Component Losses')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
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
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            self.history['epochs'].append(epoch + 1)
            self.history['train_loss_factors'].append(train_metrics['loss_factors'])
            self.history['train_loss_product'].append(train_metrics['loss_product'])
            self.history['train_loss_total'].append(train_metrics['loss_total'])
            self.history['val_loss_factors'].append(val_metrics['loss_factors'])
            self.history['val_loss_product'].append(val_metrics['loss_product'])
            self.history['val_loss_total'].append(val_metrics['loss_total'])
            
            # Check if best model (using total validation loss)
            is_best = val_metrics['loss_total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss_total']
            
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
                    'train_loss': train_metrics['loss_total'],  # For backward compatibility
                    'val_loss': val_metrics['loss_total'],  # For backward compatibility
                    'train_loss_factors': train_metrics['loss_factors'],
                    'train_loss_product': train_metrics['loss_product'],
                    'train_loss_total': train_metrics['loss_total'],
                    'val_loss_factors': val_metrics['loss_factors'],
                    'val_loss_product': val_metrics['loss_product'],
                    'val_loss_total': val_metrics['loss_total'],
                    'is_best': is_best
                })
            
            print(f"Epoch {epoch + 1}/{start_epoch + num_epochs} - "
                  f"Train: factors={train_metrics['loss_factors']:.4f}, "
                  f"product={train_metrics['loss_product']:.4f}, "
                  f"total={train_metrics['loss_total']:.4f} | "
                  f"Val: factors={val_metrics['loss_factors']:.4f}, "
                  f"product={val_metrics['loss_product']:.4f}, "
                  f"total={val_metrics['loss_total']:.4f}")
        
        # Final metrics
        metrics = {
            'final_train_loss': self.history['train_loss_total'][-1] if self.history['train_loss_total'] else None,
            'final_val_loss': self.history['val_loss_total'][-1] if self.history['val_loss_total'] else None,
            'final_train_loss_factors': self.history['train_loss_factors'][-1] if self.history['train_loss_factors'] else None,
            'final_train_loss_product': self.history['train_loss_product'][-1] if self.history['train_loss_product'] else None,
            'final_val_loss_factors': self.history['val_loss_factors'][-1] if self.history['val_loss_factors'] else None,
            'final_val_loss_product': self.history['val_loss_product'][-1] if self.history['val_loss_product'] else None,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.history['epochs']),
            'model_version': self.current_version,
            'alpha': self.alpha
        }
        
        return metrics

