"""
Training script for the factorization model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Optional
import argparse

from model import FactorizationMLP, encode_number
from dataset import load_dataset, generate_dataset, save_dataset
from core.dataset import FactorizationDataset as CoreFactorizationDataset


class FactorizationDataset(Dataset):
    """PyTorch Dataset for factorization data."""
    
    def __init__(self, data: list):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        n = sample['n']
        factor_a = sample['factor_a']
        factor_b = sample['factor_b']
        
        # Encode input
        x = encode_number(n)
        
        # Encode target
        y = torch.tensor([float(factor_a), float(factor_b)], dtype=torch.float32)
        
        # Include n for product loss
        n_tensor = torch.tensor(float(n), dtype=torch.float32)
        
        return x, y, n_tensor


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    checkpoint_dir: str = "models",
    resume_from: Optional[str] = None,
    alpha: float = 0.05
):
    """
    Train the factorization model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    start_epoch = 0
    
    # Resume from checkpoint if provided
    if resume_from and Path(resume_from).exists():
        print(f"Resuming training from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'alpha' in checkpoint:
            alpha = checkpoint['alpha']
        print(f"Resumed from epoch {start_epoch}, alpha={alpha}")
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss_factors = 0.0
        train_loss_product = 0.0
        train_loss_total = 0.0
        for batch in train_loader:
            # Handle both old format (x, y) and new format (x, y, n)
            if len(batch) == 2:
                x, y = batch
                n_values = None
            else:
                x, y, n_values = batch
                n_values = n_values.to(device)
            
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            
            # Factor loss
            loss_factors = criterion(outputs, y)
            
            # Product loss
            if n_values is not None:
                product_pred = outputs[:, 0] * outputs[:, 1]
                if n_values.dim() == 0:
                    n_values = n_values.unsqueeze(0).expand(outputs.size(0))
                elif n_values.size(0) != outputs.size(0):
                    n_values = n_values[:outputs.size(0)]
                loss_product = criterion(product_pred, n_values)
            else:
                # Fallback: compute n from targets
                n_computed = y[:, 0] * y[:, 1]
                product_pred = outputs[:, 0] * outputs[:, 1]
                loss_product = criterion(product_pred, n_computed)
            
            # Composite loss
            loss_total = loss_factors + alpha * loss_product
            loss_total.backward()
            optimizer.step()
            
            train_loss_factors += loss_factors.item()
            train_loss_product += loss_product.item()
            train_loss_total += loss_total.item()
        
        train_loss_factors /= len(train_loader)
        train_loss_product /= len(train_loader)
        train_loss_total /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss_factors = 0.0
        val_loss_product = 0.0
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # Handle both old format (x, y) and new format (x, y, n)
                if len(batch) == 2:
                    x, y = batch
                    n_values = None
                else:
                    x, y, n_values = batch
                    n_values = n_values.to(device)
                
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                
                # Factor loss
                loss_factors = criterion(outputs, y)
                
                # Product loss
                if n_values is not None:
                    product_pred = outputs[:, 0] * outputs[:, 1]
                    if n_values.dim() == 0:
                        n_values = n_values.unsqueeze(0).expand(outputs.size(0))
                    elif n_values.size(0) != outputs.size(0):
                        n_values = n_values[:outputs.size(0)]
                    loss_product = criterion(product_pred, n_values)
                else:
                    # Fallback: compute n from targets
                    n_computed = y[:, 0] * y[:, 1]
                    product_pred = outputs[:, 0] * outputs[:, 1]
                    loss_product = criterion(product_pred, n_computed)
                
                # Composite loss
                loss_total = loss_factors + alpha * loss_product
                
                val_loss_factors += loss_factors.item()
                val_loss_product += loss_product.item()
                val_loss_total += loss_total.item()
        
        val_loss_factors /= len(val_loader)
        val_loss_product /= len(val_loader)
        val_loss_total /= len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train: factors={train_loss_factors:.4f}, product={train_loss_product:.4f}, total={train_loss_total:.4f} | "
              f"Val: factors={val_loss_factors:.4f}, product={val_loss_product:.4f}, total={val_loss_total:.4f}")
        
        # Save checkpoint every epoch
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss_total,
            'val_loss': val_loss_total,
            'alpha': alpha
        }, checkpoint_path)
        
        # Save best model
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            best_model_path = Path(checkpoint_dir) / "model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_total,
                'val_loss': val_loss_total,
                'alpha': alpha
            }, best_model_path)
            print(f"Saved best model with val loss: {val_loss_total:.4f}")
    
    print("Training completed!")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train factorization model")
    parser.add_argument("--dataset", type=str, default="dataset.json", help="Path to dataset JSON")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", type=str, default="models", help="Checkpoint directory")
    parser.add_argument("--alpha", type=float, default=0.05, help="Weight for product consistency loss (default: 0.05)")
    
    args = parser.parse_args()
    
    # Load or generate dataset
    dataset = load_dataset(args.dataset)
    if not dataset:
        print(f"Dataset not found at {args.dataset}. Generating new dataset...")
        dataset = generate_dataset(num_samples=10000)
        save_dataset(dataset, args.dataset)
    
    print(f"Loaded {len(dataset)} samples")
    
    # Split dataset
    split_idx = int(0.8 * len(dataset))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    # Create data loaders
    train_dataset = FactorizationDataset(train_data)
    val_dataset = FactorizationDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = FactorizationMLP(input_dim=40, hidden_dims=[128, 256, 128], dropout=0.2)
    
    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        alpha=args.alpha
    )


if __name__ == "__main__":
    main()

