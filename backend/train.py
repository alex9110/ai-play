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
        
        return x, y


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    checkpoint_dir: str = "models",
    resume_from: Optional[str] = None
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
        print(f"Resumed from epoch {start_epoch}")
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint every epoch
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = Path(checkpoint_dir) / "model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"Saved best model with val loss: {val_loss:.4f}")
    
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
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()

