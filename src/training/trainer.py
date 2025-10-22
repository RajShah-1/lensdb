from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.count_predictor import CountPredictor
from src.utils import get_best_device

class CountTrainer:
    def __init__(self, model: CountPredictor, lr: float = 1e-3, 
                 weight_decay: float = 1e-4, device: str | None = None,
                 loss_type: str = "mse", underpredict_penalty: float = 1.0):
        self.model = model
        self.device = device or get_best_device()
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        self.loss_type = loss_type
        self.underpredict_penalty = underpredict_penalty
        
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "mae":
            self.criterion = nn.L1Loss()
        elif loss_type == "huber":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss with optional asymmetric penalty for under-prediction."""
        if self.underpredict_penalty == 1.0:
            return self.criterion(predictions, targets)
        
        residuals = predictions - targets
        
        if self.loss_type == "mse":
            squared_errors = residuals ** 2
            weights = torch.where(residuals < 0, self.underpredict_penalty, 1.0)
            return (weights * squared_errors).mean()
        elif self.loss_type == "mae":
            abs_errors = torch.abs(residuals)
            weights = torch.where(residuals < 0, self.underpredict_penalty, 1.0)
            return (weights * abs_errors).mean()
        else:
            base_loss = self.criterion(predictions, targets)
            return base_loss
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for embeddings, counts in dataloader:
            embeddings = embeddings.to(self.device)
            counts = counts.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(embeddings.float())
            loss = self.compute_loss(predictions, counts.float())

            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(embeddings)
        
        return total_loss / len(dataloader.dataset)
    
    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for embeddings, counts in dataloader:
                embeddings = embeddings.to(self.device)
                counts = counts.to(self.device)
                
                predictions = self.model(embeddings.float())
                loss = self.compute_loss(predictions, counts.float())

                total_loss += loss.item() * len(embeddings)
        
        return total_loss / len(dataloader.dataset)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, patience: int, checkpoint_path: str | None = None):
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {self.model.num_parameters():,}")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path, epoch, val_loss)
                    print(f"Saved checkpoint to {checkpoint_path}")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.model.config,
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint

