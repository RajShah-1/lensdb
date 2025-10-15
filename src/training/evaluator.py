import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.count_predictor import CountPredictor
from src.utils import get_best_device

class CountEvaluator:
    def __init__(self, model: CountPredictor, device: str | None = None):
        self.model = model
        self.device = device or get_best_device()
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, dataloader: DataLoader) -> dict:
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for embeddings, counts in dataloader:
                embeddings = embeddings.to(self.device)
                predictions = self.model(embeddings)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(counts.numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        predictions = np.maximum(predictions, 0)
        
        mae = np.mean(np.abs(predictions - targets))
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        rounded_preds = np.round(predictions)
        exact_acc = np.mean(rounded_preds == targets) * 100
        within_1 = np.mean(np.abs(rounded_preds - targets) <= 1) * 100
        within_2 = np.mean(np.abs(rounded_preds - targets) <= 2) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'exact_accuracy': exact_acc,
            'within_1_accuracy': within_1,
            'within_2_accuracy': within_2,
            'predictions': predictions,
            'targets': targets,
        }
    
    def print_results(self, results: dict, target_name: str = "Count"):
        print(f"\n{target_name} Prediction Results:")
        print(f"  MAE: {results['mae']:.3f}")
        print(f"  MSE: {results['mse']:.3f}")
        print(f"  RMSE: {results['rmse']:.3f}")
        print(f"  R²: {results['r2']:.3f}")
        print(f"  Exact accuracy: {results['exact_accuracy']:.1f}%")
        print(f"  ±1 accuracy: {results['within_1_accuracy']:.1f}%")
        print(f"  ±2 accuracy: {results['within_2_accuracy']:.1f}%")

