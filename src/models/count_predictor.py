import torch
import torch.nn as nn

from src.models.model_configs import ModelConfig

class CountPredictor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        layers = []
        in_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
    
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

