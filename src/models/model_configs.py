from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    input_dim: int = 512
    hidden_dims: list[int] = field(default_factory=list)
    dropout: float = 0.2

TINY = ModelConfig(input_dim=512, hidden_dims=[128], dropout=0.1)
SMALL = ModelConfig(input_dim=512, hidden_dims=[256, 128], dropout=0.2)
MEDIUM = ModelConfig(input_dim=512, hidden_dims=[512, 256], dropout=0.2)

