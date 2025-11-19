import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

class VIRATCountDataset(Dataset):
    def __init__(self, data_dir: str, video_names: list[str], target: str = "car"):
        self.data_dir = Path(data_dir)
        self.video_names = video_names
        self.target = target
        self.samples = []
        
        self._load_data()
    
    def _load_data(self):
        for video_name in self.video_names:
            video_dir = self.data_dir / video_name
            embeddings_file = video_dir / "embeddings" / "embds_clip_full.npy"
            counts_file = video_dir / "counts.csv"
            
            if not embeddings_file.exists() or not counts_file.exists():
                print(f"Warning: Missing data for {video_name}, skipping")
                continue
            
            embeddings = np.load(embeddings_file)
            
            with open(counts_file, 'r') as f:
                reader = csv.DictReader(f)
                counts = []
                for row in reader:
                    if self.target == "car":
                        counts.append(int(row['car_count']))
                    else:
                        counts.append(int(row['people_count']))
            
            min_len = min(len(embeddings), len(counts))
            for i in range(min_len):
                self.samples.append((embeddings[i], counts[i]))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        embedding, count = self.samples[idx]
        return torch.from_numpy(embedding).float(), float(count)

