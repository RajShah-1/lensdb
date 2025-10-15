import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from src.embeddings.embedder import CLIPEmbedder
from src.utils import get_best_device

CAR_CATEGORY_ID = 3
PERSON_CATEGORY_ID = 1

class COCOCountDataset(Dataset):
    def __init__(self, coco_dir: str, embeddings_cache: str | None = None, 
                 split: str = "train2017", target: str = "car"):
        self.coco_dir = Path(coco_dir)
        self.images_dir = self.coco_dir / split
        self.ann_file = self.coco_dir / "annotations" / f"instances_{split}.json"
        self.target = target
        self.category_id = CAR_CATEGORY_ID if target == "car" else PERSON_CATEGORY_ID
        
        self.coco = COCO(str(self.ann_file))
        self.image_ids = list(self.coco.imgs.keys())
        
        self.embeddings_cache = Path(embeddings_cache) if embeddings_cache else None
        if self.embeddings_cache:
            self.embeddings_cache.mkdir(parents=True, exist_ok=True)
        
        self.counts = self._compute_counts()
        self.embeddings = None
        
    def _compute_counts(self) -> dict[int, int]:
        counts = defaultdict(int)
        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[self.category_id])
            counts[img_id] = len(ann_ids)
        return counts
    
    def precompute_embeddings(self, embedder: CLIPEmbedder, batch_size: int = 32):
        cache_file = self.embeddings_cache / f"coco_{self.target}_embeddings.npy"
        
        if cache_file.exists():
            print(f"Loading cached embeddings from {cache_file}")
            self.embeddings = np.load(cache_file)
            return
        
        print(f"Computing CLIP embeddings for {len(self)} COCO images...")
        all_embeddings = []
        
        for i in range(0, len(self), batch_size):
            batch_imgs = []
            for j in range(i, min(i + batch_size, len(self))):
                img_info = self.coco.imgs[self.image_ids[j]]
                img_path = self.images_dir / img_info['file_name']
                img = cv2.imread(str(img_path))
                if img is not None:
                    batch_imgs.append(img)
            
            if batch_imgs:
                embeds = embedder.embed(batch_imgs)
                all_embeddings.append(embeds)
            
            if (i + batch_size) % 1000 == 0:
                print(f"Processed {i + batch_size}/{len(self)} images")
        
        self.embeddings = np.vstack(all_embeddings)
        
        if self.embeddings_cache:
            np.save(cache_file, self.embeddings)
            print(f"Saved embeddings to {cache_file}")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        img_id = self.image_ids[idx]
        count = self.counts[img_id]
        
        if self.embeddings is not None:
            embedding = self.embeddings[idx]
        else:
            img_info = self.coco.imgs[img_id]
            img_path = self.images_dir / img_info['file_name']
            img = cv2.imread(str(img_path))
            embedding = np.zeros(512)
        
        return torch.from_numpy(embedding).float(), float(count)

