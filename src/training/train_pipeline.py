from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

from src.embeddings.embedder import CLIPEmbedder, CLIP_VIT_B32
from src.models.count_predictor import CountPredictor
from src.models.model_configs import SMALL
from src.training.coco_dataset import COCOCountDataset
from src.training.virat_dataset import VIRATCountDataset
from src.training.trainer import CountTrainer
from src.training.evaluator import CountEvaluator

def pretrain_on_coco(coco_dir: str, target: str, model_config=SMALL):
    embedder = CLIPEmbedder(CLIP_VIT_B32)
    
    train_dataset = COCOCountDataset(
        coco_dir=coco_dir,
        embeddings_cache="data/coco_embeddings",
        split="train2017",
        target=target
    )
    train_dataset.precompute_embeddings(embedder, batch_size=32)
    
    val_dataset = COCOCountDataset(
        coco_dir=coco_dir,
        embeddings_cache="data/coco_embeddings",
        split="val2017",
        target=target
    )
    val_dataset.precompute_embeddings(embedder, batch_size=32)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    model = CountPredictor(model_config)
    trainer = CountTrainer(model, lr=1e-3, weight_decay=1e-4)
    
    checkpoint_path = f"models/checkpoints/{target}_coco_pretrained.pth"
    trainer.train(train_loader, val_loader, epochs=50, patience=10, checkpoint_path=checkpoint_path)
    
    print(f"\nPre-training completed for {target}")

def finetune_on_virat(data_dir: str, target: str, pretrained_checkpoint: str, 
                      train_ratio: float = 0.4, model_config=SMALL):
    video_dirs = [d.name for d in Path(data_dir).iterdir() if d.is_dir()]
    random.seed(42)
    random.shuffle(video_dirs)
    
    split_idx = int(len(video_dirs) * train_ratio)
    train_videos = video_dirs[:split_idx]
    test_videos = video_dirs[split_idx:]
    
    print(f"Training on {len(train_videos)} videos, testing on {len(test_videos)} videos")
    
    train_dataset = VIRATCountDataset(data_dir, train_videos, target=target)
    test_dataset = VIRATCountDataset(data_dir, test_videos, target=target)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    model = CountPredictor(model_config)
    trainer = CountTrainer(model, lr=1e-4, weight_decay=1e-5)

    if pretrained_checkpoint:
        checkpoint = torch.load(pretrained_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pre-trained model from {pretrained_checkpoint}")
    else:
        print("No pre-trained checkpoint provided — training from scratch.")
    
    checkpoint_path = f"models/checkpoints/{target}_virat_finetuned.pth"
    trainer.train(train_loader, test_loader, epochs=30, patience=5, checkpoint_path=checkpoint_path)
    
    evaluator = CountEvaluator(model)
    results = evaluator.evaluate(test_loader)
    evaluator.print_results(results, target_name=f"{target.capitalize()} Count")
    
    print(f"\nFine-tuning completed for {target}")

def evaluate_model(data_dir: str, target: str, checkpoint_path: str, model_config=SMALL):
    video_dirs = [d.name for d in Path(data_dir).iterdir() if d.is_dir()]
    
    dataset = VIRATCountDataset(data_dir, video_dirs, target=target)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
    
    model = CountPredictor(model_config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    evaluator = CountEvaluator(model)
    results = evaluator.evaluate(dataloader)
    evaluator.print_results(results, target_name=f"{target.capitalize()} Count")

