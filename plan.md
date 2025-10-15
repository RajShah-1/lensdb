# LensDB Plan

## Current Status
- Detection pipeline running (YOLO11x on VIRAT videos)
- Embedding generation running (CLIP ViT-B/32)
- Data structure: `data/{video_stem}/` contains `counts.csv`, `frames/`, `embeddings/`

## Next: Train Count Predictors from CLIP Embeddings

### Goal
Two-stage training: Pre-train on COCO dataset, then transfer learn on VIRAT videos. Predict car/person counts from CLIP embeddings.

### Directory Structure
```
src/
├── models/
│   ├── __init__.py
│   ├── count_predictor.py      # MLP regression model
│   └── model_configs.py        # Architecture configs (TINY, SMALL, MEDIUM)
├── training/
│   ├── __init__.py
│   ├── coco_dataset.py         # COCO dataset with CLIP embeddings
│   ├── virat_dataset.py        # VIRAT dataset with embeddings + counts
│   ├── trainer.py              # Training loop, checkpointing, metrics
│   └── evaluator.py            # Eval metrics (MAE, MSE, accuracy@k)
└── main.py                     # Add train/eval functions
```

### Two-Stage Training Strategy

**Stage 1: Pre-train on COCO**
1. Download COCO dataset (~118K train images)
2. Generate CLIP embeddings for COCO images (one-time, cached)
3. Extract car/person counts from COCO annotations (category_ids: car=3, person=1)
4. Pre-train two models from scratch on COCO

**Stage 2: Transfer Learning on VIRAT**
1. Random sample 40% of VIRAT videos for fine-tuning
2. Use YOLO11x counts as ground truth labels
3. Fine-tune pre-trained models on VIRAT subset
4. Evaluate on remaining 60% of VIRAT videos

### Data Split
- COCO: All train images for pre-training
- VIRAT: 40% train (fine-tune), 60% test (eval)
- Split by video to prevent frame leakage

### Model Architecture
```python
class CountPredictor(nn.Module):
    # Simple MLP: Linear → ReLU → Dropout → Linear → ReLU → Linear
    # Input: 512 (CLIP ViT-B/32 embedding dim)
    # Output: 1 (count prediction)
```

**Variants:**
- TINY: 512 → 128 → 1 (~66K params)
- SMALL: 512 → 256 → 128 → 1 (~200K params)
- MEDIUM: 512 → 512 → 256 → 1 (~530K params)

### Training Config

**Stage 1 (COCO Pre-training):**
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Loss: MSE (L2 regression)
- Batch size: 256
- Max epochs: 50
- Early stopping: patience=10
- Save: `models/checkpoints/{car|person}_coco_pretrained.pth`

**Stage 2 (VIRAT Fine-tuning):**
- Optimizer: Adam (lr=1e-4, weight_decay=1e-5) - lower LR
- Loss: MSE
- Batch size: 128
- Max epochs: 30
- Early stopping: patience=5
- Save: `models/checkpoints/{car|person}_virat_finetuned.pth`

### Evaluation Metrics
- **Regression**: MAE, MSE, RMSE, R²
- **Count accuracy**: exact, ±1, ±2 tolerance
- Per-video statistics
- Confusion matrix for binned counts (0, 1-2, 3-5, 6+)

### Files to Create
1. `src/models/count_predictor.py` - MLP model class
2. `src/models/model_configs.py` - Config dataclasses
3. `src/training/coco_dataset.py` - COCO loader with CLIP embeddings
4. `src/training/virat_dataset.py` - VIRAT dataset loader
5. `src/training/trainer.py` - Training loop (pre-train + fine-tune)
6. `src/training/evaluator.py` - Evaluation metrics
7. Update `src/main.py` - Add train/eval functions

### Code Style (match existing)
- Type hints everywhere
- Minimal comments (self-documenting code)
- Modular, single-responsibility classes
- Dataclasses for configs
- snake_case naming
- Files ~150 lines max

## Future: Embedding Compression
- PCA dimensionality reduction
- Vector quantization
- Embedding pruning/distillation
- Compare storage vs accuracy tradeoffs


## My Meta:

So, the embedder is now running. 

Next steps:
- Train tiny NN to predict car/person counts from CLIP embeddings
- Two-stage approach: Pre-train on COCO dataset (large, diverse), then transfer learn on 40% of VIRAT videos
- Evaluate on remaining 60% of VIRAT, measuring accuracy against YOLO11x ground truth
- Two separate models: one for car count, one for person count

Once we are done with this, we'll explore more ways of compressing the embeddings to make them storage efficient. 