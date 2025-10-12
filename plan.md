# LensDB Plan

## Current Status
- Detection pipeline running (YOLO11x on VIRAT videos)
- Embedding generation running (CLIP ViT-B/32)
- Data structure: `data/{video_stem}/` contains `counts.csv`, `frames/`, `embeddings/`

## Next: Train Count Predictors from CLIP Embeddings

### Goal
Train lightweight NNs to predict car/person counts from CLIP embeddings, using large YOLO detector as ground truth.

### Directory Structure
```
src/
├── models/
│   ├── __init__.py
│   ├── count_predictor.py      # MLP regression model
│   └── model_configs.py        # Architecture configs (TINY, SMALL, MEDIUM)
├── training/
│   ├── __init__.py
│   ├── dataset.py              # PyTorch Dataset: embeddings + counts
│   ├── trainer.py              # Training loop, checkpointing, metrics
│   └── evaluator.py            # Eval metrics (MAE, MSE, accuracy@k)
└── main.py                     # Add train/eval functions
```

### Data Flow
1. Load embeddings (`.npy`) + counts (`.csv`) from `data/{video_stem}/`
2. Create dataset: pairs of (embedding, car_count, person_count)
3. Split by video: 80/10/10 train/val/test (prevent frame leakage)
4. Train two separate models: one for cars, one for people
5. Evaluate: MAE, MSE, R², count accuracy (exact, ±1, ±2)

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
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Loss: MSE (L2 regression)
- Batch size: 256
- Max epochs: 100
- Early stopping: patience=10
- Checkpoints: `models/checkpoints/{car|person}_predictor_best.pth`

### Evaluation Metrics
- **Regression**: MAE, MSE, RMSE, R²
- **Count accuracy**: exact, ±1, ±2 tolerance
- Per-video statistics
- Confusion matrix for binned counts (0, 1-2, 3-5, 6+)

### Files to Create
1. `src/models/count_predictor.py` - MLP model class
2. `src/models/model_configs.py` - Config dataclasses
3. `src/training/dataset.py` - Dataset loader
4. `src/training/trainer.py` - Training loop
5. `src/training/evaluator.py` - Evaluation metrics
6. Update `src/main.py` - Add train/eval functions

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

Next steps are below:
- We want to train a tiny NN which predicts car and person count from CLIP embeddings.
- To do this, we take a small YOLO variant and train it on a famous image dataset (COCO or anything nice). This model will take CLIP embds and give car count (another model for person count). We'll measure accuracy against large detector accuracy.

Once we are done with this, we'll explore more ways of compressing the embeddings to make them storage efficient. 