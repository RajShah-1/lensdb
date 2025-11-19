import csv
from pathlib import Path

import numpy as np

from src.pipeline.pipeline import VideoPipeline
from src.embeddings.embedder import CLIPEmbedder, CLIP_VIT_B32
from src.models.model_configs import MEDIUM, SMALL, LARGE, LARGE3
from src.pipeline.detection_pipeline import DetectionPipeline
from src.detectors.object_detector import ObjectDetector
from src.training.train_pipeline import finetune_on_virat, pretrain_on_coco
from src.indexing.faiss_index import FAISSIndex
from src.query.semantic_query import SemanticQueryPipeline
from src.baseline import evaluate_baseline_yolo

def run_detection():
    video_path = "videos/demo.mp4"
    detector = ObjectDetector(model_name="yolov8l")
    pipeline = DetectionPipeline(video_path, detector)
    counts = pipeline.run(save=True)

def run_detection_on_dir(videos_dir: str, model_name: str, annotated: bool):
    videos = sorted(Path(videos_dir).glob("*.mp4"))
    print(f"Found {len(videos)} videos in {videos_dir}")

    detector = ObjectDetector(model_name=model_name, save_annotated=annotated)

    for vid in videos:
        print(f"\n=== Processing {vid.name} ===")
        out_dir = Path("data/VIRAT") / vid.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline = DetectionPipeline(str(vid), detector, out_dir=out_dir)
        pipeline.run(save=True)

def gen_embeddings_for_dir(videos_dir: str, embedder_config=None):
    """
    Generate embeddings for all videos in a directory.
    """
    videos = sorted(Path(videos_dir).glob("*.mp4"))
    print(f"Found {len(videos)} videos in {videos_dir}")
    
    if embedder_config is None:
        embedder_config = CLIP_VIT_B32
    
    embedder = CLIPEmbedder(embedder_config)
    
    for vid in videos:
        print(f"\n=== Processing {vid.name} ===")
        out_dir = Path("data/VIRAT") / vid.stem / "embeddings"
        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline = VideoPipeline(str(vid), embedder, out_dir=out_dir)
        pipeline.run(save=True)

def build_index(data_dir="data/VIRAT"):
    """Build FAISS index for semantic queries."""
    index = FAISSIndex(data_dir)
    index.build()

def run_semantic_query():
    """
    Run a semantic query using FAISS + MLP pipeline.
    
    Example: Find frames with cars (count >= 2)
    """
    pipeline = SemanticQueryPipeline(
        data_dir="data/VIRAT",
        checkpoint_path="models/checkpoints/car_virat_finetuned.pth",
        model_config=MEDIUM,
        threshold=0.2
    )
    
    results = pipeline.query(
        text_query="car",
        count_predicate=lambda c: c >= 2.0
    )
    return results

def load_ground_truth(data_dir: str, video_name: str, target: str = "car"):
    """Load ground truth counts from counts.csv."""
    counts_file = Path(data_dir) / video_name / "counts.csv"
    
    if not counts_file.exists():
        raise FileNotFoundError(f"Missing {counts_file}")
    
    counts = {}
    with open(counts_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_id = int(row['frame_id'])
            if target == "car":
                counts[frame_id] = int(row['car_count'])
            else:
                counts[frame_id] = int(row['people_count'])
    
    return counts

def evaluate_retrieval(data_dir: str, checkpoint_path: str, model_config,
                       target: str = "car", count_threshold: int = 2,
                       similarity_threshold: float = 0.2, num_videos: int = 2):
    """
    Evaluate retrieval quality against ground truth oracle model.
    Measures metrics at FAISS and MLP stages separately.
    """
    data_path = Path(data_dir)
    video_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith("_")])
    
    if len(video_dirs) < num_videos:
        raise ValueError(f"Only {len(video_dirs)} videos available, need {num_videos}")
    
    eval_videos = [v.name for v in video_dirs[:num_videos]]
    
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY: {target} (count >= {count_threshold})")
    print(f"{'='*70}")
    print(f"Videos: {', '.join(eval_videos)}")
    print(f"Similarity threshold: {similarity_threshold}")
    
    pipeline = SemanticQueryPipeline(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        threshold=similarity_threshold,
    )
    
    # Get metrics from pipeline which now includes timing
    metrics = pipeline.query_with_metrics(
        text_query=target,
        count_threshold=count_threshold,
        eval_videos=eval_videos,
        data_dir=data_dir,
        emb_filename="embds_clip_full.npy"
    )
    
    # Print ablation study results
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY RESULTS")
    print(f"{'='*70}")
    
    print(f"\n[LATENCY METRICS]")
    print(f"  FAISS lookup latency:  {metrics['faiss_latency_ms']:.2f} ms")
    print(f"  MLP prediction latency: {metrics['mlp_latency_ms']:.2f} ms")
    print(f"  Total pipeline latency: {metrics['total_latency_ms']:.2f} ms")
    
    print(f"\n[FAISS STAGE METRICS]")
    print(f"  Precision: {metrics['faiss_precision']:.3f}")
    print(f"  Recall:    {metrics['faiss_recall']:.3f}")
    print(f"  F1:        {metrics['faiss_f1']:.3f}")
    print(f"  Retrieved frames: {metrics['faiss_retrieved']}")
    
    print(f"\n[MLP STAGE METRICS (Final Pipeline)]")
    print(f"  Precision: {metrics['mlp_precision']:.3f}")
    print(f"  Recall:    {metrics['mlp_recall']:.3f}")
    print(f"  F1:        {metrics['mlp_f1']:.3f}")
    print(f"  Retrieved frames: {metrics['mlp_retrieved']}")
    
    print(f"\n{'='*70}")
    
    return metrics


if __name__ == "__main__":
    # ========================================
    # STEP 1: Generate embeddings for videos
    # ========================================
    # gen_embeddings()  # Single video
    # gen_embeddings_for_dir("/path/to/videos", CLIP_VIT_B32)  # Batch
    
    # ========================================
    # STEP 2: Generate ground truth counts (optional, for training)
    # ========================================
    # run_detection()  # Single video
    # run_detection_on_dir("/path/to/videos", "yolo11x.pt", False)  # Batch
    
    # ========================================
    # STEP 3: Train count predictor
    # ========================================

    faiss_index = FAISSIndex("data/VIRAT", use_keyframes=False)
    faiss_index.build()
    # pretrain_on_coco(
    #     coco_dir="data/coco",
    #     target="car",
    #     model_config=LARGE3
    # )
    finetune_on_virat(
        data_dir="data/VIRAT",
        target="car",
        pretrained_checkpoint="models/checkpoints/car_virat_finetuned.pth",
        train_ratio=0.5,
        model_config=LARGE3
    )
    
    # ========================================
    # STEP 4: Build FAISS index
    # ========================================
    # build_index("data/VIRAT")
    
    # ========================================
    # STEP 5: Run semantic queries
    # ========================================
    # Simple query
    # run_semantic_query()
    
    # Advanced query with custom predicates
    # run_advanced_query()
    
    # ========================================
    # STEP 6: Evaluate retrieval (first 2 videos)
    # ========================================
    evaluate_retrieval(
        data_dir="data/VIRAT",
        checkpoint_path="models/checkpoints/car_virat_finetuned.pth",
        model_config=LARGE3,
        target="car",
        count_threshold=1,
        similarity_threshold=0.2,
        num_videos=2
    )
    
    # ========================================
    # STEP 7: Evaluate baseline YOLO11 (for comparison)
    # ========================================
    # evaluate_baseline_yolo(
    #     data_dir="data/VIRAT",
    #     model_name="yolo11n",
    #     target="car",
    #     count_threshold=1,
    #     num_videos=2
    # )


