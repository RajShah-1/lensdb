"""
Modular benchmark functions for testing different pipeline configurations.

Provides three main benchmarking functions:
  - benchmark_baseline() - YOLO baseline evaluation
  - benchmark_embds() - Standard embedding-based pipeline (no keyframes)
  - benchmark_with_kf() - Keyframe-based pipeline

These can be imported and called independently from main.py or other scripts.
"""

from pathlib import Path
from src.models.model_configs import LARGE3
from src.embeddings.embedder import CLIP_VIT_B32
from src.query.semantic_query import SemanticQueryPipeline
from src.baseline import evaluate_baseline_yolo
from src.pipeline.keyframe_pipeline import process_video_folder
from src.indexing.faiss_index import FAISSIndex


def benchmark_baseline(
    data_dir: str = "data/VIRAT",
    target: str = "car",
    num_videos: int = 5,
    thresholds: list[int] = [0, 1, 2, 3, 4, 5],
    yolo_model: str = "yolo11m"
):
    """
    Benchmark YOLO baseline performance.
    
    Args:
        data_dir: Directory containing video data
        target: Object class to detect ("car" or "person")
        num_videos: Number of videos to test on
        thresholds: List of count thresholds to test
        yolo_model: YOLO model to use (yolo11n/s/m/l/x)
    
    Returns:
        Dictionary with baseline results for each threshold
    """
    print("\n" + "="*80)
    print("BENCHMARKING YOLO BASELINE")
    print("="*80)
    print(f"Model: {yolo_model}")
    print(f"Target: {target}")
    print(f"Videos: {num_videos}")
    print(f"Thresholds: {thresholds}")
    print("="*80 + "\n")
    
    baseline_results = {}
    
    for threshold in thresholds:
        print(f"\n{'─'*80}")
        print(f"Testing with count threshold >= {threshold}")
        print(f"{'─'*80}")
        
        metrics = evaluate_baseline_yolo(
            data_dir=data_dir,
            model_name=yolo_model,
            target=target,
            count_threshold=threshold,
            num_videos=num_videos
        )
        
        baseline_results[threshold] = metrics
    
    print("\n" + "="*80)
    print("BASELINE BENCHMARK COMPLETE")
    print("="*80 + "\n")
    
    return baseline_results


def benchmark_embds(
    data_dir: str = "data/VIRAT",
    checkpoint_path: str = "models/checkpoints/car_virat_finetuned.pth",
    model_config=LARGE3,
    target: str = "car",
    similarity_threshold: float = 0.2,
    num_videos: int = 5,
    thresholds: list[int] = [0, 1, 2, 3, 4, 5]
):
    """
    Benchmark standard embedding-based FAISS+MLP pipeline (no keyframes).
    
    Args:
        data_dir: Directory containing video data
        checkpoint_path: Path to trained model checkpoint
        model_config: Model configuration
        target: Object class to detect ("car" or "person")
        similarity_threshold: FAISS similarity threshold
        num_videos: Number of videos to test on
        thresholds: List of count thresholds to test
    
    Returns:
        Dictionary with pipeline results for each threshold
    """
    print("\n" + "="*80)
    print("BENCHMARKING EMBEDDING PIPELINE (No Keyframes)")
    print("="*80)
    print(f"Target: {target}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Videos: {num_videos}")
    print(f"Thresholds: {thresholds}")
    print("="*80 + "\n")
    
    data_path = Path(data_dir)
    video_dirs = sorted([d for d in data_path.iterdir() 
                        if d.is_dir() and not d.name.startswith("_")])
    eval_videos = [v.name for v in video_dirs[:num_videos]]
    
    print("Initializing FAISS+MLP pipeline...")
    pipeline = SemanticQueryPipeline(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        threshold=similarity_threshold
    )
    
    pipeline_results = {}
    
    for threshold in thresholds:
        print(f"\n{'─'*80}")
        print(f"Testing with count threshold >= {threshold}")
        print(f"{'─'*80}")
        
        metrics = pipeline.query_with_metrics(
            text_query=target,
            count_threshold=threshold,
            eval_videos=eval_videos,
            data_dir=data_dir
        )
        
        pipeline_results[threshold] = metrics
        
        print(f"\n[Results]")
        print(f"  Total latency: {metrics['total_latency_ms']:.2f} ms")
        print(f"  Recall: {metrics['mlp_recall']:.3f}")
        print(f"  F1: {metrics['mlp_f1']:.3f}")
    
    print("\n" + "="*80)
    print("EMBEDDING BENCHMARK COMPLETE")
    print("="*80 + "\n")
    
    return pipeline_results


def benchmark_with_kf(
    kf_method: str,
    data_dir: str = "data/VIRAT",
    checkpoint_path: str = "models/checkpoints/car_virat_finetuned.pth",
    model_config=LARGE3,
    target: str = "car",
    similarity_threshold: float = 0.2,
    num_videos: int = 5,
    thresholds: list[int] = [0, 1, 2, 3, 4, 5],
    kf_params: dict = {},
    force_regenerate: bool = False,
    videos_source_dir: str = None,
    embedder_config=CLIP_VIT_B32
):
    """
    Benchmark keyframe-based FAISS+MLP pipeline.
    
    Args:
        kf_method: Keyframe selector name ('emb_novelty', 'ssim_flow', 'kcenter')
        data_dir: Directory containing video data
        checkpoint_path: Path to trained model checkpoint
        model_config: Model configuration
        target: Object class to detect ("car" or "person")
        similarity_threshold: FAISS similarity threshold
        num_videos: Number of videos to test on
        thresholds: List of count thresholds to test
        kf_params: Parameters for the keyframe selector
        force_regenerate: If True, regenerate keyframe embeddings
        videos_source_dir: Directory containing source .mp4 video files
        embedder_config: Embedder configuration
    Returns:
        Dictionary with keyframe metadata and pipeline results for each threshold
    """
    print("\n" + "="*80)
    print(f"BENCHMARKING WITH KEYFRAMES: {kf_method.upper()}")
    print("="*80)
    print(f"Target: {target}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Videos: {num_videos}")
    print(f"Thresholds: {thresholds}")
    print(f"Keyframe method: {kf_method}")
    print(f"Keyframe params: {kf_params}")
    print("="*80 + "\n")
    
    data_path = Path(data_dir)
    video_dirs = sorted([d for d in data_path.iterdir() 
                        if d.is_dir() and not d.name.startswith("_")])
    eval_videos = [v.name for v in video_dirs[:num_videos]]
    
    print(f"Generating keyframe embeddings using {kf_method}...")
    
    kf_metadata = process_video_folder(
        data_dir=data_dir,
        videos_source_dir=videos_source_dir,
        selector_name=kf_method,
        selector_params=kf_params,
        embedder_config=embedder_config,
        num_videos=num_videos,
        force_regenerate=force_regenerate
    )
    
    # Rebuild FAISS index with keyframe embeddings
    print(f"\nRebuilding FAISS index with keyframe embeddings...")
    faiss_index = FAISSIndex(data_dir)
    faiss_index.build()
    
    # Initialize pipeline with keyframe embeddings
    print(f"\nInitializing FAISS+MLP pipeline with keyframes...")
    pipeline = SemanticQueryPipeline(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        threshold=similarity_threshold
    )
    
    pipeline_results = {}
    
    for threshold in thresholds:
        print(f"\n{'─'*80}")
        print(f"Testing with count threshold >= {threshold}")
        print(f"{'─'*80}")
        
        metrics = pipeline.query_with_metrics(
            text_query=target,
            count_threshold=threshold,
            eval_videos=eval_videos,
            data_dir=data_dir
        )
        
        pipeline_results[threshold] = metrics
        
        print(f"\n[Results]")
        print(f"  Total latency: {metrics['total_latency_ms']:.2f} ms")
        print(f"  Recall: {metrics['mlp_recall']:.3f}")
        print(f"  F1: {metrics['mlp_f1']:.3f}")
    
    print("\n" + "="*80)
    print("KEYFRAME BENCHMARK COMPLETE")
    print("="*80 + "\n")
    
    return {
        'metadata': kf_metadata,
        'pipeline_results': pipeline_results
    }

