"""Modular benchmark functions for testing different pipeline configurations."""

from pathlib import Path
import numpy as np

from src.models.model_configs import LARGE3
from src.embeddings.embedder import CLIP_VIT_B32
from src.query.semantic_query import SemanticQueryPipeline
from src.baseline import evaluate_baseline_yolo
from src.indexing.faiss_index import FAISSIndex
from src.pipeline.embedding_pipeline import generate_embeddings, get_preselector


def benchmark_baseline(data_dir: str = "data/VIRAT", target: str = "car", num_videos: int = 5,
                      thresholds: list[int] = [0, 1, 2, 3, 4, 5], yolo_model: str = "yolo11m"):
    """Benchmark YOLO baseline performance."""
    print(f"\n[YOLO Baseline: {yolo_model}]")
    baseline_results = {}
    
    for threshold in thresholds:
        metrics = evaluate_baseline_yolo(
            data_dir=data_dir,
            model_name=yolo_model,
            target=target,
            count_threshold=threshold,
            num_videos=num_videos
        )
        baseline_results[threshold] = metrics
        print(f"  Threshold >={threshold}: Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, "
              f"Latency={metrics['avg_latency_ms']:.1f}ms")
    
    return baseline_results


def benchmark_embds(data_dir: str = "data/VIRAT",
                   checkpoint_path: str = "models/checkpoints/car_virat_finetuned.pth",
                   model_config=LARGE3, target: str = "car", similarity_threshold: float = 0.2,
                   num_videos: int = 5, thresholds: list[int] = [0, 1, 2, 3, 4, 5]):
    """Benchmark standard embedding-based FAISS+MLP pipeline (no keyframes)."""
    print(f"\n[Embedding Pipeline: Standard]")
    
    data_path = Path(data_dir)
    video_dirs = sorted([d for d in data_path.iterdir() 
                        if d.is_dir() and not d.name.startswith("_")])
    eval_videos = [v.name for v in video_dirs[:num_videos]]
    
    pipeline = SemanticQueryPipeline(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        threshold=similarity_threshold
    )
    
    pipeline_results = {}
    
    for threshold in thresholds:
        metrics = pipeline.query_with_metrics(
            text_query=target,
            count_threshold=threshold,
            eval_videos=eval_videos,
            data_dir=data_dir
        )
        pipeline_results[threshold] = metrics
        print(f"  Threshold >={threshold}: Recall={metrics['mlp_recall']:.3f}, F1={metrics['mlp_f1']:.3f}, "
              f"Latency={metrics['total_latency_ms']:.1f}ms")
    
    return pipeline_results


def benchmark_with_kf(kf_method: str, data_dir: str = "data/VIRAT",
                     checkpoint_path: str = "models/checkpoints/car_virat_finetuned.pth",
                     model_config=LARGE3, target: str = "car", similarity_threshold: float = 0.2,
                     num_videos: int = 5, thresholds: list[int] = [0, 1, 2, 3, 4, 5],
                     kf_params: dict = {}, force_regenerate: bool = True,
                     videos_source_dir: str = None, embedder_config=CLIP_VIT_B32,
                     target_fps: float = 1.0):
    """Benchmark keyframe-based FAISS+MLP pipeline."""
    print(f"\n[Keyframe Pipeline: {kf_method}]")
    
    data_path = Path(data_dir)
    video_dirs = sorted([d for d in data_path.iterdir() 
                        if d.is_dir() and not d.name.startswith("_")])
    eval_videos = [v.name for v in video_dirs[:num_videos]]
    
    if num_videos is not None:
        video_dirs = video_dirs[:num_videos]
    
    # Generate keyframe embeddings using preselector
    preselector = get_preselector(kf_method, **(kf_params or {}))
    results = []
    
    for video_dir in video_dirs:
        video_path = Path(videos_source_dir) / f"{video_dir.name}.mp4"
        if not video_path.exists():
            continue
        
        embeddings_dir = video_dir / "embeddings"
        metadata_file = embeddings_dir / "metadata.npy"
        
        # Skip if already processed
        if metadata_file.exists() and not force_regenerate:
            metadata = np.load(metadata_file, allow_pickle=True).item()
            if metadata.get('uses_keyframes', False):
                results.append(metadata)
                continue
        
        # Generate embeddings with preselector
        metadata = generate_embeddings(
            video_path=str(video_path),
            out_dir=str(video_dir),
            preselector=preselector,
            embedder_config=embedder_config,
            target_fps=target_fps
        )
        results.append(metadata)
    
    if results:
        avg_comp = sum(r['compression_ratio'] for r in results) / len(results)
        print(f"  Generated keyframes: {avg_comp:.1f}x compression")
    
    # Rebuild FAISS index with keyframe embeddings
    faiss_index = FAISSIndex(data_dir)
    faiss_index.build()
    
    # Run pipeline with keyframe embeddings (expansion happens automatically in query)
    pipeline = SemanticQueryPipeline(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        threshold=similarity_threshold
    )
    
    pipeline_results = {}
    
    for threshold in thresholds:
        metrics = pipeline.query_with_metrics(
            text_query=target,
            count_threshold=threshold,
            eval_videos=eval_videos,
            data_dir=data_dir
        )
        pipeline_results[threshold] = metrics
        print(f"  Threshold >={threshold}: Recall={metrics['mlp_recall']:.3f}, F1={metrics['mlp_f1']:.3f}, "
              f"Latency={metrics['total_latency_ms']:.1f}ms")
    
    return {
        'metadata': results,
        'pipeline_results': pipeline_results
    }
