"""Modular benchmark functions for testing different pipeline configurations."""

from pathlib import Path
from src.models.model_configs import LARGE3
from src.embeddings.embedder import CLIP_VIT_B32
from src.query.semantic_query import SemanticQueryPipeline
from src.baseline import evaluate_baseline_yolo
from src.pipeline.keyframe_pipeline import process_video_folder
from src.indexing.faiss_index import FAISSIndex


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
                     kf_params: dict = {}, force_regenerate: bool = False,
                     videos_source_dir: str = None, embedder_config=CLIP_VIT_B32):
    """Benchmark keyframe-based FAISS+MLP pipeline."""
    print(f"\n[Keyframe Pipeline: {kf_method}]")
    
    data_path = Path(data_dir)
    video_dirs = sorted([d for d in data_path.iterdir() 
                        if d.is_dir() and not d.name.startswith("_")])
    eval_videos = [v.name for v in video_dirs[:num_videos]]
    
    kf_metadata = process_video_folder(
        data_dir=data_dir,
        videos_source_dir=videos_source_dir,
        method=kf_method,
        method_params=kf_params,
        embedder_config=embedder_config,
        num_videos=num_videos,
        force_regenerate=force_regenerate
    )
    
    if kf_metadata:
        avg_comp = sum(m['compression_ratio'] for m in kf_metadata) / len(kf_metadata)
        print(f"  Generated keyframes: {avg_comp:.1f}x compression")
    
    faiss_index = FAISSIndex(data_dir)
    faiss_index.build()
    
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
        'metadata': kf_metadata,
        'pipeline_results': pipeline_results
    }
