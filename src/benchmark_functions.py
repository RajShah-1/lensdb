"""Benchmark functions for testing different pipeline configurations."""

from pathlib import Path
import numpy as np

from src.embeddings.embedder import CLIPEmbedder
from src.query.semantic_query import SemanticQueryPipeline
from src.baseline import evaluate_baseline_yolo
from src.indexing.faiss_index import FAISSIndex
from src.pipeline.embedding_pipeline import generate_full_embeddings, select_keyframes_from_full
from src.keyframe.preselect_methods import FrameDiffPreselector, SSIMPreselector, MOG2Preselector, FlowPreselector


def get_preselector(method: str, **kwargs):
    """Get preselector by name."""
    selectors = {
        'framediff': FrameDiffPreselector,
        'ssim': SSIMPreselector,
        'mog2': MOG2Preselector,
        'flow': FlowPreselector
    }
    if method not in selectors:
        raise ValueError(f"Unknown method: {method}")
    return selectors[method](**kwargs)


def benchmark_baseline(data_dir: str, target: str, num_videos: int, 
                      thresholds: list[int], yolo_model: str):
    """Benchmark YOLO baseline."""
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
        print(f"  T>={threshold}: R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, "
              f"Lat={metrics['avg_latency_ms']:.1f}ms")
    
    return baseline_results


def benchmark_embds(data_dir: str, checkpoint_path: str, model_config, target: str,
                   similarity_threshold: float, num_videos: int, thresholds: list[int]):
    """Benchmark standard embedding pipeline (no keyframes)."""
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
        print(f"  T>={threshold}: R={metrics['mlp_recall']:.3f}, F1={metrics['mlp_f1']:.3f}, "
              f"Lat={metrics['total_latency_ms']:.1f}ms")
    
    return pipeline_results


def benchmark_with_kf(kf_method: str, kf_params: dict, data_dir: str, checkpoint_path: str,
                     model_config, target: str, similarity_threshold: float, num_videos: int,
                     thresholds: list[int], videos_source_dir: str, embedder: CLIPEmbedder,
                     force_regenerate: bool):
    """Benchmark keyframe-based pipeline."""
    print(f"\n[Keyframe Pipeline: {kf_method}]")
    print(f"  Device: {embedder.device}")
    
    data_path = Path(data_dir)
    video_dirs = sorted([d for d in data_path.iterdir() 
                        if d.is_dir() and not d.name.startswith("_")])[:num_videos]
    eval_videos = [v.name for v in video_dirs]
    
    preselector = get_preselector(kf_method, **kf_params)
    results = []
    
    for video_dir in video_dirs:
        video_path = Path(videos_source_dir) / f"{video_dir.name}.mp4"
        if not video_path.exists():
            continue
        
        print(f"  {video_dir.name}")
        
        generate_full_embeddings(
            video_path=str(video_path),
            out_dir=str(video_dir),
            embedder=embedder,
            force=force_regenerate
        )
        
        metadata = select_keyframes_from_full(
            video_path=str(video_path),
            out_dir=str(video_dir),
            preselector=preselector,
            embedder=embedder,
            force=force_regenerate
        )
        results.append(metadata)
    
    if results:
        avg_comp = sum(r['compression_ratio'] for r in results) / len(results)
        print(f"  Avg compression: {avg_comp:.1f}x")
    
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
        print(f"  T>={threshold}: R={metrics['mlp_recall']:.3f}, F1={metrics['mlp_f1']:.3f}, "
              f"Lat={metrics['total_latency_ms']:.1f}ms")
    
    return {
        'metadata': results,
        'pipeline_results': pipeline_results
    }
