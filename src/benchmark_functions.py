"""Benchmark functions for testing different pipeline configurations."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        print(f"    TP={metrics['mlp_tp']}, FP={metrics['mlp_fp']}, FN={metrics['mlp_fn']}")
    
    return pipeline_results


def benchmark_with_kf(
    kf_method: str,
    kf_params: dict,
    data_dir: str,
    checkpoint_path: str,
    model_config,
    target: str,
    similarity_threshold: float,
    num_videos: int,
    thresholds: list[int],
    videos_source_dir: str,
    embedder: CLIPEmbedder,
    force_regenerate: bool,
    save_keyframes: bool,
    num_workers: int = 4,
):
    """
    Benchmark keyframe-based pipeline, processing videos in parallel.

    Per-video work (embedding + keyframe selection) is done in a thread pool.
    FAISS index building and query evaluation are still done once at the end.
    """
    print(f"\n[Keyframe Pipeline: {kf_method}]")
    print(f"  Device: {embedder.device}")
    print(f"  Num workers: {num_workers}")

    data_path = Path(data_dir)
    video_dirs = sorted(
        [
            d for d in data_path.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ]
    )[:num_videos]
    eval_videos = [v.name for v in video_dirs]

    results = []

    def process_one_video(video_dir: Path):
        """Worker: run full-embedding + keyframe selection on a single video."""

        video_path = Path(videos_source_dir) / f"{video_dir.name}.mp4"
        if not video_path.exists():
            print(f"  [WARN] Missing video file for {video_dir.name}: {video_path}")
            return None

        # IMPORTANT: new preselector per video (stateful, not thread-safe to share)
        preselector = get_preselector(kf_method, **kf_params)

        print(f"  [{video_dir.name}] starting in worker")

        # 1) Dense 1 FPS embeddings (will skip if already exist and force_regenerate=False)
        generate_full_embeddings(
            video_path=str(video_path),
            out_dir=str(video_dir),
            embedder=embedder,
            target_fps=1.0,
            force=force_regenerate,
        )

        # 2) Run keyframe selection from dense stream
        metadata = select_keyframes_from_full(
            video_path=str(video_path),
            out_dir=str(video_dir),
            preselector=preselector,
            embedder=embedder,
            target_fps=1.0,
            force=force_regenerate,
            save_keyframes=save_keyframes,
        )

        print(
            f"  [{video_dir.name}] done: "
            f"compression={metadata.get('compression_ratio', 'NA')}x"
        )
        return metadata

    # ---------- Parallel per-video section ----------
    if num_workers <= 1:
        # Fallback: sequential
        for vd in video_dirs:
            md = process_one_video(vd)
            if md is not None:
                results.append(md)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            future_to_vd = {
                ex.submit(process_one_video, vd): vd.name for vd in video_dirs
            }
            for fut in as_completed(future_to_vd):
                vd_name = future_to_vd[fut]
                try:
                    md = fut.result()
                    if md is not None:
                        results.append(md)
                except Exception as e:
                    print(f"  [ERROR] Video {vd_name} failed: {e}")

    # ---------- Aggregate compression stats ----------
    if results:
        avg_comp = sum(r.get("compression_ratio", 1.0) for r in results) / len(results)
        print(f"  Avg compression: {avg_comp:.1f}x (over {len(results)} videos)")

    # ---------- Build FAISS index over resulting embeddings ----------
    faiss_index = FAISSIndex(data_dir)
    faiss_index.build()

    # ---------- Run query pipeline ----------
    pipeline = SemanticQueryPipeline(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        threshold=similarity_threshold,
    )

    pipeline_results = {}

    for threshold in thresholds:
        metrics = pipeline.query_with_metrics(
            text_query=target,
            count_threshold=threshold,
            eval_videos=eval_videos,
            data_dir=data_dir,
        )
        pipeline_results[threshold] = metrics
        print(f"  T>={threshold}: R={metrics['mlp_recall']:.3f}, F1={metrics['mlp_f1']:.3f}, "
              f"Lat={metrics['total_latency_ms']:.1f}ms")
        print(f"    TP={metrics['mlp_tp']}, FP={metrics['mlp_fp']}, FN={metrics['mlp_fn']}")
        if metrics.get('positive_kf_indices'):
            print(f"    Positive Frames: {', '.join(map(str, metrics['positive_kf_indices']))}")
        if metrics.get('negative_kf_indices'):
            print(f"    Negative Frames: {', '.join(map(str, metrics['negative_kf_indices']))}")
    
    return {
        "metadata": results,
        "pipeline_results": pipeline_results,
    }


# def benchmark_with_kf(kf_method: str, kf_params: dict, data_dir: str, checkpoint_path: str,
#                      model_config, target: str, similarity_threshold: float, num_videos: int,
#                      thresholds: list[int], videos_source_dir: str, embedder: CLIPEmbedder,
#                      force_regenerate: bool, save_keyframes: bool):
#     """Benchmark keyframe-based pipeline."""
#     print(f"\n[Keyframe Pipeline: {kf_method}]")
#     print(f"  Device: {embedder.device}")
    
#     data_path = Path(data_dir)
#     video_dirs = sorted([d for d in data_path.iterdir() 
#                         if d.is_dir() and not d.name.startswith("_")])[:num_videos]
#     eval_videos = [v.name for v in video_dirs]
    
#     preselector = get_preselector(kf_method, **kf_params)
#     results = []
    
#     for video_dir in video_dirs:
#         video_path = Path(videos_source_dir) / f"{video_dir.name}.mp4"
#         if not video_path.exists():
#             continue
        
#         print(f"  {video_dir.name}")
        
#         generate_full_embeddings(
#             video_path=str(video_path),
#             out_dir=str(video_dir),
#             embedder=embedder,
#             target_fps=1.0,
#             force=force_regenerate
#         )
        
#         metadata = select_keyframes_from_full(
#             video_path=str(video_path),
#             out_dir=str(video_dir),
#             preselector=preselector,
#             embedder=embedder,
#             target_fps=1.0,
#             force=force_regenerate,
#             save_keyframes=save_keyframes
#         )
#         results.append(metadata)
    
#     if results:
#         avg_comp = sum(r['compression_ratio'] for r in results) / len(results)
#         print(f"  Avg compression: {avg_comp:.1f}x")
    
#     faiss_index = FAISSIndex(data_dir)
#     faiss_index.build()
    
#     pipeline = SemanticQueryPipeline(
#         data_dir=data_dir,
#         checkpoint_path=checkpoint_path,
#         model_config=model_config,
#         threshold=similarity_threshold
#     )
    
#     pipeline_results = {}
    
#     for threshold in thresholds:
#         metrics = pipeline.query_with_metrics(
#             text_query=target,
#             count_threshold=threshold,
#             eval_videos=eval_videos,
#             data_dir=data_dir
#         )
#         pipeline_results[threshold] = metrics
#         print(f"  T>={threshold}: R={metrics['mlp_recall']:.3f}, F1={metrics['mlp_f1']:.3f}, "
#               f"Lat={metrics['total_latency_ms']:.1f}ms")
    
#     return {
#         'metadata': results,
#         'pipeline_results': pipeline_results
#     }
