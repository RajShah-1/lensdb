"""Benchmark functions for testing different pipeline configurations."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
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
                      thresholds: list[int], yolo_model: str,
                      videos_source_dir: str):
    """Benchmark YOLO baseline."""
    print(f"\n[YOLO Baseline: {yolo_model}]")
    baseline_results = {}
    
    for threshold in thresholds:
        metrics = evaluate_baseline_yolo(
            data_dir=data_dir,
            videos_source_dir=videos_source_dir,
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

    # ---------- Build FAISS index over resulting embeddings ----------
    print(f"\nBuilding FAISS index...")
    faiss_index = FAISSIndex(data_dir, use_keyframes=False)
    faiss_index.build()
    
    pipeline = SemanticQueryPipeline(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        threshold=similarity_threshold,
        use_keyframes=False
    )
    
    pipeline_results = {}
    
    for threshold in thresholds:
        metrics = pipeline.query_with_metrics(
            text_query=target,
            count_threshold=threshold,
            eval_videos=eval_videos,
            data_dir=data_dir,
            emb_filename="embds_clip_full.npy",
        )
        pipeline_results[threshold] = metrics
        print(f"  T>={threshold}: R={metrics['mlp_recall']:.3f}, F1={metrics['mlp_f1']:.3f}, "
              f"Lat={metrics['total_latency_ms']:.1f}ms")
        print(f"    TP={metrics['mlp_tp']}, FP={metrics['mlp_fp']}, FN={metrics['mlp_fn']}")
    
    faiss_index.clean_index()
    return pipeline_results


def _process_video_worker(args):
    """
    Worker function for multiprocessing. 
    Creates embedder fresh in each process to avoid CUDA context issues.
    
    Args: (video_dir_str, video_name, videos_source_dir, kf_method, kf_params,
           embedder_config, target_fps, force_regenerate, force_regenerate_kf, save_keyframes, worker_id, log_file)
    """
    import sys
    import traceback
    import os as os_module
    
    try:
        (video_dir_str, video_name, videos_source_dir, kf_method, kf_params,
         embedder_config, target_fps, force_regenerate, force_regenerate_kf, save_keyframes, worker_id, log_file) = args
        
        # Redirect stdout/stderr to log file
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            sys.stdout = open(log_file, 'w', buffering=1)
            sys.stderr = sys.stdout
        
        from src.embeddings.embedder import CLIPEmbedder
        
        video_dir = Path(video_dir_str)
        video_path = Path(videos_source_dir) / f"{video_name}.mp4"
        
        print(f"[Worker {worker_id}] Processing {video_name}, PID={os_module.getpid()}", flush=True)
        
        if not video_path.exists():
            print(f"[Worker {worker_id}] WARN: Missing video file: {video_path}", flush=True)
            return None
        
        # Create fresh embedder in this process with config dict
        from src.embeddings.embedder import EmbedderConfig
        cfg = EmbedderConfig(**embedder_config)
        embedder = CLIPEmbedder(cfg)
        preselector = get_preselector(kf_method, **kf_params)
        
        print(f"[Worker {worker_id}] Starting embedding generation", flush=True)
        
        # 1) Dense embeddings
        generate_full_embeddings(
            video_path=str(video_path),
            out_dir=str(video_dir),
            embedder=embedder,
            target_fps=target_fps,
            force=force_regenerate,
        )
        
        print(f"[Worker {worker_id}] Starting keyframe selection", flush=True)
        
        # 2) Keyframe selection
        metadata = select_keyframes_from_full(
            video_path=str(video_path),
            out_dir=str(video_dir),
            preselector=preselector,
            embedder=embedder,
            target_fps=target_fps,
            force=force_regenerate_kf, 
            save_keyframes=save_keyframes,
        )
        
        print(f"[Worker {worker_id}] Done: compression={metadata.get('compression_ratio', 'NA')}x", flush=True)
        return metadata
    
    except Exception as e:
        worker_id_str = args[9] if len(args) > 9 else 'unknown'
        print(f"[Worker {worker_id_str}] ERROR: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        return None
    finally:
        # Restore stdout/stderr if redirected
        try:
            if 'log_file' in locals() and log_file:
                if hasattr(sys.stdout, 'close') and sys.stdout != sys.__stdout__:
                    sys.stdout.close()
                sys.stderr = sys.__stderr__
                sys.stdout = sys.__stdout__
        except:
            pass


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
    force_regenerate_kf: bool,
    save_keyframes: bool,
    num_workers: int = 1,
):
    """
    Benchmark keyframe-based pipeline, processing videos in parallel.

    Per-video work (embedding + keyframe selection) is done in separate processes.
    FAISS index building and query evaluation are done once at the end.
    """
    print(f"\n[Keyframe Pipeline: {kf_method}]")
    print(f"  Device: {embedder.device}")
    print(f"  Num workers: {num_workers}")

    # Discover videos from source directory
    videos_source_path = Path(videos_source_dir)
    video_files = sorted([f for f in videos_source_path.glob("*.mp4")])[:num_videos]
    
    print(f"  Found {len(video_files)} videos in {videos_source_dir}")
    
    # Create data directories and prepare video info
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    video_dirs = []
    for video_file in video_files:
        video_name = video_file.stem  # filename without extension
        video_dir = data_path / video_name
        video_dir.mkdir(parents=True, exist_ok=True)
        video_dirs.append(video_dir)
    
    eval_videos = [vd.name for vd in video_dirs]

    results = []
    
    # Create log directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Logs will be saved to: {log_dir}")

    # Get the embedder config that was used (we need to pass it to workers)
    # The main process has the embedder instance, we need to get the original config
    from src.embeddings.embedder import CLIP_VIT_B32, MOBILE_CLIP_VIT_PATCH16
    
    # Determine which config was used based on embedder.name
    embedder_cfg = CLIP_VIT_B32 if embedder.name == "clip" else MOBILE_CLIP_VIT_PATCH16
    
    # Serialize embedder config for workers
    embedder_config_dict = {
        'name': embedder_cfg.name,
        'processor_name': embedder_cfg.processor_name,
        'model_name': embedder_cfg.model_name,
    }
    
    # ---------- Sequential processing of per-video section ----------
    print(f"  Running sequentially (num_workers={num_workers})")
    for idx, vd in enumerate(video_dirs, 1):
        log_file = str(log_dir / f"worker_{idx}.log")
        args = (str(vd), vd.name, videos_source_dir, kf_method, kf_params,
                embedder_config_dict, 1.0, force_regenerate, force_regenerate_kf, save_keyframes, idx, log_file)
        md = _process_video_worker(args)
        if md is not None:
            results.append(md)

    # ---------- Aggregate compression stats ----------
    if results:
        avg_comp = sum(r.get("compression_ratio", 1.0) for r in results) / len(results)
        print(f"  Avg compression: {avg_comp:.1f}x (over {len(results)} videos)")
    else:
        print(f"  [WARNING] No results collected from workers! Check if videos were processed.")

    # ---------- Build FAISS index over resulting embeddings ----------
    print(f"\nBuilding FAISS index...")
    faiss_index = FAISSIndex(data_dir, use_keyframes=True)
    faiss_index.build()

    # ---------- Run query pipeline ----------
    pipeline = SemanticQueryPipeline(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        threshold=similarity_threshold,
        use_keyframes=True,
    )

    pipeline_results = {}

    for threshold in thresholds:
        metrics = pipeline.query_with_metrics(
            text_query=target,
            count_threshold=threshold,
            eval_videos=eval_videos,
            data_dir=data_dir,
            emb_filename="embds.npy",
        )
        pipeline_results[threshold] = metrics
        print(f"  T>={threshold}: R={metrics['mlp_recall']:.3f}, F1={metrics['mlp_f1']:.3f}, "
              f"Lat={metrics['total_latency_ms']:.1f}ms")
        print(f"    TP={metrics['mlp_tp']}, FP={metrics['mlp_fp']}, FN={metrics['mlp_fn']}")
        if metrics.get('positive_kf_indices'):
            print(f"    Positive Frames: {', '.join(map(str, metrics['positive_kf_indices']))}")
        if metrics.get('negative_kf_indices'):
            print(f"    Negative Frames: {', '.join(map(str, metrics['negative_kf_indices']))}")
    
    faiss_index.clean_index()
    return {
        "metadata": results,
        "pipeline_results": pipeline_results,
    }
