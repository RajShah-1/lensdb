import csv
from pathlib import Path

import numpy as np

from src.embeddings.embedder import CLIPEmbedder, CLIP_VIT_B32
from src.models.model_configs import MEDIUM, SMALL, LARGE, LARGE3
from src.pipeline.detection_pipeline import DetectionPipeline
from src.pipeline.embedding_pipeline import generate_embeddings, get_preselector
from src.detectors.object_detector import ObjectDetector
from src.training.train_pipeline import finetune_on_virat, pretrain_on_coco
from src.indexing.faiss_index import FAISSIndex
from src.query.semantic_query import SemanticQueryPipeline
from src.baseline import evaluate_baseline_yolo
from src.benchmark_functions import benchmark_baseline, benchmark_embds, benchmark_with_kf
from src.comprehensive_test import run_comprehensive_tests

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

def gen_embeddings_for_dir(videos_dir: str, data_dir: str = "data/VIRAT",
                          method: str = None, method_params: dict = None):
    """Generate embeddings for videos (with optional keyframe preselection)."""
    videos = sorted(Path(videos_dir).glob("*.mp4"))
    print(f"Found {len(videos)} videos in {videos_dir}")
    
    preselector = get_preselector(method, **(method_params or {})) if method else None
    
    for vid in videos:
        print(f"Processing {vid.name}")
        out_dir = Path(data_dir) / vid.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        
        generate_embeddings(
            video_path=str(vid),
            out_dir=str(out_dir),
            preselector=preselector,
            embedder_config=CLIP_VIT_B32
        )

def build_index(data_dir="data/VIRAT"):
    index = FAISSIndex(data_dir)
    index.build()

def run_semantic_query():
    pipeline = SemanticQueryPipeline(
        data_dir="data/VIRAT",
        checkpoint_path="models/checkpoints/car_virat_finetuned.pth",
        model_config=MEDIUM,
        threshold=0.2
    )
    return pipeline.query(text_query="car", count_predicate=lambda c: c >= 2.0)

def load_ground_truth(data_dir: str, video_name: str, target: str = "car"):
    counts_file = Path(data_dir) / video_name / "counts.csv"
    if not counts_file.exists():
        raise FileNotFoundError(f"Missing {counts_file}")
    
    counts = {}
    with open(counts_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_id = int(row['frame_id'])
            counts[frame_id] = int(row['car_count' if target == "car" else 'people_count'])
    return counts

def evaluate_retrieval(data_dir: str, checkpoint_path: str, model_config,
                       target: str = "car", count_threshold: int = 2,
                       similarity_threshold: float = 0.2, num_videos: int = 2):
    data_path = Path(data_dir)
    video_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith("_")])
    
    if len(video_dirs) < num_videos:
        raise ValueError(f"Only {len(video_dirs)} videos available, need {num_videos}")
    
    eval_videos = [v.name for v in video_dirs[:num_videos]]
    
    print(f"\n{'='*70}")
    print(f"EVALUATION: {target} (count >= {count_threshold})")
    print(f"Videos: {', '.join(eval_videos)}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"{'='*70}")
    
    pipeline = SemanticQueryPipeline(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        threshold=similarity_threshold
    )
    
    metrics = pipeline.query_with_metrics(
        text_query=target,
        count_threshold=count_threshold,
        eval_videos=eval_videos,
        data_dir=data_dir
    )
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    
    print(f"\n[LATENCY]")
    print(f"  FAISS:  {metrics['faiss_latency_ms']:.2f} ms")
    print(f"  MLP:    {metrics['mlp_latency_ms']:.2f} ms")
    print(f"  Total:  {metrics['total_latency_ms']:.2f} ms")
    
    print(f"\n[FAISS STAGE]")
    print(f"  Precision: {metrics['faiss_precision']:.3f}")
    print(f"  Recall:    {metrics['faiss_recall']:.3f}")
    print(f"  F1:        {metrics['faiss_f1']:.3f}")
    print(f"  Retrieved: {metrics['faiss_retrieved']}")
    
    print(f"\n[FINAL PIPELINE]")
    print(f"  Precision: {metrics['mlp_precision']:.3f}")
    print(f"  Recall:    {metrics['mlp_recall']:.3f}")
    print(f"  F1:        {metrics['mlp_f1']:.3f}")
    print(f"  Retrieved: {metrics['mlp_retrieved']}")
    
    print(f"\n{'='*70}")
    
    return metrics


# ============================================================================
# BENCHMARK FUNCTIONS (modular testing interface)
# ============================================================================

def run_baseline_benchmark():
    """
    Run YOLO baseline benchmark only.
    Quick way to test baseline performance.
    """
    results = benchmark_baseline(
        data_dir="data/VIRAT",
        target="car",
        num_videos=5,
        thresholds=[0, 1, 2],
        yolo_model="yolo11m"
    )
    return results


def run_embedding_benchmark():
    """
    Run embedding-based pipeline benchmark (no keyframes).
    Tests standard FAISS+MLP performance.
    """
    results = benchmark_embds(
        data_dir="data/VIRAT",
        checkpoint_path="models/checkpoints/car_virat_finetuned.pth",
        model_config=LARGE3,
        target="car",
        similarity_threshold=0.2,
        num_videos=5,
        thresholds=[0, 1, 2]
    )
    return results


def run_keyframe_benchmark(method="framediff"):
    """
    Run keyframe-based pipeline benchmark.
    
    Args:
        method: Preselector method ('framediff', 'ssim', 'mog2', 'flow')
    """
    results = benchmark_with_kf(
        kf_method=method,
        data_dir="data/VIRAT",
        checkpoint_path="models/checkpoints/car_virat_finetuned.pth",
        model_config=LARGE3,
        target="car",
        similarity_threshold=0.2,
        num_videos=5,
        thresholds=[0, 1, 2],
        kf_params={'k_mad': 2.5, 'min_spacing': 6},
        force_regenerate=False,
        videos_source_dir="/storage/ice1/8/3/rshah647/VIRATGround/videos_original"
    )
    return results


def run_full_comparison():
    """
    Run comprehensive comparison of all methods.
    Includes baseline, embeddings, and keyframe-based approaches.
    """
    results = run_comprehensive_tests(
        data_dir="data/VIRAT",
        checkpoint_path="models/checkpoints/car_virat_finetuned.pth",
        model_config=LARGE3,
        target="car",
        similarity_threshold=0.2,
        num_videos=5,
        thresholds=[0, 1, 2, 3, 4, 5],
        yolo_model="yolo11m",
        output_file="results/comprehensive_test_results.json",
        test_keyframes=True,
        keyframe_selectors=['framediff', 'ssim', 'flow'],
        keyframe_params={
            'framediff': {'k_mad': 2.5, 'min_spacing': 6},
            'ssim': {'k_mad': 2.5, 'min_spacing': 6},
            'flow': {'k_mad': 2.5, 'min_spacing': 6}
        },
        force_regenerate_keyframes=False,
        videos_source_dir="/storage/ice1/8/3/rshah647/VIRATGround/videos_original"
    )
    return results


if __name__ == "__main__":
    run_keyframe_benchmark()



