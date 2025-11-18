"""Main entry points for LensDB pipeline."""

from pathlib import Path
from src.embeddings.embedder import CLIPEmbedder, CLIP_VIT_B32
from src.models.model_configs import LARGE3
from src.baseline import evaluate_baseline_yolo
from src.benchmark_functions import benchmark_baseline, benchmark_embds, benchmark_with_kf
from src.comprehensive_test import run_comprehensive_tests


def run_baseline_benchmark():
    """Run YOLO baseline benchmark."""
    return benchmark_baseline(
        data_dir="data/VIRAT",
        target="car",
        num_videos=5,
        thresholds=[0, 1, 2],
        yolo_model="yolo11m"
    )


def run_embedding_benchmark():
    """Run standard embedding pipeline benchmark."""
    return benchmark_embds(
        data_dir="data/VIRAT",
        checkpoint_path="models/checkpoints/car_virat_finetuned.pth",
        model_config=LARGE3,
        target="car",
        similarity_threshold=0.2,
        num_videos=5,
        thresholds=[0, 1, 2]
    )


def run_keyframe_benchmark():
    """Run keyframe-based pipeline benchmark."""
    embedder = CLIPEmbedder(CLIP_VIT_B32)
    
    return benchmark_with_kf(
        kf_method="framediff",
        kf_params={'k_mad': 2.5, 'min_spacing': 6},
        data_dir="data/VIRAT",
        checkpoint_path="models/checkpoints/car_virat_finetuned.pth",
        model_config=LARGE3,
        target="car",
        similarity_threshold=0.2,
        num_videos=5,
        thresholds=[0, 1, 2],
        videos_source_dir="/storage/ice1/8/3/rshah647/VIRATGround/videos_original",
        embedder=embedder,
        force_regenerate=False,
        save_keyframes=False
    )


def run_full_comparison():
    """Run comprehensive comparison of all methods."""
    return run_comprehensive_tests(
        data_dir="data/VIRAT",
        checkpoint_path="models/checkpoints/car_virat_finetuned.pth",
        model_config=LARGE3,
        target="car",
        similarity_threshold=0.2,
        num_videos=5,
        thresholds=[0, 1, 2, 3, 4, 5],
        yolo_model="yolo11m",
        output_file="results/comprehensive_test_results.json",
        videos_source_dir="/storage/ice1/8/3/rshah647/VIRATGround/videos_original",
        test_keyframes=True,
        force_regenerate_keyframes=False,
        save_keyframes=False,
        keyframe_selectors=['framediff', 'ssim', 'flow'],
        keyframe_params={
            'framediff': {'k_mad': 2.5, 'min_spacing': 6},
            'ssim': {'k_mad': 2.5, 'min_spacing': 6},
            'flow': {'k_mad': 2.5, 'min_spacing': 6}
        }
    )


if __name__ == "__main__":
    run_keyframe_benchmark()
