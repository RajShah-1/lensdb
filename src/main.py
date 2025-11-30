"""Main entry points for LensDB pipeline."""

import multiprocessing
import time
# MUST set spawn before any CUDA initialization for multiprocessing to work
multiprocessing.set_start_method('spawn', force=True)

from pathlib import Path
from src.embeddings.embedder import CLIPEmbedder, CLIP_VIT_B32
from src.models.model_configs import LARGE3
from src.baseline import evaluate_baseline_yolo
from src.benchmark_functions import benchmark_baseline, benchmark_embds, benchmark_with_kf
from src.comprehensive_test import run_comprehensive_tests
from src.video.video_reader import iter_video_frames


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
        kf_params={'k_mad': 1.0, 'min_spacing': 2},
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
        save_keyframes=True,
        num_workers=1
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
        force_regenerate_keyframes=True,
        force_regenerate_embeddings=False,
        save_keyframes=True,
        keyframe_selectors=['flow', "mog2"],
        # keyframe_selectors=['mog2'],
        keyframe_params={
            'framediff': {'k_mad': 2.5, 'min_spacing': 6},
            'ssim': {'k_mad': 2.5, 'min_spacing': 6},
            'flow': {'k_mad': 2.5, 'min_spacing': 6}
        }
    )


def save_frames_to_disk():
    import os
    vids = "/storage/ice1/8/3/rshah647/VIRATGround/videos_original"
    names = ["VIRAT_S_000001.mp4",  "VIRAT_S_000003.mp4",  "VIRAT_S_000006.mp4", "VIRAT_S_000002.mp4",  "VIRAT_S_000004.mp4",]
    for vid in names:
        vid_path = os.path.join(vids, vid)
        vid_name = vid.split(".")[0]
        print(f"Processing {vid_name}...")
        out_dir = f"data/VIRAT/{vid_name}/"
        for frame, frame_idx in iter_video_frames(vid_path, out_dir, target_fps=1.0):
            # print(frame_idx)
            pass
        

if __name__ == "__main__":
    print("Running full comparison...", time.strftime("%Y-%m-%d %H:%M:%S"))
    run_full_comparison()
    # run_embedding_benchmark()
    print("Full comparison done!", time.strftime("%Y-%m-%d %H:%M:%S"))