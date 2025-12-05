"""
Baseline YOLO model for ablation study comparison.
Runs a small YOLO model and measures latency and accuracy.
Decodes frames directly from video at 1 fps for realistic timing.
"""

import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# COCO dataset class IDs
COCO_CLASS_IDS = {
    "person": 0,
    "car": 2,
}


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


def evaluate_baseline_yolo(data_dir: str = "data/VIRAT",
                           videos_source_dir: str = "/storage/ice1/8/3/rshah647/VIRATGround/videos_original",
                           model_name: str = "yolo11n",
                           target: str = "car", count_threshold: int = 2,
                           num_videos: int = 2, target_fps: float = 1.0):
    """
    Evaluate baseline YOLO11 model on VIRAT videos.
    
    Reads frames directly from video files at target_fps for realistic decode timing.
    
    Args:
        data_dir: Directory containing video data with counts.csv
        videos_source_dir: Directory containing original .mp4 video files
        model_name: YOLO model size (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
        target: Object class to detect ("car" or "person")
        count_threshold: Minimum count threshold for positive classification
        num_videos: Number of videos to evaluate
        target_fps: Target frames per second to sample (default 1.0)
    
    Returns:
        Dictionary with latency and accuracy metrics
    """
    data_path = Path(data_dir)
    videos_path = Path(videos_source_dir)
    
    video_dirs = sorted([d for d in data_path.iterdir() 
                        if d.is_dir() and not d.name.startswith("_")])
    
    if len(video_dirs) < num_videos:
        raise ValueError(f"Only {len(video_dirs)} videos available, need {num_videos}")
    
    eval_videos = [v.name for v in video_dirs[:num_videos]]
    
    print(f"\n{'='*70}")
    print(f"BASELINE YOLO11 EVALUATION: {model_name}")
    print(f"{'='*70}")
    print(f"Videos: {', '.join(eval_videos)}")
    print(f"Target: {target}")
    print(f"Count threshold: {count_threshold}")
    print(f"Reading from video at {target_fps} fps")
    
    # Load YOLO model
    print(f"\nLoading YOLO model: {model_name}")
    model = YOLO(f"{model_name}.pt")
    
    # Get target class ID
    if target not in COCO_CLASS_IDS:
        raise ValueError(f"Unknown target '{target}'. Valid options: {list(COCO_CLASS_IDS.keys())}")
    target_class_id = COCO_CLASS_IDS[target]
    
    total_tp, total_fp, total_fn = 0, 0, 0
    total_decode_latency_ms = 0
    total_inference_latency_ms = 0
    total_frames_processed = 0
    
    for video_name in eval_videos:
        print(f"\nProcessing {video_name}...")
        
        # Load ground truth
        gt_counts = load_ground_truth(data_dir, video_name, target)
        
        # Find video file
        video_file = videos_path / f"{video_name}.mp4"
        if not video_file.exists():
            print(f"  Warning: video file not found at {video_file}, skipping")
            continue
        
        # Open video
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"  Warning: could not open video {video_file}, skipping")
            continue
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(video_fps / target_fps)  # Sample every N frames
        
        print(f"  Video: {video_fps:.1f} fps, {total_frames} frames, sampling every {frame_interval} frames")
        
        predictions = {}
        video_decode_latency_ms = 0
        video_inference_latency_ms = 0
        
        frame_idx = 0
        video_frame_num = 0  # Actual frame number in video
        
        while video_frame_num < total_frames:
            if frame_idx not in gt_counts:
                frame_idx += 1
                video_frame_num += frame_interval
                continue
            
            decode_start = time.perf_counter()
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_num)
            ret, frame = cap.read()
            decode_latency_ms = (time.perf_counter() - decode_start) * 1000
            
            if not ret:
                frame_idx += 1
                video_frame_num += frame_interval
                continue
            
            video_decode_latency_ms += decode_latency_ms
            
            # Inference: Run YOLO detection
            inference_start = time.perf_counter()
            results = model(frame, verbose=False)
            inference_latency_ms = (time.perf_counter() - inference_start) * 1000
            video_inference_latency_ms += inference_latency_ms
            
            total_frames_processed += 1
            
            # Count detections of target class
            count = 0
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    if int(box.cls[0]) == target_class_id:
                        count += 1
            
            predictions[frame_idx] = count
            
            frame_idx += 1
            video_frame_num += frame_interval
        
        cap.release()
        
        # Calculate metrics for this video
        gt_positive = {fid for fid, count in gt_counts.items() 
                      if count >= count_threshold}
        pred_positive = {fid for fid, count in predictions.items() 
                        if count >= count_threshold}
        
        tp = len(gt_positive & pred_positive)
        fp = len(pred_positive - gt_positive)
        fn = len(gt_positive - pred_positive)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_decode_latency_ms += video_decode_latency_ms
        total_inference_latency_ms += video_inference_latency_ms
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        num_predictions = len(predictions) if predictions else 1
        video_latency_ms = video_decode_latency_ms + video_inference_latency_ms
        avg_latency = video_latency_ms / num_predictions
        
        print(f"  Processed {len(predictions)} frames")
        print(f"  Avg latency per frame: {avg_latency:.2f} ms "
              f"(decode: {video_decode_latency_ms/num_predictions:.2f}, "
              f"inference: {video_inference_latency_ms/num_predictions:.2f})")
        print(f"  TP={tp}, FP={fp}, FN={fn}")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
        if (overall_precision + overall_recall) > 0 else 0.0
    
    total_latency_ms = total_decode_latency_ms + total_inference_latency_ms
    avg_latency = total_latency_ms / total_frames_processed if total_frames_processed > 0 else 0
    avg_decode_latency = total_decode_latency_ms / total_frames_processed if total_frames_processed > 0 else 0
    avg_inference_latency = total_inference_latency_ms / total_frames_processed if total_frames_processed > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"BASELINE YOLO11 RESULTS")
    print(f"{'='*70}")
    
    print(f"\n[LATENCY METRICS]")
    print(f"  Avg latency per frame: {avg_latency:.2f} ms")
    print(f"    - Decode (video):    {avg_decode_latency:.2f} ms")
    print(f"    - Inference:         {avg_inference_latency:.2f} ms")
    print(f"  Total frames processed: {total_frames_processed}")
    print(f"  Total latency: {total_latency_ms:.2f} ms")
    
    print(f"\n[ACCURACY METRICS]")
    print(f"  Precision: {overall_precision:.3f}")
    print(f"  Recall:    {overall_recall:.3f}")
    print(f"  F1:        {overall_f1:.3f}")
    print(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")
    
    print(f"\n{'='*70}")
    
    return {
        'model': model_name,
        'avg_latency_ms': avg_latency,
        'avg_decode_latency_ms': avg_decode_latency,
        'avg_inference_latency_ms': avg_inference_latency,
        'total_latency_ms': total_latency_ms,
        'total_decode_latency_ms': total_decode_latency_ms,
        'total_inference_latency_ms': total_inference_latency_ms,
        'frames_processed': total_frames_processed,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


if __name__ == "__main__":
    # Run baseline evaluation with small YOLO11 model
    evaluate_baseline_yolo(
        data_dir="data/VIRAT",
        videos_source_dir="/storage/ice1/8/3/rshah647/VIRATGround/videos_original",
        model_name="yolo11n",
        target="car",
        count_threshold=1,
        num_videos=2,
        target_fps=1.0
    )
