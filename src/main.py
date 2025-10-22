import csv
from pathlib import Path

import numpy as np

from src.pipeline.pipeline import VideoPipeline
from src.embeddings.embedder import CLIPEmbedder, CLIP_VIT_B32
from src.models.model_configs import MEDIUM, SMALL, LARGE, LARGE3
from src.pipeline.detection_pipeline import DetectionPipeline
from src.detectors.object_detector import ObjectDetector
from src.training.train_pipeline import finetune_on_virat, pretrain_on_coco
from src.indexing.faiss_index import FAISSIndex
from src.query.semantic_query import SemanticQueryPipeline, simple_query

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

def gen_embeddings_for_dir(videos_dir: str, embedder_config=None):
    """
    Generate embeddings for all videos in a directory.
    """
    videos = sorted(Path(videos_dir).glob("*.mp4"))
    print(f"Found {len(videos)} videos in {videos_dir}")
    
    if embedder_config is None:
        embedder_config = CLIP_VIT_B32
    
    embedder = CLIPEmbedder(embedder_config)
    
    for vid in videos:
        print(f"\n=== Processing {vid.name} ===")
        out_dir = Path("data/VIRAT") / vid.stem / "embeddings"
        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline = VideoPipeline(str(vid), embedder, out_dir=out_dir)
        pipeline.run(save=True)

def build_index(data_dir="data/VIRAT"):
    """Build FAISS index for semantic queries."""
    index = FAISSIndex(data_dir)
    index.build()

def run_semantic_query():
    """
    Run a semantic query using FAISS + MLP pipeline.
    
    Example: Find frames with cars (count >= 2)
    """
    results = simple_query(
        data_dir="data/VIRAT",
        checkpoint_path="models/checkpoints/car_virat_finetuned.pth",
        model_config=MEDIUM,
        text_query="car",
        min_count=2.0,              # Only frames with 2+ cars
        similarity_threshold=0.2     # Cosine similarity threshold
    )
    return results

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

def evaluate_retrieval(data_dir: str, checkpoint_path: str, model_config,
                       target: str = "car", count_threshold: int = 2,
                       similarity_threshold: float = 0.2, num_videos: int = 2):
    """
    Evaluate retrieval quality against ground truth oracle model.
    Retrieves frames with count >= count_threshold for first num_videos.
    """
    data_path = Path(data_dir)
    video_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and not d.name.startswith("_")])
    
    if len(video_dirs) < num_videos:
        raise ValueError(f"Only {len(video_dirs)} videos available, need {num_videos}")
    
    eval_videos = [v.name for v in video_dirs[:num_videos]]
    
    print(f"\n{'='*70}")
    print(f"RETRIEVAL EVALUATION: {target} (count >= {count_threshold})")
    print(f"{'='*70}")
    print(f"Videos: {', '.join(eval_videos)}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Count threshold: {count_threshold}")
    
    pipeline = SemanticQueryPipeline(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        threshold=similarity_threshold
    )
    
    print(f"\n{'='*70}")
    print(f"BEFORE COUNT FILTERING - FAISS + Predictions")
    print(f"{'='*70}")
    
    unfiltered_results = pipeline.query(
        text_query=target,
        count_predicate=None
    )
    
    unfiltered_eval = {k: v for k, v in unfiltered_results.items() if k in eval_videos}
    print(f"\nFrames from eval videos BEFORE count filter:")
    for vid in eval_videos:
        if vid in unfiltered_eval:
            frames = unfiltered_eval[vid]['frames']
            print(f"  {vid}: {len(frames)} frames, avg_count={unfiltered_eval[vid]['avg_count']:.2f}")
            if frames:
                counts_list = [f['predicted_count'] for f in frames]
                print(f"    Count range: [{min(counts_list):.2f}, {max(counts_list):.2f}]")
                passing = sum(1 for c in counts_list if c >= count_threshold)
                print(f"    Frames with count >= {count_threshold}: {passing}/{len(frames)}")
                
                gt_counts = load_ground_truth(data_dir, vid, target)
                gt_positive_frames = [fid for fid, count in gt_counts.items() if count >= count_threshold]
                if gt_positive_frames:
                    print(f"    GT frames with count >= {count_threshold}: {gt_positive_frames}")
                    frame_idx_to_pred = {f['frame_idx']: f['predicted_count'] for f in frames}
                    print(f"    Model predictions for GT positive frames:")
                    for fid in gt_positive_frames:
                        if fid in frame_idx_to_pred:
                            print(f"      Frame {fid}: GT={gt_counts[fid]}, Pred={frame_idx_to_pred[fid]:.2f}")
                        else:
                            print(f"      Frame {fid}: GT={gt_counts[fid]}, Pred=NOT_RETRIEVED (filtered by FAISS)")
        else:
            print(f"  {vid}: 0 frames")
    
    print(f"\n{'='*70}")
    print(f"AFTER COUNT FILTERING (count >= {count_threshold})")
    print(f"{'='*70}")
    
    all_results = pipeline.query(
        text_query=target,
        count_predicate=lambda c: c >= count_threshold
    )
    
    results = {k: v for k, v in all_results.items() if k in eval_videos}
    
    print(f"\n{'='*70}")
    print(f"GROUND TRUTH COMPARISON")
    print(f"{'='*70}")
    print(f"Filtering results to evaluation videos only...")
    print(f"Results in eval videos: {list(results.keys())}")
    
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for video_name in eval_videos:
        gt_counts = load_ground_truth(data_dir, video_name, target)
        total_frames = len(gt_counts)
        
        gt_positive = {fid for fid, count in gt_counts.items() if count >= count_threshold}
        
        if video_name in results:
            retrieved_frames = {f['frame_idx'] for f in results[video_name]['frames']}
        else:
            retrieved_frames = set()
        
        tp = len(gt_positive & retrieved_frames)
        fp = len(retrieved_frames - gt_positive)
        fn = len(gt_positive - retrieved_frames)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\n{video_name}")
        print(f"  Total frames: {total_frames}")
        print(f"  GT positives (count >= {count_threshold}): {len(gt_positive)}")
        print(f"  Retrieved: {len(retrieved_frames)}")
        print(f"  TP={tp}, FP={fp}, FN={fn}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1: {f1:.3f}")
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
        if (overall_precision + overall_recall) > 0 else 0.0
    
    print(f"\n{'='*70}")
    print(f"OVERALL METRICS")
    print(f"{'='*70}")
    print(f"TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"Precision: {overall_precision:.3f}")
    print(f"Recall: {overall_recall:.3f}")
    print(f"F1: {overall_f1:.3f}")

    return {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


if __name__ == "__main__":
    # ========================================
    # STEP 1: Generate embeddings for videos
    # ========================================
    # gen_embeddings()  # Single video
    # gen_embeddings_for_dir("/path/to/videos", CLIP_VIT_B32)  # Batch
    
    # ========================================
    # STEP 2: Generate ground truth counts (optional, for training)
    # ========================================
    # run_detection()  # Single video
    # run_detection_on_dir("/path/to/videos", "yolo11x.pt", False)  # Batch
    
    # ========================================
    # STEP 3: Train count predictor
    # ========================================
    pretrain_on_coco(
        coco_dir="data/coco",
        target="car",
        model_config=LARGE3
    )
    finetune_on_virat(
        data_dir="data/VIRAT",
        target="car",
        pretrained_checkpoint="models/checkpoints/car_coco_pretrained.pth",
        train_ratio=0.5,
        model_config=LARGE3
    )
    
    # ========================================
    # STEP 4: Build FAISS index
    # ========================================
    # build_index("data/VIRAT")
    
    # ========================================
    # STEP 5: Run semantic queries
    # ========================================
    # Simple query
    # run_semantic_query()
    
    # Advanced query with custom predicates
    # run_advanced_query()
    
    # ========================================
    # STEP 6: Evaluate retrieval (first 2 videos)
    # ========================================
    evaluate_retrieval(
        data_dir="data/VIRAT",
        checkpoint_path="models/checkpoints/car_virat_finetuned.pth",
        model_config=LARGE3,
        target="car",
        count_threshold=1,
        similarity_threshold=0.2,
        num_videos=2
    )



