"""
Comprehensive testing script for ablation study.
Tests first 5 videos with different car count thresholds (>= 1, 2, 3, 4, 5).
Compares FAISS+MLP pipeline vs YOLO baseline.
"""

import json
import time
from pathlib import Path
from tabulate import tabulate

from src.models.model_configs import LARGE3
from src.query.semantic_query import SemanticQueryPipeline
from src.baseline import evaluate_baseline_yolo


def run_comprehensive_tests(
    data_dir: str = "data/VIRAT",
    checkpoint_path: str = "models/checkpoints/car_virat_finetuned.pth",
    model_config=LARGE3,
    target: str = "car",
    similarity_threshold: float = 0.2,
    num_videos: int = 5,
    thresholds: list[int] = [1, 2, 3, 4, 5],
    yolo_model: str = "yolo11m",
    output_file: str = "results/comprehensive_test_results.json"
):
    """
    Run comprehensive tests on first N videos with different count thresholds.
    
    Args:
        data_dir: Directory containing video data
        checkpoint_path: Path to trained model checkpoint
        model_config: Model configuration
        target: Object class to detect ("car" or "person")
        similarity_threshold: FAISS similarity threshold
        num_videos: Number of videos to test on
        thresholds: List of count thresholds to test
        yolo_model: YOLO model to use for baseline (yolo11n/s/m/l/x)
        output_file: Path to save results JSON
    """
    data_path = Path(data_dir)
    video_dirs = sorted([d for d in data_path.iterdir() 
                        if d.is_dir() and not d.name.startswith("_")])
    
    if len(video_dirs) < num_videos:
        print(f"Warning: Only {len(video_dirs)} videos available, need {num_videos}")
        num_videos = len(video_dirs)
    
    eval_videos = [v.name for v in video_dirs[:num_videos]]
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ABLATION STUDY")
    print(f"{'='*80}")
    print(f"Videos: {', '.join(eval_videos)}")
    print(f"Target: {target}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Count thresholds: {thresholds}")
    print(f"{'='*80}\n")
    
    # Initialize pipeline once
    print("Initializing FAISS+MLP pipeline...")
    pipeline = SemanticQueryPipeline(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        threshold=similarity_threshold
    )
    
    # Store all results
    all_results = {
        'config': {
            'data_dir': data_dir,
            'checkpoint_path': checkpoint_path,
            'target': target,
            'similarity_threshold': similarity_threshold,
            'num_videos': num_videos,
            'eval_videos': eval_videos,
            'thresholds': thresholds,
            'yolo_model': yolo_model
        },
        'pipeline_results': {},
        'baseline_results': {}
    }
    
    # Test FAISS+MLP pipeline for each threshold
    # Note: Each threshold is queried independently to get accurate end-to-end latency
    print("\n" + "="*80)
    print("TESTING FAISS+MLP PIPELINE")
    print("="*80)
    
    for threshold in thresholds:
        print(f"\n{'─'*80}")
        print(f"Testing with count threshold >= {threshold}")
        print(f"{'─'*80}")
        
        # Measure end-to-end latency for this specific threshold
        metrics = pipeline.query_with_metrics(
            text_query=target,
            count_threshold=threshold,
            eval_videos=eval_videos,
            data_dir=data_dir
        )
        
        all_results['pipeline_results'][threshold] = metrics
        
        # Print results
        print(f"\n[LATENCY METRICS - End-to-End for threshold >= {threshold}]")
        print(f"  FAISS lookup:       {metrics['faiss_latency_ms']:.2f} ms")
        print(f"  MLP prediction:     {metrics['mlp_latency_ms']:.2f} ms")
        print(f"  Total pipeline:     {metrics['total_latency_ms']:.2f} ms")
        
        print(f"\n[FAISS STAGE]")
        print(f"  Precision: {metrics['faiss_precision']:.3f}")
        print(f"  Recall:    {metrics['faiss_recall']:.3f}")
        print(f"  F1:        {metrics['faiss_f1']:.3f}")
        print(f"  Retrieved: {metrics['faiss_retrieved']} frames")
        
        print(f"\n[MLP STAGE (Final)]")
        print(f"  Precision: {metrics['mlp_precision']:.3f}")
        print(f"  Recall:    {metrics['mlp_recall']:.3f}")
        print(f"  F1:        {metrics['mlp_f1']:.3f}")
        print(f"  Retrieved: {metrics['mlp_retrieved']} frames")
    
    # Test YOLO baseline for each threshold
    # Note: Each threshold is evaluated independently to get accurate end-to-end latency
    print("\n" + "="*80)
    print("TESTING YOLO BASELINE")
    print("="*80)
    
    for threshold in thresholds:
        print(f"\n{'─'*80}")
        print(f"Testing with count threshold >= {threshold}")
        print(f"{'─'*80}")
        
        # Measure end-to-end latency for baseline at this specific threshold
        baseline_metrics = evaluate_baseline_yolo(
            data_dir=data_dir,
            model_name=yolo_model,
            target=target,
            count_threshold=threshold,
            num_videos=num_videos
        )
        
        all_results['baseline_results'][threshold] = baseline_metrics
    
    # Generate comparison tables
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    # Latency comparison (End-to-End measurements for all thresholds)
    print(f"\n{'─'*80}")
    print("END-TO-END LATENCY COMPARISON (milliseconds)")
    print(f"{'─'*80}")
    latency_table = []
    for threshold in thresholds:
        pipeline_metrics = all_results['pipeline_results'][threshold]
        baseline_metrics = all_results['baseline_results'][threshold]
        
        latency_table.append([
            f">= {threshold}",
            f"{pipeline_metrics['faiss_latency_ms']:.2f}",
            f"{pipeline_metrics['mlp_latency_ms']:.2f}",
            f"{pipeline_metrics['total_latency_ms']:.2f}",
            f"{baseline_metrics['avg_latency_ms']:.2f}",
            f"{baseline_metrics['avg_latency_ms'] / pipeline_metrics['total_latency_ms']:.2f}x"
        ])
    
    print(tabulate(
        latency_table,
        headers=['Threshold', 'FAISS (ms)', 'MLP (ms)', 'Pipeline Total', 'YOLO Baseline', 'Speedup'],
        tablefmt='grid'
    ))
    
    # FAISS Stage vs Baseline comparison
    print(f"\n{'─'*80}")
    print("FAISS STAGE vs YOLO BASELINE")
    print(f"{'─'*80}")
    faiss_vs_baseline_table = []
    for threshold in thresholds:
        pipeline_metrics = all_results['pipeline_results'][threshold]
        baseline_metrics = all_results['baseline_results'][threshold]
        
        faiss_vs_baseline_table.append([
            f">= {threshold}",
            f"{pipeline_metrics['faiss_precision']:.3f}",
            f"{pipeline_metrics['faiss_recall']:.3f}",
            f"{pipeline_metrics['faiss_f1']:.3f}",
            f"{baseline_metrics['precision']:.3f}",
            f"{baseline_metrics['recall']:.3f}",
            f"{baseline_metrics['f1']:.3f}",
        ])
    
    print(tabulate(
        faiss_vs_baseline_table,
        headers=['Threshold', 'FAISS P', 'FAISS R', 'FAISS F1', 'YOLO P', 'YOLO R', 'YOLO F1'],
        tablefmt='grid'
    ))
    
    # MLP (Final Pipeline) vs Baseline comparison
    print(f"\n{'─'*80}")
    print("FINAL PIPELINE (FAISS+MLP) vs YOLO BASELINE")
    print(f"{'─'*80}")
    mlp_vs_baseline_table = []
    for threshold in thresholds:
        pipeline_metrics = all_results['pipeline_results'][threshold]
        baseline_metrics = all_results['baseline_results'][threshold]
        
        mlp_vs_baseline_table.append([
            f">= {threshold}",
            f"{pipeline_metrics['mlp_precision']:.3f}",
            f"{pipeline_metrics['mlp_recall']:.3f}",
            f"{pipeline_metrics['mlp_f1']:.3f}",
            f"{baseline_metrics['precision']:.3f}",
            f"{baseline_metrics['recall']:.3f}",
            f"{baseline_metrics['f1']:.3f}",
        ])
    
    print(tabulate(
        mlp_vs_baseline_table,
        headers=['Threshold', 'Pipeline P', 'Pipeline R', 'Pipeline F1', 'YOLO P', 'YOLO R', 'YOLO F1'],
        tablefmt='grid'
    ))
    
    # Retrieved frames comparison
    print(f"\n{'─'*80}")
    print("RETRIEVED FRAMES COMPARISON")
    print(f"{'─'*80}")
    retrieved_table = []
    for threshold in thresholds:
        pipeline_metrics = all_results['pipeline_results'][threshold]
        baseline_metrics = all_results['baseline_results'][threshold]
        
        retrieved_table.append([
            f">= {threshold}",
            pipeline_metrics['faiss_retrieved'],
            pipeline_metrics['mlp_retrieved'],
            baseline_metrics['frames_processed']
        ])
    
    print(tabulate(
        retrieved_table,
        headers=['Threshold', 'FAISS Retrieved', 'Final Pipeline', 'YOLO (all frames)'],
        tablefmt='grid'
    ))
    
    # Save results to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    return all_results


if __name__ == "__main__":
    results = run_comprehensive_tests(
        data_dir="data/VIRAT",
        checkpoint_path="models/checkpoints/car_virat_finetuned.pth",
        model_config=LARGE3,
        target="car",
        similarity_threshold=0.2,
        num_videos=5,
        thresholds=[1, 2, 3, 4, 5],
        yolo_model="yolo11m",  # Medium model for fair comparison
        output_file="results/comprehensive_test_results.json"
    )

