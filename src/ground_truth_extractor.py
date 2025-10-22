"""
Ground truth extractor for comprehensive test results.
Reads the JSON generated from comprehensive_test.py and adds ground truth numbers.
Outputs updated results with ground truth comparisons in tabular format.
"""

import json
import csv
from pathlib import Path
from tabulate import tabulate


def load_ground_truth_for_video(data_dir: str, video_name: str, target: str = "car"):
    """Load ground truth counts from counts.csv for a single video."""
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


def calculate_ground_truth_metrics(data_dir: str, eval_videos: list, target: str, 
                                   count_threshold: int):
    """
    Calculate ground truth statistics for a given threshold.
    
    Returns:
        Dictionary with ground truth metrics
    """
    total_frames = 0
    total_positive_frames = 0
    total_object_count = 0
    
    for video_name in eval_videos:
        gt_counts = load_ground_truth_for_video(data_dir, video_name, target)
        
        for frame_id, count in gt_counts.items():
            total_frames += 1
            total_object_count += count
            if count >= count_threshold:
                total_positive_frames += 1
    
    return {
        'total_frames': total_frames,
        'positive_frames': total_positive_frames,
        'negative_frames': total_frames - total_positive_frames,
        'total_object_count': total_object_count,
        'avg_objects_per_frame': total_object_count / total_frames if total_frames > 0 else 0,
        'positive_ratio': total_positive_frames / total_frames if total_frames > 0 else 0
    }


def add_ground_truth_to_results(
    results_file: str,
    output_file: str = "results/updated_comprehensive_results.json"
):
    """
    Read comprehensive test results JSON and add ground truth statistics.
    
    Args:
        results_file: Path to comprehensive_test_results.json
        output_file: Path to save updated results with ground truth
    """
    # Load existing results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    config = results['config']
    data_dir = config['data_dir']
    eval_videos = config['eval_videos']
    target = config['target']
    thresholds = config['thresholds']
    
    print(f"\n{'='*80}")
    print(f"GROUND TRUTH EXTRACTION")
    print(f"{'='*80}")
    print(f"Data directory: {data_dir}")
    print(f"Videos: {', '.join(eval_videos)}")
    print(f"Target: {target}")
    print(f"Thresholds: {thresholds}")
    
    # Add ground truth for each threshold
    results['ground_truth'] = {}
    
    for threshold in thresholds:
        print(f"\nProcessing threshold >= {threshold}...")
        gt_metrics = calculate_ground_truth_metrics(
            data_dir, eval_videos, target, threshold
        )
        results['ground_truth'][threshold] = gt_metrics
        
        print(f"  Total frames: {gt_metrics['total_frames']}")
        print(f"  Positive frames: {gt_metrics['positive_frames']}")
        print(f"  Negative frames: {gt_metrics['negative_frames']}")
        print(f"  Positive ratio: {gt_metrics['positive_ratio']:.3f}")
    
    # Save updated results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Updated results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    return results


def generate_comparison_tables(results_file: str, 
                               output_dir: str = "results/updated_result"):
    """
    Generate comprehensive comparison tables with ground truth data.
    
    Args:
        results_file: Path to JSON file with ground truth added
        output_dir: Directory to save output tables
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    config = results['config']
    thresholds = config['thresholds']
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE COMPARISON WITH GROUND TRUTH")
    print(f"{'='*80}\n")
    
    # Table 1: Ground Truth Statistics
    print(f"{'─'*80}")
    print("TABLE 1: GROUND TRUTH STATISTICS")
    print(f"{'─'*80}")
    gt_table = []
    for threshold in thresholds:
        gt = results['ground_truth'][threshold]
        gt_table.append([
            f">= {threshold}",
            gt['total_frames'],
            gt['positive_frames'],
            gt['negative_frames'],
            f"{gt['positive_ratio']:.3f}",
            f"{gt['avg_objects_per_frame']:.2f}"
        ])
    
    gt_table_str = tabulate(
        gt_table,
        headers=['Threshold', 'Total Frames', 'Positive', 'Negative', 'Pos Ratio', 'Avg Objects/Frame'],
        tablefmt='grid'
    )
    print(gt_table_str)
    
    with open(output_path / "01_ground_truth_stats.txt", 'w') as f:
        f.write("GROUND TRUTH STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(gt_table_str)
    
    # Table 2: Latency Comparison
    print(f"\n{'─'*80}")
    print("TABLE 2: LATENCY COMPARISON (milliseconds)")
    print(f"{'─'*80}")
    latency_table = []
    for threshold in thresholds:
        pipeline_metrics = results['pipeline_results'][str(threshold)]
        baseline_metrics = results['baseline_results'][str(threshold)]
        
        latency_table.append([
            f">= {threshold}",
            f"{pipeline_metrics['faiss_latency_ms']:.2f}",
            f"{pipeline_metrics['mlp_latency_ms']:.2f}",
            f"{pipeline_metrics['total_latency_ms']:.2f}",
            f"{baseline_metrics['avg_latency_ms']:.2f}",
            f"{baseline_metrics['avg_latency_ms'] / pipeline_metrics['total_latency_ms']:.2f}x"
        ])
    
    latency_table_str = tabulate(
        latency_table,
        headers=['Threshold', 'FAISS (ms)', 'MLP (ms)', 'Pipeline Total', 'YOLO Baseline', 'Speedup'],
        tablefmt='grid'
    )
    print(latency_table_str)
    
    with open(output_path / "02_latency_comparison.txt", 'w') as f:
        f.write("LATENCY COMPARISON\n")
        f.write("="*80 + "\n\n")
        f.write(latency_table_str)
    
    # Table 3: Accuracy Comparison - Pipeline vs Ground Truth
    print(f"\n{'─'*80}")
    print("TABLE 3: PIPELINE (FAISS+MLP) ACCURACY vs GROUND TRUTH")
    print(f"{'─'*80}")
    pipeline_accuracy_table = []
    for threshold in thresholds:
        pipeline_metrics = results['pipeline_results'][str(threshold)]
        gt = results['ground_truth'][threshold]
        
        pipeline_accuracy_table.append([
            f">= {threshold}",
            f"{pipeline_metrics['mlp_precision']:.3f}",
            f"{pipeline_metrics['mlp_recall']:.3f}",
            f"{pipeline_metrics['mlp_f1']:.3f}",
            pipeline_metrics['mlp_retrieved'],
            gt['positive_frames'],
            f"{pipeline_metrics['mlp_retrieved'] / gt['total_frames']:.3f}"
        ])
    
    pipeline_accuracy_str = tabulate(
        pipeline_accuracy_table,
        headers=['Threshold', 'Precision', 'Recall', 'F1', 'Retrieved', 'GT Positive', 'Retrieval Rate'],
        tablefmt='grid'
    )
    print(pipeline_accuracy_str)
    
    with open(output_path / "03_pipeline_accuracy.txt", 'w') as f:
        f.write("PIPELINE (FAISS+MLP) ACCURACY vs GROUND TRUTH\n")
        f.write("="*80 + "\n\n")
        f.write(pipeline_accuracy_str)
    
    # Table 4: Baseline Accuracy vs Ground Truth
    print(f"\n{'─'*80}")
    print("TABLE 4: YOLO BASELINE ACCURACY vs GROUND TRUTH")
    print(f"{'─'*80}")
    baseline_accuracy_table = []
    for threshold in thresholds:
        baseline_metrics = results['baseline_results'][str(threshold)]
        gt = results['ground_truth'][threshold]
        
        baseline_accuracy_table.append([
            f">= {threshold}",
            f"{baseline_metrics['precision']:.3f}",
            f"{baseline_metrics['recall']:.3f}",
            f"{baseline_metrics['f1']:.3f}",
            baseline_metrics['frames_processed'],
            gt['positive_frames'],
            f"{baseline_metrics['frames_processed'] / gt['total_frames']:.3f}"
        ])
    
    baseline_accuracy_str = tabulate(
        baseline_accuracy_table,
        headers=['Threshold', 'Precision', 'Recall', 'F1', 'Processed', 'GT Positive', 'Processing Rate'],
        tablefmt='grid'
    )
    print(baseline_accuracy_str)
    
    with open(output_path / "04_baseline_accuracy.txt", 'w') as f:
        f.write("YOLO BASELINE ACCURACY vs GROUND TRUTH\n")
        f.write("="*80 + "\n\n")
        f.write(baseline_accuracy_str)
    
    # Table 5: Pipeline vs Baseline Direct Comparison
    print(f"\n{'─'*80}")
    print("TABLE 5: PIPELINE vs BASELINE - DIRECT COMPARISON")
    print(f"{'─'*80}")
    comparison_table = []
    for threshold in thresholds:
        pipeline_metrics = results['pipeline_results'][str(threshold)]
        baseline_metrics = results['baseline_results'][str(threshold)]
        
        comparison_table.append([
            f">= {threshold}",
            f"{pipeline_metrics['mlp_precision']:.3f}",
            f"{baseline_metrics['precision']:.3f}",
            f"{pipeline_metrics['mlp_recall']:.3f}",
            f"{baseline_metrics['recall']:.3f}",
            f"{pipeline_metrics['mlp_f1']:.3f}",
            f"{baseline_metrics['f1']:.3f}",
            f"{pipeline_metrics['total_latency_ms']:.2f}",
            f"{baseline_metrics['avg_latency_ms']:.2f}"
        ])
    
    comparison_str = tabulate(
        comparison_table,
        headers=['Threshold', 'Pipe P', 'YOLO P', 'Pipe R', 'YOLO R', 'Pipe F1', 'YOLO F1', 'Pipe Lat', 'YOLO Lat'],
        tablefmt='grid'
    )
    print(comparison_str)
    
    with open(output_path / "05_pipeline_vs_baseline.txt", 'w') as f:
        f.write("PIPELINE vs BASELINE - DIRECT COMPARISON\n")
        f.write("="*80 + "\n\n")
        f.write(comparison_str)
    
    # Table 6: FAISS Stage Performance
    print(f"\n{'─'*80}")
    print("TABLE 6: FAISS PREFILTER STAGE PERFORMANCE")
    print(f"{'─'*80}")
    faiss_table = []
    for threshold in thresholds:
        pipeline_metrics = results['pipeline_results'][str(threshold)]
        gt = results['ground_truth'][threshold]
        
        faiss_table.append([
            f">= {threshold}",
            f"{pipeline_metrics['faiss_precision']:.3f}",
            f"{pipeline_metrics['faiss_recall']:.3f}",
            f"{pipeline_metrics['faiss_f1']:.3f}",
            pipeline_metrics['faiss_retrieved'],
            gt['positive_frames'],
            f"{pipeline_metrics['faiss_latency_ms']:.2f}"
        ])
    
    faiss_str = tabulate(
        faiss_table,
        headers=['Threshold', 'Precision', 'Recall', 'F1', 'Retrieved', 'GT Positive', 'Latency (ms)'],
        tablefmt='grid'
    )
    print(faiss_str)
    
    with open(output_path / "06_faiss_stage_performance.txt", 'w') as f:
        f.write("FAISS PREFILTER STAGE PERFORMANCE\n")
        f.write("="*80 + "\n\n")
        f.write(faiss_str)
    
    # Save summary
    print(f"\n{'='*80}")
    print(f"All tables saved to: {output_dir}/")
    print(f"{'='*80}\n")
    
    # Create a combined summary file
    with open(output_path / "00_summary.txt", 'w') as f:
        f.write("COMPREHENSIVE TEST RESULTS WITH GROUND TRUTH\n")
        f.write("="*80 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Data Directory: {config['data_dir']}\n")
        f.write(f"  Videos: {', '.join(config['eval_videos'])}\n")
        f.write(f"  Target: {config['target']}\n")
        f.write(f"  Similarity Threshold: {config['similarity_threshold']}\n")
        f.write(f"  YOLO Model: {config['yolo_model']}\n")
        f.write(f"  Thresholds: {config['thresholds']}\n")
        f.write("\n" + "="*80 + "\n\n")
        f.write("Generated Tables:\n")
        f.write("  01_ground_truth_stats.txt\n")
        f.write("  02_latency_comparison.txt\n")
        f.write("  03_pipeline_accuracy.txt\n")
        f.write("  04_baseline_accuracy.txt\n")
        f.write("  05_pipeline_vs_baseline.txt\n")
        f.write("  06_faiss_stage_performance.txt\n")


def run_full_extraction(
    results_file: str = "results/comprehensive_test_results.json",
    output_json: str = "results/updated_comprehensive_results.json",
    output_tables_dir: str = "results/updated_result"
):
    """
    Run the full ground truth extraction and table generation pipeline.
    
    Args:
        results_file: Path to comprehensive test results JSON
        output_json: Path to save updated JSON with ground truth
        output_tables_dir: Directory to save output tables
    """
    print(f"\n{'='*80}")
    print(f"GROUND TRUTH EXTRACTION PIPELINE")
    print(f"{'='*80}\n")
    
    # Step 1: Add ground truth to results
    print("Step 1: Extracting ground truth and updating results...")
    updated_results = add_ground_truth_to_results(results_file, output_json)
    
    # Step 2: Generate comparison tables
    print("\nStep 2: Generating comparison tables...")
    generate_comparison_tables(output_json, output_tables_dir)
    
    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Updated JSON: {output_json}")
    print(f"Tables: {output_tables_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Run the full extraction pipeline
    run_full_extraction(
        results_file="results/comprehensive_test_results.json",
        output_json="results/updated_comprehensive_results.json",
        output_tables_dir="results/updated_result"
    )

