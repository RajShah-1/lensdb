"""
Comprehensive testing script for ablation study with keyframe selection support.
Combines benchmark functions and generates detailed comparison reports.
"""

import json
import numpy as np
from pathlib import Path
from tabulate import tabulate

from src.models.model_configs import LARGE3
from src.benchmark_functions import benchmark_baseline, benchmark_embds, benchmark_with_kf


def compute_averages(results_dict, thresholds, keys):
    """Compute averages across thresholds for specified keys."""
    return {key: np.mean([results_dict[t][key] for t in thresholds]) for key in keys}


def generate_compression_table(keyframe_results, keyframe_selectors):
    """Generate keyframe compression statistics table."""
    table = []
    for selector_name in keyframe_selectors:
        kf_meta = keyframe_results[selector_name]['metadata']
        total_frames = sum(m['total_frames'] for m in kf_meta)
        total_keyframes = sum(m['num_keyframes'] for m in kf_meta)
        avg_compression = sum(m['compression_ratio'] for m in kf_meta) / len(kf_meta)
        
        table.append([
            selector_name,
            total_frames,
            total_keyframes,
            f"{avg_compression:.2f}x",
            f"{(1 - total_keyframes/total_frames) * 100:.1f}%"
        ])
    
    return tabulate(
        table,
        headers=['Selector', 'Total Frames', 'Keyframes', 'Avg Compression', 'Space Saved'],
        tablefmt='grid'
    )


def generate_latency_table(threshold, no_kf_results, keyframe_results, baseline_results, 
                           keyframe_selectors, yolo_model):
    """Generate latency comparison table for a specific threshold."""
    table = []
    
    no_kf = no_kf_results[threshold]
    table.append([
        'No Keyframes',
        f"{no_kf['total_latency_ms']:.2f}",
        f"{no_kf['mlp_recall']:.3f}",
        f"{no_kf['mlp_f1']:.3f}",
        "1.00x"
    ])
    
    for selector_name in keyframe_selectors:
        kf = keyframe_results[selector_name]['pipeline_results'][threshold]
        speedup = no_kf['total_latency_ms'] / kf['total_latency_ms']
        table.append([
            f"w/ {selector_name}",
            f"{kf['total_latency_ms']:.2f}",
            f"{kf['mlp_recall']:.3f}",
            f"{kf['mlp_f1']:.3f}",
            f"{speedup:.2f}x"
        ])
    
    baseline = baseline_results[threshold]
    yolo_speedup = baseline['avg_latency_ms'] / no_kf['total_latency_ms']
    table.append([
        f'YOLO {yolo_model}',
        f"{baseline['avg_latency_ms']:.2f}",
        f"{baseline['recall']:.3f}",
        f"{baseline['f1']:.3f}",
        f"{yolo_speedup:.2f}x"
    ])
    
    return tabulate(
        table,
        headers=['Method', 'Latency (ms)', 'Recall', 'F1', 'Speedup vs No-KF'],
        tablefmt='grid'
    )


def generate_detailed_metrics_table(no_kf_results, keyframe_results, baseline_results,
                                    thresholds, keyframe_selectors, yolo_model):
    """Generate detailed metrics breakdown table (averaged across thresholds)."""
    table = []
    
    no_kf_avg = compute_averages(no_kf_results, thresholds, [
        'total_latency_ms', 'mlp_recall', 'mlp_precision', 'mlp_f1',
        'faiss_latency_ms', 'mlp_latency_ms'
    ])
    
    table.append([
        'No Keyframes',
        f"{no_kf_avg['total_latency_ms']:.2f}",
        f"{no_kf_avg['faiss_latency_ms']:.2f}",
        f"{no_kf_avg['mlp_latency_ms']:.2f}",
        f"{no_kf_avg['mlp_precision']:.3f}",
        f"{no_kf_avg['mlp_recall']:.3f}",
        f"{no_kf_avg['mlp_f1']:.3f}"
    ])
    
    for selector_name in keyframe_selectors:
        kf_results = keyframe_results[selector_name]['pipeline_results']
        kf_avg = compute_averages(kf_results, thresholds, [
            'total_latency_ms', 'mlp_recall', 'mlp_precision', 'mlp_f1',
            'faiss_latency_ms', 'mlp_latency_ms'
        ])
        
        table.append([
            f"w/ {selector_name}",
            f"{kf_avg['total_latency_ms']:.2f}",
            f"{kf_avg['faiss_latency_ms']:.2f}",
            f"{kf_avg['mlp_latency_ms']:.2f}",
            f"{kf_avg['mlp_precision']:.3f}",
            f"{kf_avg['mlp_recall']:.3f}",
            f"{kf_avg['mlp_f1']:.3f}"
        ])
    
    yolo_avg = compute_averages(baseline_results, thresholds, [
        'avg_latency_ms', 'recall', 'precision', 'f1'
    ])
    
    table.append([
        f"YOLO {yolo_model}",
        f"{yolo_avg['avg_latency_ms']:.2f}",
        "N/A",
        "N/A",
        f"{yolo_avg['precision']:.3f}",
        f"{yolo_avg['recall']:.3f}",
        f"{yolo_avg['f1']:.3f}"
    ])
    
    return tabulate(
        table,
        headers=['Method', 'Total (ms)', 'FAISS (ms)', 'MLP (ms)', 'Precision', 'Recall', 'F1'],
        tablefmt='grid'
    )


def generate_recall_retention_table(no_kf_results, kf_results, thresholds):
    """Generate recall retention analysis table for a specific keyframe selector."""
    table = []
    for threshold in thresholds:
        no_kf_recall = no_kf_results[threshold]['mlp_recall']
        kf_recall = kf_results[threshold]['mlp_recall']
        retention_pct = (kf_recall / no_kf_recall * 100) if no_kf_recall > 0 else 0
        recall_diff = kf_recall - no_kf_recall
        
        table.append([
            f">= {threshold}",
            f"{no_kf_recall:.3f}",
            f"{kf_recall:.3f}",
            f"{recall_diff:+.3f}",
            f"{retention_pct:.1f}%"
        ])
    
    return tabulate(
        table,
        headers=['Threshold', 'No KF Recall', 'KF Recall', 'Difference', 'Retention'],
        tablefmt='grid'
    )


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def run_comprehensive_tests(
    data_dir: str = "data/VIRAT",
    checkpoint_path: str = "models/checkpoints/car_virat_finetuned.pth",
    model_config=LARGE3,
    target: str = "car",
    similarity_threshold: float = 0.2,
    num_videos: int = 5,
    thresholds: list[int] = [0, 1, 2, 3, 4, 5],
    yolo_model: str = "yolo11m",
    output_file: str = "results/comprehensive_test_results.json",
    test_keyframes: bool = True,
    keyframe_selectors: list[str] = None,
    keyframe_params: dict = None,
    force_regenerate_keyframes: bool = False,
    videos_source_dir: str = None
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
        yolo_model: YOLO model to use for baseline
        output_file: Path to save results JSON
        test_keyframes: Whether to test with keyframe selection
        keyframe_selectors: List of keyframe selectors to test
        keyframe_params: Parameters for keyframe selectors
        force_regenerate_keyframes: If True, regenerate keyframe embeddings
        videos_source_dir: Directory containing source .mp4 video files
    
    Returns:
        Dictionary containing all test results
    """
    if test_keyframes and keyframe_selectors is None:
        keyframe_selectors = ['framediff']
    elif not test_keyframes:
        keyframe_selectors = []
    
    keyframe_params = keyframe_params or {}
    
    data_path = Path(data_dir)
    video_dirs = sorted([d for d in data_path.iterdir() 
                        if d.is_dir() and not d.name.startswith("_")])
    
    if len(video_dirs) < num_videos:
        print(f"Warning: Only {len(video_dirs)} videos available, need {num_videos}")
        num_videos = len(video_dirs)
    
    eval_videos = [v.name for v in video_dirs[:num_videos]]
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ABLATION STUDY WITH KEYFRAME SELECTION")
    print(f"{'='*80}")
    print(f"Videos: {', '.join(eval_videos)}")
    print(f"Target: {target}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Count thresholds: {thresholds}")
    print(f"Test keyframes: {test_keyframes}")
    if test_keyframes:
        print(f"Keyframe selectors: {', '.join(keyframe_selectors)}")
    print(f"{'='*80}\n")
    
    all_results = {
        'config': {
            'data_dir': data_dir,
            'checkpoint_path': checkpoint_path,
            'target': target,
            'similarity_threshold': similarity_threshold,
            'num_videos': num_videos,
            'eval_videos': eval_videos,
            'thresholds': thresholds,
            'yolo_model': yolo_model,
            'test_keyframes': test_keyframes,
            'keyframe_selectors': keyframe_selectors
        },
        'no_keyframes': {'pipeline_results': {}},
        'keyframe_results': {},
        'baseline_results': {}
    }
    
    print("\n" + "="*80)
    print("SECTION 1: TESTING WITHOUT KEYFRAMES (Standard Pipeline)")
    print("="*80)
    
    all_results['no_keyframes']['pipeline_results'] = benchmark_embds(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        target=target,
        similarity_threshold=similarity_threshold,
        num_videos=num_videos,
        thresholds=thresholds
    )
    
    if test_keyframes:
        for selector_name in keyframe_selectors:
            print(f"\n{'='*80}")
            print(f"SECTION 2: TESTING WITH KEYFRAMES ({selector_name.upper()})")
            print(f"{'='*80}")
            
            all_results['keyframe_results'][selector_name] = benchmark_with_kf(
                kf_method=selector_name,
                data_dir=data_dir,
                checkpoint_path=checkpoint_path,
                model_config=model_config,
                target=target,
                similarity_threshold=similarity_threshold,
                num_videos=num_videos,
                thresholds=thresholds,
                kf_params=keyframe_params.get(selector_name, {}),
                force_regenerate=force_regenerate_keyframes,
                videos_source_dir=videos_source_dir
            )
    
    print("\n" + "="*80)
    print("SECTION 3: TESTING YOLO BASELINE")
    print("="*80)
    
    all_results['baseline_results'] = benchmark_baseline(
        data_dir=data_dir,
        target=target,
        num_videos=num_videos,
        thresholds=thresholds,
        yolo_model=yolo_model
    )
    
    print("\n" + "="*80)
    print("SECTION 4: COMPARISON SUMMARY")
    print("="*80)
    
    if test_keyframes:
        print(f"\n{'─'*80}")
        print("TABLE 1: KEYFRAME COMPRESSION STATISTICS")
        print(f"{'─'*80}")
        print(generate_compression_table(
            all_results['keyframe_results'], 
            keyframe_selectors
        ))
    
    print(f"\n{'─'*80}")
    print("TABLE 2: END-TO-END LATENCY COMPARISON (milliseconds)")
    print(f"{'─'*80}")
    
    for threshold in thresholds:
        print(f"\nCount Threshold >= {threshold}:")
        print(generate_latency_table(
            threshold,
            all_results['no_keyframes']['pipeline_results'],
            all_results['keyframe_results'],
            all_results['baseline_results'],
            keyframe_selectors if test_keyframes else [],
            yolo_model
        ))
    
    print(f"\n{'─'*80}")
    print("TABLE 3: DETAILED METRICS BREAKDOWN (Averaged)")
    print(f"{'─'*80}")
    print(generate_detailed_metrics_table(
        all_results['no_keyframes']['pipeline_results'],
        all_results['keyframe_results'],
        all_results['baseline_results'],
        thresholds,
        keyframe_selectors if test_keyframes else [],
        yolo_model
    ))
    
    if test_keyframes:
        print("\n" + "="*80)
        print("TABLE 4: RECALL RETENTION WITH KEYFRAMES")
        print("="*80)
        
        for selector_name in keyframe_selectors:
            print(f"\nSelector: {selector_name}")
            print(generate_recall_retention_table(
                all_results['no_keyframes']['pipeline_results'],
                all_results['keyframe_results'][selector_name]['pipeline_results'],
                thresholds
            ))
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"COMPLETE!")
    print(f"{'='*80}")
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
        thresholds=[0, 1, 2, 3, 4, 5],
        yolo_model="yolo11m",
        output_file="results/comprehensive_test_keyframes.json",
        test_keyframes=True,
        keyframe_selectors=['framediff'],
        keyframe_params={'framediff': {'k_mad': 2.5, 'min_spacing': 6}},
        force_regenerate_keyframes=False,
        videos_source_dir="/storage/ice1/8/3/rshah647/VIRATGround/videos_original"
    )
