"""Comprehensive testing script for ablation study."""

import json
import numpy as np
from pathlib import Path
from tabulate import tabulate

from src.indexing.faiss_index import FAISSIndex
from src.models.model_configs import LARGE3
from src.embeddings.embedder import CLIPEmbedder, CLIP_VIT_B32
from src.benchmark_functions import benchmark_baseline, benchmark_embds, benchmark_with_kf


def compute_averages(results_dict, thresholds, keys):
    """Compute averages across thresholds."""
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
    """Generate latency comparison table."""
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
    """Generate detailed metrics breakdown table."""
    table = []
    
    no_kf_avg = compute_averages(no_kf_results, thresholds, [
        'total_latency_ms', 'mlp_recall', 'mlp_precision', 'mlp_f1',
        'faiss_latency_ms', 'decode_latency_ms', 'inference_latency_ms'
    ])
    
    table.append([
        'No Keyframes',
        f"{no_kf_avg['total_latency_ms']:.2f}",
        f"{no_kf_avg['faiss_latency_ms']:.2f}",
        f"{no_kf_avg['decode_latency_ms']:.2f}",
        f"{no_kf_avg['inference_latency_ms']:.2f}",
        f"{no_kf_avg['mlp_recall']:.3f}",
        f"{no_kf_avg['mlp_precision']:.3f}",
        f"{no_kf_avg['mlp_f1']:.3f}"
    ])
    
    for selector_name in keyframe_selectors:
        kf_avg = compute_averages(keyframe_results[selector_name]['pipeline_results'], 
                                  thresholds, [
            'total_latency_ms', 'mlp_recall', 'mlp_precision', 'mlp_f1',
            'faiss_latency_ms', 'decode_latency_ms', 'inference_latency_ms'
        ])
        table.append([
            f"w/ {selector_name}",
            f"{kf_avg['total_latency_ms']:.2f}",
            f"{kf_avg['faiss_latency_ms']:.2f}",
            f"{kf_avg['decode_latency_ms']:.2f}",
            f"{kf_avg['inference_latency_ms']:.2f}",
            f"{kf_avg['mlp_recall']:.3f}",
            f"{kf_avg['mlp_precision']:.3f}",
            f"{kf_avg['mlp_f1']:.3f}"
        ])
    
    baseline_avg = compute_averages(baseline_results, thresholds, 
                                    ['avg_latency_ms', 'avg_decode_latency_ms', 'avg_inference_latency_ms',
                                     'recall', 'precision', 'f1'])
    table.append([
        f'YOLO {yolo_model}',
        f"{baseline_avg['avg_latency_ms']:.2f}",
        "N/A",
        f"{baseline_avg['avg_decode_latency_ms']:.2f}",
        f"{baseline_avg['avg_inference_latency_ms']:.2f}",
        f"{baseline_avg['recall']:.3f}",
        f"{baseline_avg['precision']:.3f}",
        f"{baseline_avg['f1']:.3f}"
    ])
    
    return tabulate(
        table,
        headers=['Method', 'Total (ms)', 'FAISS (ms)', 'Decode (ms)', 'Inference (ms)', 'Recall', 'Precision', 'F1'],
        tablefmt='grid'
    )


def generate_recall_retention_table(no_kf_results, kf_results, thresholds):
    """Generate recall retention table."""
    table = []
    for threshold in thresholds:
        no_kf_recall = no_kf_results[threshold]['mlp_recall']
        kf_recall = kf_results[threshold]['mlp_recall']
        retention = (kf_recall / no_kf_recall * 100) if no_kf_recall > 0 else 0
        
        table.append([
            f">= {threshold}",
            f"{no_kf_recall:.3f}",
            f"{kf_recall:.3f}",
            f"{retention:.1f}%"
        ])
    
    return tabulate(
        table,
        headers=['Threshold', 'No-KF Recall', 'With-KF Recall', 'Retention'],
        tablefmt='grid'
    )


def convert_to_serializable(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_comprehensive_tests(
    data_dir: str,
    checkpoint_path: str,
    model_config,
    target: str,
    similarity_threshold: float,
    num_videos: int,
    thresholds: list[int],
    yolo_model: str,
    output_file: str,
    videos_source_dir: str,
    test_keyframes: bool,
    force_regenerate_keyframes: bool,
    force_regenerate_embeddings: bool,
    save_keyframes: bool,
    keyframe_selectors: list[str] = None,
    keyframe_params: dict = None
):
    """Run comprehensive tests."""

    embedder = CLIPEmbedder(CLIP_VIT_B32)
    def generate_dense_embds():
        from benchmark_functions import generate_full_embeddings
        videos_source_path = Path(videos_source_dir)
        video_files = sorted([f for f in videos_source_path.glob("*.mp4")])[:num_videos]
        for video_file in video_files:
            generate_full_embeddings(
                video_path=str(video_file),
                out_dir=str(video_file.parent),
                embedder=embedder,
                target_fps=1.0,
                force=force_regenerate_embeddings,
            )

    if test_keyframes and keyframe_selectors is None:
        keyframe_selectors = ['framediff']
    elif not test_keyframes:
        keyframe_selectors = []
    
    keyframe_params = keyframe_params or {}
    
    data_path = Path(data_dir)
    video_dirs = sorted([d for d in data_path.iterdir() 
                        if d.is_dir() and not d.name.startswith("_")])
    
    if len(video_dirs) < num_videos:
        print(f"Warning: Only {len(video_dirs)} videos available")
        num_videos = len(video_dirs)
    
    eval_videos = [v.name for v in video_dirs[:num_videos]]
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE TEST: {target} on {num_videos} videos")
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

    generate_dense_embds()
    
    # all_results['no_keyframes']['pipeline_results'] = benchmark_embds(
    #     data_dir=data_dir,
    #     checkpoint_path=checkpoint_path,
    #     model_config=model_config,
    #     target=target,
    #     similarity_threshold=similarity_threshold,
    #     num_videos=num_videos,
    #     thresholds=thresholds
    # )
    
    if test_keyframes:
        embedder = CLIPEmbedder(CLIP_VIT_B32)
        print(f"  Embedder: {embedder.name} on {embedder.device}")
        
        for selector_name in keyframe_selectors:
            all_results['keyframe_results'][selector_name] = benchmark_with_kf(
                kf_method=selector_name,
                kf_params=keyframe_params.get(selector_name, {}),
                data_dir=data_dir,
                checkpoint_path=checkpoint_path,
                model_config=model_config,
                target=target,
                similarity_threshold=similarity_threshold,
                num_videos=num_videos,
                thresholds=thresholds,
                videos_source_dir=videos_source_dir,
                embedder=embedder,
                force_regenerate=force_regenerate_embeddings,
                force_regenerate_kf=force_regenerate_keyframes,
                save_keyframes=save_keyframes
            )
    
    all_results['baseline_results'] = benchmark_baseline(
        data_dir=data_dir,
        target=target,
        num_videos=num_videos,
        thresholds=thresholds,
        yolo_model=yolo_model,
        videos_source_dir=videos_source_dir
    )
    
    print("\n" + "="*80)
    print("RESULTS")
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
    print("TABLE 2: END-TO-END LATENCY COMPARISON")
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
    print("TABLE 3: DETAILED METRICS (Averaged)")
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
        print(f"\n{'─'*80}")
        print("TABLE 4: RECALL RETENTION WITH KEYFRAMES")
        print(f"{'─'*80}")
        
        for selector_name in keyframe_selectors:
            print(f"\n{selector_name}:")
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
    print(f"Saved: {output_file}")
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
        videos_source_dir="/storage/ice1/8/3/rshah647/VIRATGround/videos_original",
        test_keyframes=True,
        force_regenerate_keyframes=False,
        save_keyframes=False,
        keyframe_selectors=['framediff'],
        keyframe_params={'framediff': {'k_mad': 2.5, 'min_spacing': 6}}
    )
