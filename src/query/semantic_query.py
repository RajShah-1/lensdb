import numpy as np
import torch
import time
import csv
from pathlib import Path

from src.indexing.faiss_index import FAISSIndex
from src.models.count_predictor import CountPredictor
from src.models.model_configs import ModelConfig
from src.utils import get_best_device


class SemanticQueryPipeline:
    def __init__(self, data_dir: str, checkpoint_path: str, model_config: ModelConfig,
                 threshold: float = 0.2, top_k: int | None = None):
        self.data_dir = Path(data_dir)
        self.threshold = threshold
        self.top_k = top_k
        
        self.index = FAISSIndex(data_dir)
        self.index.load()
        
        self.device = get_best_device()
        self.model = CountPredictor(model_config).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Device: {self.device}, Parameters: {self.model.num_parameters():,}")
    
    def query(self, text_query: str, count_predicate=None):
        """
        Run semantic query and return matching frames.
        
        Args:
            text_query: Text description to search for (e.g., "car", "person")
            count_predicate: Optional function to filter by predicted count
        
        Returns:
            Dictionary of results by video name
        """
        print(f"\n{'='*60}")
        print(f"SEMANTIC QUERY: '{text_query}'")
        print(f"{'='*60}")
        
        print(f"\n[Stage 1] FAISS Prefilter")
        candidates, _ = self.index.query_text(text_query, top_k=self.top_k, 
                                             similarity_threshold=self.threshold)
        
        if not candidates:
            print("No frames passed prefilter")
            return {}
        
        print(f"\n[Stage 2] Count Prediction")
        results = {}
        
        for video_name, frames_data in candidates.items():
            emb_path = self.data_dir / video_name / "embeddings" / "embds.npy"
            embeddings = np.load(emb_path).astype("float32")
            
            frame_indices = [f['frame_idx'] for f in frames_data]
            frame_embeddings = embeddings[frame_indices]
            
            with torch.no_grad():
                emb_tensor = torch.from_numpy(frame_embeddings).to(self.device)
                predictions = self.model(emb_tensor).cpu().numpy()
            
            video_results = []
            for i, frame_data in enumerate(frames_data):
                pred_count = float(predictions[i])
                
                if count_predicate and not count_predicate(pred_count):
                    continue
                
                video_results.append({
                    'frame_idx': frame_data['frame_idx'],
                    'similarity': frame_data['similarity'],
                    'predicted_count': pred_count
                })
            
            if video_results:
                results[video_name] = {
                    'frames': video_results,
                    'total_frames': len(video_results),
                    'avg_count': float(np.mean([f['predicted_count'] for f in video_results]))
                }
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        
        total_frames = sum(v['total_frames'] for v in results.values())
        print(f"Total matching frames: {total_frames}")
        
        for video_name, video_data in results.items():
            print(f"\n{video_name}")
            print(f"  Frames: {video_data['total_frames']}")
            print(f"  Avg count: {video_data['avg_count']:.2f}")
            
            for frame in video_data['frames'][:3]:
                print(f"    Frame {frame['frame_idx']:4d}: "
                      f"sim={frame['similarity']:.3f}, "
                      f"count={frame['predicted_count']:.2f}")
            if len(video_data['frames']) > 3:
                print(f"    ... and {len(video_data['frames']) - 3} more")
        
        return results
    
    def query_with_metrics(self, text_query: str, count_threshold: int,
                          eval_videos: list, data_dir: str):
        """
        Query with detailed metrics tracking for ablation study.
        Returns metrics at FAISS and MLP stages separately.
        """
        # Load ground truth for eval videos
        def load_ground_truth(video_name: str, target: str):
            counts_file = Path(data_dir) / video_name / "counts.csv"
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
        
        # Stage 1: FAISS lookup with timing
        candidates, faiss_latency_ms = self.index.query_text(
            text_query, 
            top_k=self.top_k,
            similarity_threshold=self.threshold
        )
        
        # Calculate FAISS stage metrics
        faiss_tp, faiss_fp, faiss_fn = 0, 0, 0
        faiss_retrieved_count = 0
        
        for video_name in eval_videos:
            gt_counts = load_ground_truth(video_name, text_query)
            gt_positive = {fid for fid, count in gt_counts.items() if count >= count_threshold}
            
            if video_name in candidates:
                retrieved_frames = {f['frame_idx'] for f in candidates[video_name]}
            else:
                retrieved_frames = set()
            
            faiss_retrieved_count += len(retrieved_frames)
            faiss_tp += len(gt_positive & retrieved_frames)
            faiss_fp += len(retrieved_frames - gt_positive)
            faiss_fn += len(gt_positive - retrieved_frames)
        
        faiss_precision = faiss_tp / (faiss_tp + faiss_fp) if (faiss_tp + faiss_fp) > 0 else 0.0
        faiss_recall = faiss_tp / (faiss_tp + faiss_fn) if (faiss_tp + faiss_fn) > 0 else 0.0
        faiss_f1 = 2 * faiss_precision * faiss_recall / (faiss_precision + faiss_recall) if (faiss_precision + faiss_recall) > 0 else 0.0
        
        # Stage 2: MLP prediction with timing
        mlp_start = time.perf_counter()
        results = {}
        
        for video_name, frames_data in candidates.items():
            emb_path = self.data_dir / video_name / "embeddings" / "embds.npy"
            embeddings = np.load(emb_path).astype("float32")
            
            frame_indices = [f['frame_idx'] for f in frames_data]
            frame_embeddings = embeddings[frame_indices]
            
            with torch.no_grad():
                emb_tensor = torch.from_numpy(frame_embeddings).to(self.device)
                predictions = self.model(emb_tensor).cpu().numpy()
            
            video_results = []
            for i, frame_data in enumerate(frames_data):
                pred_count = float(predictions[i])
                
                if pred_count >= count_threshold:
                    video_results.append({
                        'frame_idx': frame_data['frame_idx'],
                        'similarity': frame_data['similarity'],
                        'predicted_count': pred_count
                    })
            
            if video_results:
                results[video_name] = {
                    'frames': video_results,
                    'total_frames': len(video_results),
                    'avg_count': float(np.mean([f['predicted_count'] for f in video_results]))
                }
        
        mlp_latency_ms = (time.perf_counter() - mlp_start) * 1000
        
        # Calculate MLP stage metrics (final pipeline)
        mlp_tp, mlp_fp, mlp_fn = 0, 0, 0
        mlp_retrieved_count = 0
        
        for video_name in eval_videos:
            gt_counts = load_ground_truth(video_name, text_query)
            gt_positive = {fid for fid, count in gt_counts.items() if count >= count_threshold}
            
            if video_name in results:
                retrieved_frames = {f['frame_idx'] for f in results[video_name]['frames']}
            else:
                retrieved_frames = set()
            
            mlp_retrieved_count += len(retrieved_frames)
            mlp_tp += len(gt_positive & retrieved_frames)
            mlp_fp += len(retrieved_frames - gt_positive)
            mlp_fn += len(gt_positive - retrieved_frames)
        
        mlp_precision = mlp_tp / (mlp_tp + mlp_fp) if (mlp_tp + mlp_fp) > 0 else 0.0
        mlp_recall = mlp_tp / (mlp_tp + mlp_fn) if (mlp_tp + mlp_fn) > 0 else 0.0
        mlp_f1 = 2 * mlp_precision * mlp_recall / (mlp_precision + mlp_recall) if (mlp_precision + mlp_recall) > 0 else 0.0
        
        return {
            'faiss_latency_ms': faiss_latency_ms,
            'mlp_latency_ms': mlp_latency_ms,
            'total_latency_ms': faiss_latency_ms + mlp_latency_ms,
            'faiss_precision': faiss_precision,
            'faiss_recall': faiss_recall,
            'faiss_f1': faiss_f1,
            'faiss_retrieved': faiss_retrieved_count,
            'mlp_precision': mlp_precision,
            'mlp_recall': mlp_recall,
            'mlp_f1': mlp_f1,
            'mlp_retrieved': mlp_retrieved_count,
        }
