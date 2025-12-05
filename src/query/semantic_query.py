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
                 threshold: float = 0.2, top_k: int | None = None, use_keyframes: bool = False):
        self.data_dir = Path(data_dir)
        self.threshold = threshold
        self.top_k = top_k
        self.use_keyframes = use_keyframes
        self.index = FAISSIndex(data_dir)
        self.index.load()
        
        self.device = get_best_device()
        self.model = CountPredictor(model_config).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Device: {self.device}, Parameters: {self.model.num_parameters():,}")
    
    def _load_keyframe_metadata(self, video_name: str):
        embeddings_dir = self.data_dir / video_name / "embeddings"
        metadata_path = embeddings_dir / "metadata.npy"
        
        if not metadata_path.exists():
            return None, None, None
            
        metadata = np.load(metadata_path, allow_pickle=True).item()
        if not metadata.get('uses_keyframes', False):
            return None, None, None
            
        keyframe_indices = np.load(embeddings_dir / "keyframe_indices.npy")
        keyframe_mapping = np.load(embeddings_dir / "keyframe_mapping.npy", allow_pickle=True).item()
        return keyframe_indices, keyframe_mapping, metadata
    
    def _expand_keyframes_to_frames(self, video_name: str, keyframe_results: list):
        if not self.use_keyframes:
            return keyframe_results
        
        keyframe_indices, keyframe_mapping, _ = self._load_keyframe_metadata(video_name)
        print(f"Keyframe indices: {keyframe_indices}")
        
        if keyframe_mapping is None:
            print(f"WARNING: Keyframe mapping not found for video {video_name}")
            return keyframe_results
        
        frame_results = []
        for kf_result in keyframe_results:
            kf_idx = kf_result['frame_idx']
            actual_kf_idx = keyframe_indices[kf_idx]
            represented_frames = keyframe_mapping.get(actual_kf_idx, [actual_kf_idx])
            
            for frame_idx in represented_frames:
                frame_results.append({
                    'frame_idx': frame_idx,
                    'similarity': kf_result['similarity'],
                    'predicted_count': kf_result['predicted_count'],
                    'from_keyframe': actual_kf_idx
                })
        
        frame_results.sort(key=lambda x: x['frame_idx'])
        return frame_results
    
    def query(self, text_query: str, emb_filename: str, count_predicate=None):
        print(f"\n{'='*60}")
        print(f"QUERY: '{text_query}'")
        print(f"{'='*60}")
        
        print(f"\n[1/2] FAISS Prefilter")
        candidates, _ = self.index.query_text(text_query, top_k=self.top_k, 
                                             similarity_threshold=self.threshold)
        
        if not candidates:
            print("No candidates found")
            return {}
        
        print(f"\n[2/2] Count Prediction")
        results = {}
        
        for video_name, frames_data in candidates.items():
            emb_path = self.data_dir / video_name / "embeddings" / emb_filename
            embeddings = np.load(emb_path).astype("float32")
            
            frame_indices = [f['frame_idx'] for f in frames_data]
            frame_embeddings = embeddings[frame_indices]
            
            with torch.no_grad():
                emb_tensor = torch.from_numpy(frame_embeddings).to(self.device)
                predictions = self.model(emb_tensor).cpu().numpy()
            
            video_results = []
            for i, frame_data in enumerate(frames_data):
                pred_count = float(predictions[i])
                pred_count_rounded = round(pred_count)
                
                if not count_predicate or count_predicate(pred_count_rounded):
                    video_results.append({
                        'frame_idx': frame_data['frame_idx'],
                        'similarity': frame_data['similarity'],
                        'predicted_count': pred_count_rounded
                    })
                
            if self.use_keyframes:
                video_results = self._expand_keyframes_to_frames(video_name, video_results)
            
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
                          eval_videos: list, data_dir: str, emb_filename: str):
        def load_ground_truth(video_name: str, target: str):
            counts_file = Path(data_dir) / video_name / "counts.csv"
            counts = {}
            with open(counts_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    frame_id = int(row['frame_id'])
                    counts[frame_id] = int(row['car_count' if target == "car" else 'people_count'])
            return counts
        
        candidates, faiss_latency_ms = self.index.query_text(
            text_query, top_k=self.top_k, similarity_threshold=self.threshold
        )
        
        faiss_tp, faiss_fp, faiss_fn, faiss_retrieved_count = 0, 0, 0, 0
        
        for video_name in eval_videos:
            gt_counts = load_ground_truth(video_name, text_query)
            gt_positive = {fid for fid, count in gt_counts.items() if count >= count_threshold}
            
            if video_name in candidates:
                # Expand keyframes to frames for proper FAISS evaluation
                keyframe_results = [{'frame_idx': f['frame_idx'], 
                                    'similarity': f['similarity'], 
                                    'predicted_count': 0} 
                                   for f in candidates[video_name]]
                expanded_results = keyframe_results
                if self.use_keyframes:
                    expanded_results = self._expand_keyframes_to_frames(video_name, keyframe_results)
                retrieved_frames = {f['frame_idx'] for f in expanded_results}
            else:
                retrieved_frames = set()
            
            faiss_retrieved_count += len(retrieved_frames)
            faiss_tp += len(gt_positive & retrieved_frames)
            faiss_fp += len(retrieved_frames - gt_positive)
            faiss_fn += len(gt_positive - retrieved_frames)
        
        faiss_denom_p = faiss_tp + faiss_fp
        faiss_denom_r = faiss_tp + faiss_fn
        faiss_precision = faiss_tp / faiss_denom_p if faiss_denom_p > 0 else 0.0
        faiss_recall = faiss_tp / faiss_denom_r if faiss_denom_r > 0 else 0.0
        faiss_denom_f1 = faiss_precision + faiss_recall
        faiss_f1 = 2 * faiss_precision * faiss_recall / faiss_denom_f1 if faiss_denom_f1 > 0 else 0.0
        
        results = {}
        decode_latency_ms = 0.0
        inference_latency_ms = 0.0
        
        for video_name, frames_data in candidates.items():
            # Decode: Load embeddings from disk
            decode_start = time.perf_counter()
            emb_path = self.data_dir / video_name / "embeddings" / emb_filename
            embeddings = np.load(emb_path).astype("float32")
            frame_indices = [f['frame_idx'] for f in frames_data]
            frame_embeddings = embeddings[frame_indices]
            decode_latency_ms += (time.perf_counter() - decode_start) * 1000
            
            # Inference: Model forward pass
            inference_start = time.perf_counter()
            with torch.no_grad():
                emb_tensor = torch.from_numpy(frame_embeddings).to(self.device)
                predictions = self.model(emb_tensor).cpu().numpy()
            inference_latency_ms += (time.perf_counter() - inference_start) * 1000
            
            video_results = []
            for i, frame_data in enumerate(frames_data):
                pred_count_rounded = round(float(predictions[i]))
                # print(f"P_rounded: {pred_count_rounded} for frame {frame_data['frame_idx']}")
                if pred_count_rounded >= count_threshold:
                    video_results.append({
                        'frame_idx': frame_data['frame_idx'],
                        'similarity': frame_data['similarity'],
                        'predicted_count': pred_count_rounded
                    })
            
            if self.use_keyframes:
                video_results = self._expand_keyframes_to_frames(video_name, video_results)
            
            if video_results:
                results[video_name] = {
                    'frames': video_results,
                    'total_frames': len(video_results),
                    'avg_count': float(np.mean([f['predicted_count'] for f in video_results]))
                }
        
        mlp_latency_ms = decode_latency_ms + inference_latency_ms
        
        mlp_tp, mlp_fp, mlp_fn, mlp_retrieved_count = 0, 0, 0, 0
        positive_kf_indices = []
        negative_kf_indices = []
        
        for video_name in eval_videos:
            gt_counts = load_ground_truth(video_name, text_query)
            gt_positive = {fid for fid, count in gt_counts.items() if count >= count_threshold}
            
            if video_name in results:
                retrieved_frames = {f['frame_idx'] for f in results[video_name]['frames']}
            else:
                retrieved_frames = set()
            
            if video_name in candidates:
                candidate_kf_indices = [f['frame_idx'] for f in candidates[video_name]]
                result_kf_indices = [f['frame_idx'] for f in results.get(video_name, {}).get('frames', [])] if video_name in results else []
                
                positive_kf_indices.extend(result_kf_indices)
                negative_kf_indices.extend([idx for idx in candidate_kf_indices if idx not in result_kf_indices])
            
            mlp_retrieved_count += len(retrieved_frames)
            mlp_tp += len(gt_positive & retrieved_frames)
            mlp_fp += len(retrieved_frames - gt_positive)
            mlp_fn += len(gt_positive - retrieved_frames)
        
        mlp_denom_p = mlp_tp + mlp_fp
        mlp_denom_r = mlp_tp + mlp_fn
        mlp_precision = mlp_tp / mlp_denom_p if mlp_denom_p > 0 else 0.0
        mlp_recall = mlp_tp / mlp_denom_r if mlp_denom_r > 0 else 0.0
        mlp_denom_f1 = mlp_precision + mlp_recall
        mlp_f1 = 2 * mlp_precision * mlp_recall / mlp_denom_f1 if mlp_denom_f1 > 0 else 0.0
        
        return {
            'faiss_latency_ms': faiss_latency_ms,
            'decode_latency_ms': decode_latency_ms,
            'inference_latency_ms': inference_latency_ms,
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
            'mlp_tp': mlp_tp,
            'mlp_fp': mlp_fp,
            'mlp_fn': mlp_fn,
            'positive_kf_indices': positive_kf_indices,
            'negative_kf_indices': negative_kf_indices,
        }
