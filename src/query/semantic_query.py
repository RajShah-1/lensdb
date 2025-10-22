import numpy as np
import torch
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
    
    def query(self, text_query: str, count_predicate=None, return_embeddings: bool = False):
        print(f"\n{'='*60}")
        print(f"SEMANTIC QUERY: '{text_query}'")
        print(f"{'='*60}")
        
        print(f"\n[Stage 1] FAISS Prefilter")
        candidates = self.index.query_text(text_query, top_k=self.top_k, 
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
                
                frame_result = {
                    'frame_idx': frame_data['frame_idx'],
                    'similarity': frame_data['similarity'],
                    'predicted_count': pred_count
                }
                
                if return_embeddings:
                    frame_result['embedding'] = frame_embeddings[i]
                
                video_results.append(frame_result)
            
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
    
    def query_and_save(self, text_query: str, count_predicate=None, output_dir: str | None = None):
        results = self.query(text_query, count_predicate)
        
        if not results:
            return
        
        if output_dir is None:
            output_dir = self.data_dir / "_query_results" / text_query.replace(" ", "_")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / "query_results.npy", results)
        
        for video_name, video_data in results.items():
            frame_indices = [f['frame_idx'] for f in video_data['frames']]
            np.savetxt(output_dir / f"{video_name}_frames.txt", frame_indices, fmt='%d')
        
        print(f"\nResults saved to {output_dir}")


def simple_query(data_dir: str, checkpoint_path: str, model_config: ModelConfig,
                 text_query: str, min_count: float | None = None, 
                 similarity_threshold: float = 0.2):
    pipeline = SemanticQueryPipeline(data_dir, checkpoint_path, model_config, 
                                    threshold=similarity_threshold)
    
    predicate = (lambda c: c >= min_count) if min_count else None
    return pipeline.query(text_query, count_predicate=predicate)
