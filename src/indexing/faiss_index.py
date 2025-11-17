import faiss
import numpy as np
import torch
import time
from pathlib import Path

from src.embeddings.embedder import CLIPEmbedder, CLIP_VIT_B32


class FAISSIndex:
    def __init__(self, data_dir: str, dim: int = 512):
        self.data_dir = Path(data_dir)
        self.dim = dim
        self.index_path = self.data_dir / "_index" / "clip.index"
        self.metadata_path = self.data_dir / "_index" / "video_metadata.npy"
        self.index = None
        self.video_metadata = None
    
    def build(self):
        print(f"Building FAISS index from {self.data_dir}")
        
        all_embeddings = []
        video_metadata = []
        
        for video_dir in sorted(self.data_dir.iterdir()):
            if not video_dir.is_dir() or video_dir.name.startswith("_"):
                continue
            
            emb_path = video_dir / "embeddings" / "embds.npy"
            if not emb_path.exists():
                continue
            
            embs = np.load(emb_path).astype("float32")
            embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
            
            metadata_path = video_dir / "embeddings" / "metadata.npy"
            uses_keyframes = False
            if metadata_path.exists():
                metadata = np.load(metadata_path, allow_pickle=True).item()
                uses_keyframes = metadata.get('uses_keyframes', False)
            
            all_embeddings.append(embs)
            video_metadata.append({
                'video_name': video_dir.name,
                'num_frames': len(embs),
                'start_idx': sum(len(e) for e in all_embeddings[:-1]),
                'uses_keyframes': uses_keyframes
            })
            
            print(f"  {video_dir.name}: {len(embs)} {'keyframes' if uses_keyframes else 'frames'}")
        
        if not all_embeddings:
            raise ValueError(f"No embeddings found in {self.data_dir}")
        
        all_embeddings = np.vstack(all_embeddings)
        print(f"Total embeddings: {all_embeddings.shape[0]}")
        
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(all_embeddings)
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        np.save(self.metadata_path, video_metadata)
        
        print(f"Index saved to {self.index_path}")
        self.video_metadata = video_metadata
        return self
    
    def load(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found. Run build() first.")
        
        self.index = faiss.read_index(str(self.index_path))
        self.video_metadata = np.load(self.metadata_path, allow_pickle=True)
        
        print(f"Loaded index with {self.index.ntotal} embeddings")
        return self
    
    def query_text(self, text_query: str, top_k: int | None = None, 
                   similarity_threshold: float = 0.0):
        if self.index is None:
            self.load()
        
        embedder = CLIPEmbedder(CLIP_VIT_B32)
        inputs = embedder.processor(text=[text_query], return_tensors="pt").to(embedder.device)
        
        with torch.no_grad():
            text_features = embedder.model.get_text_features(**inputs)
        
        query_vec = text_features.cpu().numpy().astype("float32")
        query_vec = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-8)
        
        k = top_k or self.index.ntotal
        
        start_time = time.perf_counter()
        similarities, indices = self.index.search(query_vec, k)
        faiss_latency_ms = (time.perf_counter() - start_time) * 1000
        
        mask = similarities[0] >= similarity_threshold
        filtered_indices = indices[0][mask]
        filtered_scores = similarities[0][mask]
        
        print(f"\nQuery: '{text_query}' (threshold={similarity_threshold:.2f})")
        print(f"Retrieved {len(filtered_indices)} frames")
        
        results = {}
        for global_idx, score in zip(filtered_indices, filtered_scores):
            for vid_meta in self.video_metadata:
                start = vid_meta['start_idx']
                end = start + vid_meta['num_frames']
                
                if start <= global_idx < end:
                    video_name = vid_meta['video_name']
                    local_frame_idx = int(global_idx - start)
                    
                    if video_name not in results:
                        results[video_name] = []
                    
                    results[video_name].append({
                        'frame_idx': local_frame_idx,
                        'similarity': float(score)
                    })
                    break
        
        for video_name in results:
            results[video_name] = sorted(results[video_name], key=lambda x: x['frame_idx'])
        
        for video_name, frames in results.items():
            print(f"  {video_name}: {len(frames)} frames")
        
        return results, faiss_latency_ms
