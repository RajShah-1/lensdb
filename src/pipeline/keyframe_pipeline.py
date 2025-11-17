from pathlib import Path
import numpy as np

from src.pipeline.video_reader import VideoReader
from src.embeddings.embedder import Embedder
from src.keyframe.keyframe_base import BaseKeyframeSelector
from src.utils import get_best_device

BATCH_SIZE = 128


class KeyframePipeline:
    """Pipeline: video → sample frames → embed all → select keyframes → save keyframe embeddings only."""
    
    def __init__(
        self, 
        video_path: str, 
        embedder: Embedder,
        keyframe_selector: BaseKeyframeSelector,
        out_dir: str | None = None
    ):
        self.video_path = Path(video_path)
        self.embedder = embedder
        self.keyframe_selector = keyframe_selector
        self.out_dir = Path(out_dir) if out_dir else Path("data/VIRAT") / self.video_path.stem
        self.frames_dir = self.out_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def _sample_frames(self):
        """Sample frames from video at 1 fps."""
        reader = VideoReader(str(self.video_path), "time", 0, 1, BATCH_SIZE)
        frames = []
        for batch in reader:
            frames.extend(batch)
        return frames
    
    def _embed_frames(self, frames):
        """Generate embeddings for all frames."""
        embeddings = []
        for i in range(0, len(frames), BATCH_SIZE):
            batch = frames[i:i+BATCH_SIZE]
            embs = self.embedder.embed(batch)
            embeddings.extend(embs)
        return np.array(embeddings)
    
    def _select_keyframes(self, embeddings, frames):
        """Select keyframes from embeddings."""
        normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        return self.keyframe_selector.select(embs=normalized, frames=frames, meta=None)
    
    def _build_keyframe_mapping(self, keyframe_indices, total_frames):
        """Build mapping from keyframe to represented frames."""
        mapping = {}
        for i, kf_idx in enumerate(keyframe_indices):
            next_kf_idx = keyframe_indices[i + 1] if i < len(keyframe_indices) - 1 else total_frames
            mapping[kf_idx] = list(range(kf_idx, next_kf_idx))
        return mapping
    
    def _save_results(self, embeddings, keyframe_indices, mapping, scores, total_frames):
        """Save embeddings and metadata."""
        embeddings_dir = self.out_dir / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(embeddings_dir / "embds.npy", embeddings)
        np.save(embeddings_dir / "keyframe_indices.npy", np.array(keyframe_indices))
        np.save(embeddings_dir / "keyframe_mapping.npy", mapping)
        np.save(embeddings_dir / "keyframe_scores.npy", scores)
        
        metadata = {
            'total_frames': total_frames,
            'num_keyframes': len(keyframe_indices),
            'compression_ratio': total_frames / len(keyframe_indices),
            'selector_name': self.keyframe_selector.name,
            'uses_keyframes': True
        }
        np.save(embeddings_dir / "metadata.npy", metadata)
        
        return metadata

    def run(self, save: bool = True):
        """Run the keyframe pipeline."""
        print(f"Device: {get_best_device()}")
        print(f"Keyframe selector: {self.keyframe_selector.name}")
        
        print("\n[1/4] Sampling frames...")
        frames = self._sample_frames()
        print(f"Sampled {len(frames)} frames")
        
        # If the keyframe selector needs embeddings, only then we need to generate embeddings for all frames
        # Otherwise, we can select keyframes based on the frames directly.
        if self.keyframe_selector.needs_embeddings:
            print("\n[2/4] Generating embeddings...")
            all_embeddings = self._embed_frames(frames)
            
            print("\n[3/4] Selecting keyframes...")
            kf_result = self._select_keyframes(all_embeddings, frames)
            keyframe_indices = kf_result.indices
            compression = len(frames) / len(keyframe_indices)
            print(f"Selected {len(keyframe_indices)} keyframes (compression: {compression:.1f}x)")
            
            print("\n[4/4] Building metadata...")
            keyframe_embeddings = all_embeddings[keyframe_indices]
        else:
            print("\n[2/4] Selecting keyframes (frame-based)...")
            kf_result = self._select_keyframes(None, frames)
            keyframe_indices = kf_result.indices
            compression = len(frames) / len(keyframe_indices)
            print(f"Selected {len(keyframe_indices)} keyframes (compression: {compression:.1f}x)")
            
            print("\n[3/4] Generating embeddings for keyframes only...")
            keyframe_frames = [frames[i] for i in keyframe_indices]
            keyframe_embeddings = self._embed_frames(keyframe_frames)
            
            print("\n[4/4] Building metadata...")
        
        keyframe_mapping = self._build_keyframe_mapping(keyframe_indices, len(frames))
        
        if save:
            metadata = self._save_results(
                keyframe_embeddings, 
                keyframe_indices, 
                keyframe_mapping,
                kf_result.score,
                len(frames)
            )
            print(f"\nSaved {len(keyframe_embeddings)} keyframe embeddings")
            print(f"Compression: {metadata['compression_ratio']:.1f}x")
        
        return {
            'keyframe_embeddings': keyframe_embeddings,
            'keyframe_indices': keyframe_indices,
            'keyframe_mapping': keyframe_mapping,
            'all_frame_count': len(frames),
            'keyframe_scores': kf_result.score
        }
