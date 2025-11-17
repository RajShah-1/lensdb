from pathlib import Path
import numpy as np

from src.pipeline.video_reader import VideoReader
from src.embeddings.embedder import Embedder, CLIPEmbedder, CLIP_VIT_B32
from src.keyframe.keyframe_base import BaseKeyframeSelector
from src.keyframe.keyframe_selectors import EmbeddingNoveltyKF, SSIMFlowKF, WindowKCenterKF
from src.utils import get_best_device

BATCH_SIZE = 128


def get_keyframe_selector(selector_name: str, **kwargs):
    """Get keyframe selector by name."""
    selectors = {
        'emb_novelty': EmbeddingNoveltyKF,
        'ssim_flow': SSIMFlowKF,
        'kcenter': WindowKCenterKF
    }
    if selector_name not in selectors:
        raise ValueError(f"Unknown selector: {selector_name}. Choose from {list(selectors.keys())}")
    return selectors[selector_name](**kwargs)


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
        """Select keyframes from embeddings or frames."""
        if embeddings is not None:
            normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            return self.keyframe_selector.select(embs=normalized, frames=frames, meta=None)
        else:
            return self.keyframe_selector.select(embs=None, frames=frames, meta=None)
    
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


def process_video_folder(
    data_dir: str,
    videos_source_dir: str,
    selector_name: str = 'emb_novelty',
    selector_params: dict = None,
    embedder_config = CLIP_VIT_B32,
    num_videos: int = None,
    force_regenerate: bool = False
):
    """
    Process multiple videos in a folder with keyframe selection.
    
    Args:
        data_dir: Output directory for embeddings
        videos_source_dir: Directory containing source .mp4 files
        selector_name: Keyframe selector name
        selector_params: Parameters for the selector
        embedder_config: Embedder configuration
        num_videos: Number of videos to process (None = all)
        force_regenerate: If True, regenerate even if embeddings exist
    
    Returns:
        List of metadata dictionaries for each video
    """
    data_path = Path(data_dir)
    video_dirs = sorted([d for d in data_path.iterdir() 
                        if d.is_dir() and not d.name.startswith("_")])
    
    if num_videos is not None:
        video_dirs = video_dirs[:num_videos]
    
    embedder = CLIPEmbedder(embedder_config)
    selector = get_keyframe_selector(selector_name, **(selector_params or {}))
    
    results = []
    print(f"\n{'='*80}")
    print(f"PROCESSING {len(video_dirs)} VIDEOS WITH KEYFRAMES")
    print(f"Selector: {selector_name}, Source: {videos_source_dir}")
    print(f"{'='*80}\n")
    
    for i, video_dir in enumerate(video_dirs, 1):
        video_path = Path(videos_source_dir) / f"{video_dir.name}.mp4"
        if not video_path.exists():
            print(f"[{i}/{len(video_dirs)}] ✗ Video not found: {video_path}")
            continue
        
        embeddings_dir = video_dir / "embeddings"
        metadata_file = embeddings_dir / "metadata.npy"
        
        if metadata_file.exists() and not force_regenerate:
            metadata = np.load(metadata_file, allow_pickle=True).item()
            if metadata.get('uses_keyframes', False):
                print(f"[{i}/{len(video_dirs)}] ✓ Keyframes already exist for {video_dir.name}")
                results.append({
                    'video_name': video_dir.name,
                    'total_frames': metadata['total_frames'],
                    'num_keyframes': metadata['num_keyframes'],
                    'compression_ratio': metadata['compression_ratio'],
                    'selector_name': selector_name
                })
                continue
        
        try:
            print(f"[{i}/{len(video_dirs)}] Processing {video_dir.name}...")
            pipeline = KeyframePipeline(str(video_path), embedder, selector, out_dir=str(video_dir))
            result = pipeline.run(save=True)
            
            metadata = {
                'video_name': video_dir.name,
                'total_frames': result['all_frame_count'],
                'num_keyframes': len(result['keyframe_indices']),
                'compression_ratio': result['all_frame_count'] / len(result['keyframe_indices']),
                'selector_name': selector_name
            }
            results.append(metadata)
            print(f"✓ {metadata['num_keyframes']} keyframes ({metadata['compression_ratio']:.1f}x)\n")
        except Exception as e:
            print(f"✗ Error: {e}\n")
    
    if results:
        print(f"{'='*80}")
        print(f"SUMMARY: {len(results)}/{len(video_dirs)} videos processed")
        avg_compression = sum(r['compression_ratio'] for r in results) / len(results)
        print(f"Average compression: {avg_compression:.1f}x")
        print(f"{'='*80}\n")
    
    return results
