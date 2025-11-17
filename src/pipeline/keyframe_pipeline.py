"""Keyframe pipeline using streaming preselectors."""

from pathlib import Path
import numpy as np
import cv2

from src.embeddings.embedder import CLIPEmbedder, CLIP_VIT_B32
from src.keyframe.preselect_base import BasePreselector
from src.keyframe.preselect_methods import FrameDiffPreselector, SSIMPreselector, MOG2Preselector, FlowPreselector

BATCH_SIZE = 128


def get_preselector(method: str, **kwargs):
    """Get preselector by name."""
    selectors = {
        'framediff': FrameDiffPreselector,
        'ssim': SSIMPreselector,
        'mog2': MOG2Preselector,
        'flow': FlowPreselector
    }
    if method not in selectors:
        raise ValueError(f"Unknown method: {method}. Choose from {list(selectors.keys())}")
    return selectors[method](**kwargs)


def process_video(video_path: str, out_dir: str, preselector: BasePreselector, 
                 embedder_config=CLIP_VIT_B32, target_fps: float = 1.0):
    """Process single video: select keyframes with preselector, then embed only those."""
    out_path = Path(out_dir)
    embeddings_dir = out_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Select keyframes using preselector (streaming, no embeddings needed)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 30.0
    
    stride = max(1, int(round(native_fps / target_fps)))
    
    preselector.start()
    sampled_frames = []
    sampled_to_orig = []
    orig_idx = 0
    samp_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if orig_idx % stride == 0:
            preselector.process(frame, samp_idx)
            sampled_frames.append(frame)
            sampled_to_orig.append(orig_idx)
            samp_idx += 1
        
        orig_idx += 1
    
    cap.release()
    result = preselector.finalize()
    
    total_sampled = len(sampled_frames)
    num_keyframes = len(result.indices)
    
    # Embed only the selected keyframes
    embedder = CLIPEmbedder(embedder_config)
    keyframe_frames = [sampled_frames[i] for i in result.indices]
    keyframe_embeddings = []
    
    for i in range(0, len(keyframe_frames), BATCH_SIZE):
        batch = keyframe_frames[i:i+BATCH_SIZE]
        batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch]
        embs = embedder.embed(batch_rgb)
        keyframe_embeddings.extend(embs)
    
    keyframe_embeddings = np.array(keyframe_embeddings)
    
    # Build mapping
    keyframe_mapping = {}
    for i, kf_idx in enumerate(result.indices):
        next_kf = result.indices[i + 1] if i < len(result.indices) - 1 else total_sampled
        keyframe_mapping[kf_idx] = list(range(kf_idx, next_kf))
    
    # Save results
    np.save(embeddings_dir / "embds.npy", keyframe_embeddings)
    np.save(embeddings_dir / "keyframe_indices.npy", np.array(result.indices))
    np.save(embeddings_dir / "keyframe_mapping.npy", keyframe_mapping)
    np.save(embeddings_dir / "keyframe_scores.npy", result.scores)
    np.save(embeddings_dir / "sampled_to_orig.npy", np.array(sampled_to_orig))
    
    metadata = {
        'total_frames': total_sampled,
        'num_keyframes': num_keyframes,
        'compression_ratio': total_sampled / max(num_keyframes, 1),
        'method': preselector.__class__.__name__,
        'uses_keyframes': True
    }
    np.save(embeddings_dir / "metadata.npy", metadata)
    
    return {
        'total_sampled': total_sampled,
        'num_keyframes': num_keyframes,
        'compression_ratio': metadata['compression_ratio'],
        'keyframe_indices': result.indices
    }


def process_video_folder(data_dir: str, videos_source_dir: str, method: str = 'framediff',
                        method_params: dict = None, embedder_config=CLIP_VIT_B32,
                        num_videos: int = None, force_regenerate: bool = False,
                        target_fps: float = 1.0):
    """Process multiple videos in a folder with keyframe selection."""
    data_path = Path(data_dir)
    video_dirs = sorted([d for d in data_path.iterdir() 
                        if d.is_dir() and not d.name.startswith("_")])
    
    if num_videos is not None:
        video_dirs = video_dirs[:num_videos]
    
    preselector = get_preselector(method, **(method_params or {}))
    results = []
    
    for i, video_dir in enumerate(video_dirs, 1):
        video_path = Path(videos_source_dir) / f"{video_dir.name}.mp4"
        if not video_path.exists():
            continue
        
        embeddings_dir = video_dir / "embeddings"
        metadata_file = embeddings_dir / "metadata.npy"
        
        if metadata_file.exists() and not force_regenerate:
            metadata = np.load(metadata_file, allow_pickle=True).item()
            if metadata.get('uses_keyframes', False):
                results.append({
                    'video_name': video_dir.name,
                    'total_frames': metadata['total_frames'],
                    'num_keyframes': metadata['num_keyframes'],
                    'compression_ratio': metadata['compression_ratio'],
                    'selector_name': method
                })
                continue
        
        try:
            result = process_video(
                video_path=str(video_path),
                out_dir=str(video_dir),
                preselector=preselector,
                embedder_config=embedder_config,
                target_fps=target_fps
            )
            
            results.append({
                'video_name': video_dir.name,
                'total_frames': result['total_sampled'],
                'num_keyframes': result['num_keyframes'],
                'compression_ratio': result['compression_ratio'],
                'selector_name': method
            })
        except Exception as e:
            print(f"Error processing {video_dir.name}: {e}")
    
    return results
