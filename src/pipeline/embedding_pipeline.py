"""Pipeline for generating embeddings with keyframe preselection."""

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
        raise ValueError(f"Unknown: {method}. Choose from {list(selectors.keys())}")
    return selectors[method](**kwargs)


def generate_embeddings(video_path: str, out_dir: str, preselector: BasePreselector = None,
                       embedder_config=CLIP_VIT_B32, target_fps: float = 1.0):
    """
    Generate embeddings for a video with keyframe preselection.
    
    If preselector is provided: selects keyframes first, then embeds only those.
    If preselector is None: embeds all sampled frames.
    """
    out_path = Path(out_dir)
    embeddings_dir = out_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 30.0
    
    stride = max(1, int(round(native_fps / target_fps)))
    
    # Sample frames
    sampled_frames = []
    sampled_to_orig = []
    orig_idx = 0
    samp_idx = 0
    
    if preselector:
        preselector.start()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if orig_idx % stride == 0:
            if preselector:
                preselector.process(frame, samp_idx)
            sampled_frames.append(frame)
            sampled_to_orig.append(orig_idx)
            samp_idx += 1
        
        orig_idx += 1
    
    cap.release()
    
    # Select keyframes if preselector provided
    if preselector:
        result = preselector.finalize()
        keyframe_indices = result.indices
        frames_to_embed = [sampled_frames[i] for i in keyframe_indices]
        
        # Build mapping: keyframe -> list of frames it represents
        keyframe_mapping = {}
        for i, kf_idx in enumerate(keyframe_indices):
            next_kf = keyframe_indices[i + 1] if i < len(keyframe_indices) - 1 else len(sampled_frames)
            keyframe_mapping[kf_idx] = list(range(kf_idx, next_kf))
    else:
        keyframe_indices = list(range(len(sampled_frames)))
        frames_to_embed = sampled_frames
        keyframe_mapping = {i: [i] for i in range(len(sampled_frames))}
    
    # Embed selected frames
    embedder = CLIPEmbedder(embedder_config)
    embeddings = []
    
    for i in range(0, len(frames_to_embed), BATCH_SIZE):
        batch = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_to_embed[i:i+BATCH_SIZE]]
        embs = embedder.embed(batch)
        embeddings.extend(embs)
    
    embeddings = np.array(embeddings)
    
    # Save
    np.save(embeddings_dir / "embds.npy", embeddings)
    np.save(embeddings_dir / "keyframe_indices.npy", np.array(keyframe_indices))
    np.save(embeddings_dir / "keyframe_mapping.npy", keyframe_mapping)
    np.save(embeddings_dir / "sampled_to_orig.npy", np.array(sampled_to_orig))
    
    metadata = {
        'total_frames': len(sampled_frames),
        'num_keyframes': len(keyframe_indices),
        'compression_ratio': len(sampled_frames) / max(len(keyframe_indices), 1),
        'uses_keyframes': preselector is not None,
        'method': preselector.__class__.__name__ if preselector else 'all_frames'
    }
    np.save(embeddings_dir / "metadata.npy", metadata)
    
    return metadata

