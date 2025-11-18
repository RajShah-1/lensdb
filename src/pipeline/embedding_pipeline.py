"""Generate embeddings with optional keyframe selection."""

from pathlib import Path
import numpy as np
import cv2

from src.embeddings.embedder import CLIPEmbedder
from src.keyframe.preselect_base import BasePreselector

BATCH_SIZE = 128


def generate_full_embeddings(video_path: str, out_dir: str, embedder: CLIPEmbedder, 
                             target_fps: float = 1.0, force: bool = False):
    """
    Generate embeddings for ALL sampled frames. Run once per video.
    Saves: embds_<model>_full.npy, sampled_to_orig.npy
    """
    out_path = Path(out_dir)
    embeddings_dir = out_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    full_embds_file = embeddings_dir / f"embds_{embedder.name}_full.npy"
    
    if full_embds_file.exists() and not force:
        print(f"  Full embeddings exist, skipping")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 30.0
    
    stride = max(1, int(round(native_fps / target_fps)))
    
    sampled_to_orig = []
    embeddings = []
    batch = []
    orig_idx = 0
    
    print(f"  Embedding frames in batches of {BATCH_SIZE}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if orig_idx % stride == 0:
            batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            sampled_to_orig.append(orig_idx)
            
            if len(batch) == BATCH_SIZE:
                print(f"  Embedding batch of {len(batch)} frames...")
                embs = embedder.embed(batch)
                embeddings.extend(embs)
                batch = []
        
        orig_idx += 1
    
    if batch:
        embs = embedder.embed(batch)
        embeddings.extend(embs)
    
    cap.release()
    
    embeddings = np.array(embeddings)
    
    np.save(full_embds_file, embeddings)
    np.save(embeddings_dir / "sampled_to_orig.npy", np.array(sampled_to_orig))
    print(f"  Saved {len(embeddings)} embeddings to {full_embds_file.name}")


def select_keyframes_from_full(video_path: str, out_dir: str, preselector: BasePreselector,
                               embedder: CLIPEmbedder, target_fps: float = 1.0, force: bool = False):
    """
    Run keyframe selection on video, pick embeddings from pre-computed full embeddings.
    Requires: embds_<model>_full.npy (run generate_full_embeddings first)
    Saves: embds.npy, keyframe_indices.npy, keyframe_mapping.npy, metadata.npy
    """
    out_path = Path(out_dir)
    embeddings_dir = out_path / "embeddings"
    
    full_embds_file = embeddings_dir / f"embds_{embedder.name}_full.npy"
    
    if not full_embds_file.exists():
        raise RuntimeError(f"Full embeddings not found: {full_embds_file}")
    
    metadata_file = embeddings_dir / "metadata.npy"
    if metadata_file.exists() and not force:
        metadata = np.load(metadata_file, allow_pickle=True).item()
        if metadata.get('method') == preselector.__class__.__name__:
            print(f"  Keyframes exist for {preselector.__class__.__name__}, skipping")
            return metadata
    
    full_embeddings = np.load(full_embds_file)
    total_frames = len(full_embeddings)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 30.0
    
    stride = max(1, int(round(native_fps / target_fps)))
    
    preselector.start()
    orig_idx = 0
    samp_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if orig_idx % stride == 0:
            preselector.process(frame, samp_idx)
            samp_idx += 1
        
        orig_idx += 1
    
    cap.release()
    
    result = preselector.finalize()
    keyframe_indices = result.indices
    num_keyframes = len(keyframe_indices)
    
    keyframe_embeddings = full_embeddings[keyframe_indices]
    
    keyframe_mapping = {}
    for i, kf_idx in enumerate(keyframe_indices):
        next_kf = keyframe_indices[i + 1] if i < num_keyframes - 1 else total_frames
        keyframe_mapping[kf_idx] = list(range(kf_idx, next_kf))
    
    np.save(embeddings_dir / "embds.npy", keyframe_embeddings)
    np.save(embeddings_dir / "keyframe_indices.npy", np.array(keyframe_indices))
    np.save(embeddings_dir / "keyframe_mapping.npy", keyframe_mapping)
    
    metadata = {
        'total_frames': total_frames,
        'num_keyframes': num_keyframes,
        'compression_ratio': total_frames / max(num_keyframes, 1),
        'uses_keyframes': True,
        'method': preselector.__class__.__name__
    }
    np.save(metadata_file, metadata)
    
    print(f"  Selected {num_keyframes}/{total_frames} keyframes ({metadata['compression_ratio']:.1f}x)")
    return metadata
