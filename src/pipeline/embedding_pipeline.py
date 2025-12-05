"""Generate embeddings with optional keyframe selection."""

from pathlib import Path
import numpy as np
import cv2

from src.embeddings.embedder import CLIPEmbedder
from src.keyframe.preselect_base import BasePreselector
from src.video.video_reader import iter_video_frames

BATCH_SIZE = 128


def generate_full_embeddings(video_path: str, out_dir: str, embedder: CLIPEmbedder, 
                             target_fps: float, force: bool):
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
    
    # Stream frames with caching - memory efficient
    embedding_batches = []
    sampled_to_orig = []
    batch = []
    
    print(f"  Embedding frames in batches of {BATCH_SIZE}...")

    for frame, frame_idx in iter_video_frames(video_path, out_dir, target_fps):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch.append(rgb_frame)
        sampled_to_orig.append(frame_idx)
        
        if len(batch) == BATCH_SIZE:
            embs = embedder.embed(batch)
            embedding_batches.append(embs)
            batch = []
    
    if batch:
        embs = embedder.embed(batch)
        embedding_batches.append(embs)
    
    embeddings = np.vstack(embedding_batches)
    
    np.save(full_embds_file, embeddings)
    np.save(embeddings_dir / "sampled_to_orig.npy", np.array(sampled_to_orig))
    print(f"  Saved {len(embeddings)} embeddings to {full_embds_file.name}")


def select_keyframes_from_full(video_path: str, out_dir: str, preselector: BasePreselector,
                               embedder: CLIPEmbedder, target_fps: float, force: bool,
                               save_keyframes: bool):
    """
    Run keyframe selection on video, pick embeddings from pre-computed full embeddings.
    Requires: embds_<model>_full.npy (run generate_full_embeddings first)
    Saves: embds.npy, keyframe_indices.npy, keyframe_mapping.npy, metadata.npy
    Optionally saves keyframe images if save_keyframes=True
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
    
    # Stream frames with caching - memory efficient
    preselector.start()
    
    # Store frames only if saving keyframes
    frames_list = [] if save_keyframes else None
    
    for samp_idx, (frame, _) in enumerate(iter_video_frames(video_path, out_dir, target_fps)):
        preselector.process(frame, samp_idx)
        if save_keyframes:
            frames_list.append(frame)
    
    result = preselector.finalize()
    keyframe_indices = result.indices
    num_keyframes = len(keyframe_indices)
    
    if save_keyframes:
        keyframes_dir = out_path / "keyframes" / preselector.__class__.__name__
        keyframes_dir.mkdir(exist_ok=True)
        
        sampled_to_orig = np.load(embeddings_dir / "sampled_to_orig.npy")
        
        # Get native FPS for timestamps
        native_fps = 30.0
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0 and not np.isnan(fps):
                native_fps = fps
            else:
                print(f"  [WARNING] FPS not found for {video_path}")
            cap.release()
        
        with open(keyframes_dir / "timestamps.txt", 'w') as f:
            f.write("frame_idx,orig_frame,timestamp_sec\n")
            for kf_idx in keyframe_indices:
                frame = frames_list[kf_idx]
                cv2.imwrite(str(keyframes_dir / f"frame_{kf_idx:06d}.jpg"), frame)
                orig_frame = sampled_to_orig[kf_idx]
                timestamp = orig_frame / native_fps
                f.write(f"{kf_idx},{orig_frame},{timestamp:.3f}\n")
        
        print(f"  Saved {num_keyframes} keyframe images")
    
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
