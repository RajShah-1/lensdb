"""Simple video reader with frame caching at 1 FPS."""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple


def get_native_fps(cap) -> float:
    """Get video FPS, default to 30 if unknown."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        return 30.0
    return fps


def iter_video_frames(video_path: str, out_dir: str, target_fps: float = 1.0) -> Iterator[Tuple[np.ndarray, int]]:
    """
    Iterate over video frames at target_fps, with disk caching.
    
    Yields:
        (frame, frame_index): BGR frame (numpy array) and original frame index
    """
    out_path = Path(out_dir)
    cache_dir = out_path / "fps1frames"
    
    # Try to load from cache first
    if cache_dir.exists():
        frame_files = sorted(cache_dir.glob("frame_*.jpg"))
        if frame_files:
            print(f"  Loading from cache: {cache_dir}")
            for frame_file in frame_files:
                # Extract frame index from filename: frame_000123.jpg -> 123
                idx = int(frame_file.stem.split('_')[1])
                frame = cv2.imread(str(frame_file))
                if frame is not None:
                    yield frame, idx
            return
    
    # Cache miss - decode video and save frames
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    
    native_fps = get_native_fps(cap)
    stride = max(1, int(round(native_fps / target_fps)))
    
    print(f"  Decoding and caching frames at {target_fps} FPS (stride={stride})...")
    
    orig_idx = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if orig_idx % stride == 0:
            # Save to cache
            frame_file = cache_dir / f"frame_{orig_idx:06d}.jpg"
            cv2.imwrite(str(frame_file), frame)
            
            yield frame, orig_idx
            frame_count += 1
        
        orig_idx += 1
    
    cap.release()
    print(f"  Cached {frame_count} frames to {cache_dir}")

