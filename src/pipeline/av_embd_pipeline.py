"""Generate embeddings with optional keyframe selection (PyAV-only)."""

from pathlib import Path
import numpy as np
import cv2
import av
from PIL import Image

from src.embeddings.embedder import CLIPEmbedder
from src.keyframe.preselect_base import BasePreselector

BATCH_SIZE = 128


def _get_native_fps(stream) -> float:
    """Extract FPS or fallback to 30."""
    if stream.average_rate is not None:
        try:
            return float(stream.average_rate)
        except (TypeError, ZeroDivisionError):
            pass
    return 30.0


def generate_full_embeddings(
    video_path: str,
    out_dir: str,
    embedder: CLIPEmbedder,
    target_fps: float,
    force: bool,
):
    """
    Generate embeddings for ALL sampled frames.
    Saves: embds_<model>_full.npy, sampled_to_orig.npy
    """
    out_path = Path(out_dir)
    embeddings_dir = out_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    full_embds_file = embeddings_dir / f"embds_{embedder.name}_full.npy"

    if full_embds_file.exists() and not force:
        print("  Full embeddings exist, skipping")
        return

    # --- PyAV decode ---
    container = av.open(video_path)
    stream = next(s for s in container.streams if s.type == "video")
    stream.thread_type = "AUTO"

    native_fps = _get_native_fps(stream)
    stride = max(1, int(round(native_fps / target_fps)))

    sampled_to_orig = []
    embeddings = []
    batch = []
    orig_idx = 0

    print(f"  Embedding frames in batches of {BATCH_SIZE}...")

    for frame in container.decode(stream):
        if orig_idx % stride == 0:
            rgb = frame.to_ndarray(format="rgb24")
            batch.append(rgb)
            sampled_to_orig.append(orig_idx)

            if len(batch) == BATCH_SIZE:
                print(f"  Embedding batch of {len(batch)} frames...")
                embs = embedder.embed(batch)
                embeddings.extend(embs)
                batch = []

        orig_idx += 1

    container.close()

    if batch:
        embs = embedder.embed(batch)
        embeddings.extend(embs)

    embeddings = np.array(embeddings)

    np.save(full_embds_file, embeddings)
    np.save(embeddings_dir / "sampled_to_orig.npy", np.array(sampled_to_orig))

    print(f"  Saved {len(embeddings)} embeddings to {full_embds_file.name}")


def select_keyframes_from_full(
    video_path: str,
    out_dir: str,
    preselector: BasePreselector,
    embedder: CLIPEmbedder,
    target_fps: float,
    force: bool,
    save_keyframes: bool,
):
    """
    Run keyframe selection on video using pre-computed embeddings.
    Optionally saves keyframe images.
    """
    out_path = Path(out_dir)
    embeddings_dir = out_path / "embeddings"

    full_embds_file = embeddings_dir / f"embds_{embedder.name}_full.npy"
    if not full_embds_file.exists():
        raise RuntimeError(f"Full embeddings not found: {full_embds_file}")

    metadata_file = embeddings_dir / "metadata.npy"

    if metadata_file.exists() and not force:
        metadata = np.load(metadata_file, allow_pickle=True).item()
        if metadata.get("method") == preselector.__class__.__name__:
            print(f"  Keyframes exist for {preselector.__class__.__name__}, skipping")
            return metadata

    full_embeddings = np.load(full_embds_file)
    total_frames = len(full_embeddings)

    # --- PyAV decode ---
    container = av.open(video_path)
    stream = next(s for s in container.streams if s.type == "video")
    stream.thread_type = "AUTO"

    native_fps = _get_native_fps(stream)
    stride = max(1, int(round(native_fps / target_fps)))

    preselector.start()
    orig_idx = 0
    samp_idx = 0
    sampled_frames = [] if save_keyframes else None

    for frame in container.decode(stream):
        if orig_idx % stride == 0:
            rgb = frame.to_ndarray(format="rgb24")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            preselector.process(bgr, samp_idx)

            if save_keyframes:
                sampled_frames.append(bgr)

            samp_idx += 1

        orig_idx += 1

    container.close()

    # === Finalize ===
    result = preselector.finalize()
    keyframe_indices = result.indices
    num_keyframes = len(keyframe_indices)

    # === Save keyframes ===
    if save_keyframes:
        keyframes_dir = out_path / "keyframes"
        keyframes_dir.mkdir(exist_ok=True)

        sampled_to_orig = np.load(embeddings_dir / "sampled_to_orig.npy")

        with open(keyframes_dir / "timestamps.txt", "w") as f:
            f.write("frame_idx,orig_frame,timestamp_sec\n")

            for kf_idx in keyframe_indices:
                bgr = sampled_frames[kf_idx]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                Image.fromarray(rgb).save(keyframes_dir / f"frame_{kf_idx:06d}.jpg")

                orig_frame = sampled_to_orig[kf_idx]
                ts = orig_frame / native_fps
                f.write(f"{kf_idx},{orig_frame},{ts:.3f}\n")

        print(f"  Saved {num_keyframes} keyframe images")

    # === Embeddings ===
    keyframe_embeddings = full_embeddings[keyframe_indices]

    # === Keyframe mapping ===
    keyframe_mapping = {}
    for i, kf_idx in enumerate(keyframe_indices):
        next_kf = keyframe_indices[i + 1] if i < num_keyframes - 1 else total_frames
        keyframe_mapping[kf_idx] = list(range(kf_idx, next_kf))

    np.save(embeddings_dir / "embds.npy", keyframe_embeddings)
    np.save(embeddings_dir / "keyframe_indices.npy", np.array(keyframe_indices))
    np.save(embeddings_dir / "keyframe_mapping.npy", keyframe_mapping)

    metadata = {
        "total_frames": total_frames,
        "num_keyframes": num_keyframes,
        "compression_ratio": total_frames / max(num_keyframes, 1),
        "uses_keyframes": True,
        "method": preselector.__class__.__name__,
    }

    np.save(metadata_file, metadata)

    print(
        f"  Selected {num_keyframes}/{total_frames} keyframes "
        f"({metadata['compression_ratio']:.1f}x)"
    )

    return metadata
