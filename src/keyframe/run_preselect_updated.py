# src/keyframe/run_preselect.py
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.keyframe.preselect_methods import (
    FrameDiffPreselector,
    SSIMPreselector,
    MOG2Preselector,
    FlowPreselector,
)
from src.keyframe.preselect_base import BasePreselector


def run_preselector(
    video_path: str,
    selector: BasePreselector,
    target_fps: float = 1.0,
    max_sampled_frames: int | None = None,
    plot: bool = True,
    save_frames: bool = False,
    out_root: Path | None = None,
):
    """
    Run one preselector on a video, downsampling to target_fps first.

    - Reads video with OpenCV.
    - Downsamples frames to ~ target_fps by skipping frames.
    - Feeds ONLY those sampled frames to the preselector.
    - Returns preselector result + stores mapping from sampled index -> original index.
    - Optionally:
        * saves novelty plot
        * saves selected keyframe images to disk
        * stores metadata as .npy and .csv
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        # Fallback if FPS is not set properly
        print(f"  [WARNING] FPS not found for {video_path}")
        native_fps = 30.0

    # stride in original frames to get approx target_fps
    stride = max(1, int(round(native_fps / target_fps)))

    video_name = Path(video_path).stem
    name = selector.__class__.__name__
    if out_root is None:
        out_root = Path("runs/preselect") / video_name / name
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n[{name}] on {video_name}")
    print(f"  native_fps = {native_fps:.2f}, target_fps = {target_fps:.2f}, stride = {stride}")

    selector.start()

    orig_idx = 0          # index in original video
    samp_idx = 0          # index in sampled 1-FPS stream
    sampled_to_orig: List[int] = []  # mapping sampled index -> original frame index

    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every 'stride'-th frame to approximate 1 FPS
        if orig_idx % stride == 0:
            keep = selector.process(frame, samp_idx)
            sampled_to_orig.append(orig_idx)
            samp_idx += 1

            if max_sampled_frames and samp_idx >= max_sampled_frames:
                break

        orig_idx += 1

    cap.release()
    res = selector.finalize()
    t1 = time.time()

    total_orig_frames = orig_idx
    total_sampled_frames = samp_idx
    kept_count = len(res.indices)

    # Basic stats
    print(f"  Total original frames:   {total_orig_frames}")
    print(f"  Sampled frames (@~1fps): {total_sampled_frames}")
    print(f"  Kept keyframes (sample): {kept_count}")
    print(f"  Compression (vs sampled): {kept_count / max(total_sampled_frames, 1):.4f}")
    print(f"  Runtime (sec):           {t1 - t0:.2f}")
    print(f"  Sampled FPS processed:   {total_sampled_frames / max(t1 - t0, 1e-6):.2f}")

    # Map sampled keyframe indices back to original frame indices and timestamps
    sampled_to_orig = np.array(sampled_to_orig, dtype=int)   # shape: [T_sampled]
    kf_sampled = np.array(res.indices, dtype=int)            # indices in sampled space
    kf_orig = sampled_to_orig[kf_sampled]                    # indices in original video
    kf_time_sec = kf_orig / native_fps

    # Save metadata: which frames were picked
    np.save(out_root / "scores.npy", np.array(res.scores, dtype=float))
    np.save(out_root / "sampled_to_orig.npy", sampled_to_orig)
    np.save(out_root / "keyframes_sampled_idx.npy", kf_sampled)
    np.save(out_root / "keyframes_orig_idx.npy", kf_orig)
    np.save(out_root / "keyframes_time_sec.npy", kf_time_sec)

    # Also dump a CSV for easy eyeballing
    import csv
    csv_path = out_root / "keyframes.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sampled_idx", "orig_idx", "time_sec"])
        for s_idx, o_idx, t_sec in zip(kf_sampled, kf_orig, kf_time_sec):
            writer.writerow([int(s_idx), int(o_idx), float(t_sec)])
    print(f"  Saved keyframe metadata to {csv_path}")

    # Optional: save keyframe images
    if save_frames and kept_count > 0:
        print("  Saving keyframe images ...")
        cap2 = cv2.VideoCapture(video_path)
        if not cap2.isOpened():
            print("  [WARN] Could not reopen video to save frames.")
        else:
            frames_dir = out_root / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for o_idx in kf_orig:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, int(o_idx))
                ret, fr = cap2.read()
                if not ret:
                    continue
                out_path = frames_dir / f"frame_orig_{int(o_idx):06d}.jpg"
                cv2.imwrite(str(out_path), fr)
            cap2.release()
            print(f"  Saved {kept_count} keyframe images to {frames_dir}")

    # Plot novelty scores + selected indices
    if plot and len(res.scores) > 0:
        import matplotlib
        # Use non-interactive backend to be safe on servers
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        scores = np.array(res.scores, dtype=float)
        plt.figure(figsize=(12, 3))
        plt.plot(scores, lw=1.0)
        if kept_count > 0:
            plt.scatter(
                kf_sampled,
                scores[kf_sampled],
                s=20,
                label="Selected",
            )
        plt.title(f"{name} (sampled @ {target_fps:.1f} FPS)")
        plt.xlabel("Sampled frame index")
        plt.ylabel("Novelty score")
        if kept_count > 0:
            plt.legend()
        plt.tight_layout()
        plot_path = out_root / "scores.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"  Saved score plot to {plot_path}")

    return {
        "native_fps": native_fps,
        "stride": stride,
        "total_orig_frames": total_orig_frames,
        "total_sampled_frames": total_sampled_frames,
        "kept_count": kept_count,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--target_fps", type=float, default=1.0, help="FPS to downsample to before selection")
    ap.add_argument("--max_sampled_frames", type=int, default=None,
                    help="Limit number of sampled frames (after downsampling) for quick tests")
    ap.add_argument("--no_plot", action="store_true", help="Disable score plots")
    ap.add_argument("--save_frames", action="store_true",
                    help="Also save selected keyframe images as JPEGs")
    args = ap.parse_args()

    video = args.video
    target_fps = args.target_fps
    max_sampled_frames = args.max_sampled_frames
    plot = not args.no_plot
    save_frames = args.save_frames

    video_name = Path(video).stem
    out_root = Path("runs/preselect") / video_name

    selectors: list[BasePreselector] = [
        FrameDiffPreselector(k_mad=2.5, min_spacing=6, keep_top1_per_window=150),
        SSIMPreselector(k_mad=2.5, min_spacing=6, keep_top1_per_window=150),
        MOG2Preselector(k_mad=2.5, min_spacing=6, keep_top1_per_window=150),
        FlowPreselector(k_mad=2.5, min_spacing=6, keep_top1_per_window=150),
    ]

    print(f"Running preselectors on {video} (target_fps={target_fps}) ...")

    for sel in selectors:
        run_preselector(
            video_path=video,
            selector=sel,
            target_fps=target_fps,
            max_sampled_frames=max_sampled_frames,
            plot=plot,
            save_frames=save_frames,
            out_root=out_root / sel.__class__.__name__,
        )


if __name__ == "__main__":
    main()
