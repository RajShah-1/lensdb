# src/keyframe/run_preselect.py
from __future__ import annotations
import argparse
import cv2
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt

from src.keyframe.preselect_methods import (
    FrameDiffPreselector,
    SSIMPreselector,
    MOG2Preselector,
    FlowPreselector,
)
from src.keyframe.preselect_base import BasePreselector


def run_preselector(video_path: str, selector: BasePreselector, max_frames: int | None = None, plot: bool = True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    total = 0
    selector.start()

    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keep = selector.process(frame, total)
        total += 1
        if max_frames and total >= max_frames:
            break
    cap.release()
    res = selector.finalize()
    t1 = time.time()

    print(f"\n[{selector.__class__.__name__}]")
    print(f"  Total frames:    {total}")
    print(f"  Kept keyframes:  {len(res.indices)}")
    print(f"  Compression:     {len(res.indices)/total:.4f}")
    print(f"  Runtime (sec):   {t1 - t0:.2f}")
    print(f"  FPS processed:   {total / (t1 - t0):.2f}")

    if plot:
        out_dir = Path("runs/preselect_plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 3))
        plt.plot(res.scores, lw=1.0)
        plt.scatter(res.indices, np.array(res.scores)[res.indices], color="red", s=20, label="Selected")
        plt.title(selector.__class__.__name__)
        plt.xlabel("Frame index")
        plt.ylabel("Novelty score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{selector.__class__.__name__}.png")
        plt.close()

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--max_frames", type=int, default=None, help="Limit frames for quicker testing")
    ap.add_argument("--no_plot", action="store_true", help="Disable matplotlib plotting")
    args = ap.parse_args()

    video = args.video
    plot = not args.no_plot
    max_frames = args.max_frames

    # instantiate preselectors
    selectors = [
        FrameDiffPreselector(k_mad=2.5, min_spacing=6, keep_top1_per_window=150),
        SSIMPreselector(k_mad=2.5, min_spacing=6, keep_top1_per_window=150),
        MOG2Preselector(k_mad=2.5, min_spacing=6, keep_top1_per_window=150),
        FlowPreselector(k_mad=2.5, min_spacing=6, keep_top1_per_window=150),
    ]

    print(f"Running preselectors on {video} ...\n")
    for sel in selectors:
        run_preselector(video, sel, max_frames=max_frames, plot=plot)


if __name__ == "__main__":
    main()
