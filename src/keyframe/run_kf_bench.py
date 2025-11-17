from pathlib import Path
import os, numpy as np, cv2
from src.keyframe.keyframe_selectors import EmbeddingNoveltyKF, SSIMFlowKF, WindowKCenterKF
from src.keyframe.kf_eval import evaluate_selector, summarize_to_csv

def load_frames(frames_dir: str | None):
    if not frames_dir: return None
    fnames = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg",".png"))])
    return [cv2.imread(str(Path(frames_dir) / f)) for f in fnames]

def run(embs_path: str, frames_dir: str | None, out_dir: str, text_vecs_path: str | None = None):
    embs = np.load(embs_path)   # e.g., runs/embeddings/embds.npy (L2-normalized)
    frames = load_frames(frames_dir)
    text_vecs = np.load(text_vecs_path) if text_vecs_path else None

    methods = {
        "emb_novelty": EmbeddingNoveltyKF(k_mad=3.0, min_spacing=12, diversity_delta=0.12, ema_alpha=0.2),
        "ssim_flow":  SSIMFlowKF(w_flow=0.3, k_mad=3.0, min_spacing=12, ema_alpha=0.2),
        "kcenter_w150_k3": WindowKCenterKF(window=150, k=3, delta=0.12),
    }

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, selector in methods.items():
        m = evaluate_selector(name, selector, embs, frames=frames, text_vecs=text_vecs, out_dir=out / name)
        rows.append(m)
        print(name, "=>", m)

    summarize_to_csv(rows, out / "summary.csv")
    print("Saved:", out / "summary.csv")

if __name__ == "__main__":
    # Update these to match where your pipeline saves things:
    run(
        embs_path="data/VIRAT/VIRAT_S_000001/embds.npy",
        frames_dir=None,         # or None if you didn't save frames
        out_dir="runs/kf_bench",
        text_vecs_path=None               # optional: npy of CLIP text embeddings (Q, D)
    )
