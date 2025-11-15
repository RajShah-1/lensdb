# src/keyframe/run_kf_bench_embeddings.py
from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import numpy as np

from src.keyframe.keyframe_selectors import EmbeddingNoveltyKF, WindowKCenterKF
from src.keyframe.kf_eval import evaluate_selector, summarize_to_csv

def find_sequences(patterns):
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p, recursive=True))
    # Keep only embds.npy
    paths = [p for p in paths if p.endswith("embds.npy")]
    return sorted(paths)

def run_one(embs_path: str, out_dir: Path):
    embs = np.load(embs_path)  # (T, D)
    methods = {
        "emb_novelty": EmbeddingNoveltyKF(k_mad=3.0, min_spacing=12, diversity_delta=0.12, ema_alpha=0.2),
        "kcenter_w150_k3": WindowKCenterKF(window=150, k=3, delta=0.12),
    }
    rows = []
    for name, selector in methods.items():
        m = evaluate_selector(name, selector, embs, frames=None, text_vecs=None, out_dir=out_dir / name)
        m["embs_path"] = embs_path
        rows.append(m)
        print(f"{Path(embs_path).parent.name}:{name} =>", m)
    summarize_to_csv(rows, out_dir / "summary.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", default=["data/**/embds.npy"],
                    help="Glob(s) to find embds.npy files (e.g., data/VIRAT/*/embds.npy)")
    ap.add_argument("--out_root", type=str, default="runs/kf_bench_embeddings",
                    help="Where to write per-sequence results")
    args = ap.parse_args()

    seqs = find_sequences(args.glob)
    if not seqs:
        print("No embds.npy found. Check your --glob.")
        return

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for embs_path in seqs:
        seq_name = Path(embs_path).parent.name
        out_dir = out_root / seq_name
        out_dir.mkdir(parents=True, exist_ok=True)
        run_one(embs_path, out_dir)
        # collect into all_rows
        import csv
        with open(out_dir / "summary.csv") as f:
            import csv as _csv
            r = list(_csv.DictReader(f))
            for row in r:
                row["sequence"] = seq_name
                all_rows.append(row)

    # write aggregate
    from src.keyframe.kf_eval import summarize_to_csv
    summarize_to_csv(all_rows, out_root / "ALL_summary.csv")
    print("Saved:", out_root / "ALL_summary.csv")

if __name__ == "__main__":
    main()
