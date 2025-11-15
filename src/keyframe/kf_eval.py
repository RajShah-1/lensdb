# src/keyframe/kf_eval.py
from __future__ import annotations
import time, csv
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from .keyframe_base import KFResult, BaseKeyframeSelector

def compression_rate(T: int, K: int) -> float:
    return K / max(T, 1)

def avg_gap(indices: List[int]) -> float:
    if len(indices) < 2: return float('inf')
    diffs = np.diff(sorted(indices))
    return float(np.mean(diffs))

def _pairwise_mean_cos(X: np.ndarray) -> float:
    n = len(X)
    if n < 2: return 0.0
    S = X @ X.T
    iu = np.triu_indices(n, 1)
    return float(np.mean(S[iu]))

def redundancy_cos(embs: np.ndarray, indices: List[int]) -> float:
    """Mean pairwise cosine similarity among selected frames (lower is better)."""
    if len(indices) < 2: return 0.0
    return _pairwise_mean_cos(embs[indices])

def coverage_stats(embs_all: np.ndarray, indices: List[int]) -> Dict[str, float]:
    """
    Coverage of the full sequence by the selected set:
      For each frame i, find max cosine similarity to any selected frame.
      Report mean / median / 5th percentile and worst-case (min).
      Higher is better for coverage metrics.
    """
    if len(indices) == 0:
        return dict(coverage_mean=0.0, coverage_median=0.0, coverage_p5=0.0, coverage_min=0.0)
    X = embs_all  # (T, D), L2-normalized
    S = X @ X[indices].T          # (T, K) cosine sims to selected set
    nearest = np.max(S, axis=1)   # (T,)
    return dict(
        coverage_mean=float(np.mean(nearest)),
        coverage_median=float(np.median(nearest)),
        coverage_p5=float(np.percentile(nearest, 5)),
        coverage_min=float(np.min(nearest)),
    )

def evaluate_selector(
    name: str,
    selector: BaseKeyframeSelector,
    embs: np.ndarray,
    frames=None,
    text_vecs: np.ndarray | None = None,  # unused in embedding-only mode
    out_dir: Path | None = None
) -> Dict[str, float]:
    # Ensure L2 norm (in case it wasn't saved normalized)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs = embs / norms

    T = len(embs)
    t0 = time.time()
    res: KFResult = selector.select(embs, frames=None, meta=None)
    dt = time.time() - t0
    K = len(res.indices)

    metrics = {
        "method": name,
        "time_sec": dt,
        "T_frames": T,
        "K_keyframes": K,
        "compression": compression_rate(T, K),
        "avg_gap": avg_gap(res.indices),
        "redundancy_cos": redundancy_cos(embs, res.indices),  # lower is better
    }
    metrics.update(coverage_stats(embs, res.indices))          # higher is better
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{name}_idx.npy", np.array(res.indices))
        np.save(out_dir / f"{name}_score.npy", res.score)
    return metrics

def summarize_to_csv(rows: List[Dict[str, float]], csv_path: Path):
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
