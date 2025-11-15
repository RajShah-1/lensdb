from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass
class KFResult:
    """Output of a keyframe selector."""
    indices: List[int]           # selected frame indices (sorted, unique)
    score: np.ndarray            # per-frame novelty/score used by the method (T,)

class BaseKeyframeSelector(ABC):
    """Base class for all keyframe selectors."""
    name: str = "base"

    @abstractmethod
    def select(
        self,
        embs: np.ndarray,                           # (T, D) L2-normalized embeddings (optional for some methods)
        frames: Optional[List[np.ndarray]] = None,  # list of BGR frames if needed
        meta: Optional[Dict[str, Any]] = None,      # timestamps, counts, precomputed flow, etc.
    ) -> KFResult:
        raise NotImplementedError

# ---------- Shared utilities ----------
def ema(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    if len(x) == 0: return y
    y[0] = float(x[0])
    for t in range(1, len(x)):
        y[t] = alpha * float(x[t]) + (1.0 - alpha) * y[t - 1]
    return y

def robust_threshold(x: np.ndarray, k: float = 3.0) -> float:
    if len(x) == 0: return float("inf")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-8
    return med + 1.4826 * k * mad

def peak_pick(score: np.ndarray, thr: float, min_spacing: int = 10) -> List[int]:
    T = len(score)
    peaks: List[int] = []
    last = -10**9
    for t in range(1, T - 1):
        if score[t] > thr and score[t] >= score[t - 1] and score[t] >= score[t + 1]:
            if t - last >= min_spacing:
                peaks.append(t); last = t
    return peaks

def diversify_with_embeddings(
    embs: np.ndarray, cand_indices: List[int], diversity_delta: float = 0.12
) -> List[int]:
    if len(cand_indices) <= 1: return cand_indices
    chosen: List[int] = []
    for t in cand_indices:
        if not chosen:
            chosen.append(t); continue
        sims = embs[t] @ embs[chosen].T  # cosine sim (assumes embs are L2-normalized)
        if float(np.max(sims)) < (1.0 - diversity_delta):
            chosen.append(t)
    return chosen
