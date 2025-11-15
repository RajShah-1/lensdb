from __future__ import annotations
import numpy as np, cv2
from typing import Optional, Dict, Any, List
from .keyframe_base import (
    BaseKeyframeSelector, KFResult, ema, robust_threshold, peak_pick, diversify_with_embeddings
)

# ---------- 1) Embedding Novelty ----------
class EmbeddingNoveltyKF(BaseKeyframeSelector):
    name = "emb_novelty"
    def __init__(self, k_mad=2.0, min_spacing=6, diversity_delta=0.12, ema_alpha=0.2):
        self.k_mad = k_mad; self.min_spacing = min_spacing
        self.diversity_delta = diversity_delta; self.ema_alpha = ema_alpha

    def select(self, embs: np.ndarray, frames=None, meta=None) -> KFResult:
        # score = 1 - cos(e_t, e_{t-1})
        sims = np.sum(embs[1:] * embs[:-1], axis=1)
        novelty = np.concatenate([[0.0], 1.0 - np.clip(sims, -1, 1)])
        score = ema(novelty, self.ema_alpha)
        thr = robust_threshold(score, self.k_mad)
        cand = peak_pick(score, thr, self.min_spacing)
        picked = diversify_with_embeddings(embs, cand, self.diversity_delta)
        return KFResult(indices=picked, score=score)

# ---------- 2) SSIM + (optional) Optical Flow ----------
class SSIMFlowKF(BaseKeyframeSelector):
    name = "ssim_flow"
    def __init__(self, w_flow=0.3, k_mad=3.0, min_spacing=12, ema_alpha=0.2):
        self.w_flow = w_flow; self.k_mad = k_mad; self.min_spacing = min_spacing; self.ema_alpha = ema_alpha

    def select(self, embs: Optional[np.ndarray], frames: List[np.ndarray] | None, meta=None) -> KFResult:
        assert frames is not None and len(frames) > 1, "SSIMFlowKF needs frames."
        ssim_vals, flow_vals = [0.0], [0.0]
        prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        # SSIM (OpenCV-contrib) fallback if unavailable
        have_quality = hasattr(cv2, "quality") and hasattr(cv2.quality, "QualitySSIM_compute")

        for t in range(1, len(frames)):
            g = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
            if have_quality:
                ssim = float(cv2.quality.QualitySSIM_compute(prev, g)[0][0])
                ssim_vals.append(1.0 - ssim)
            else:
                # fallback: MSE-based structural change proxy
                diff = (g.astype(np.float32) - prev.astype(np.float32)) / 255.0
                ssim_vals.append(float(np.mean(diff*diff)))

            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.1, 0)
            flow_mag = np.mean(np.linalg.norm(flow, axis=-1))
            flow_vals.append(float(flow_mag))
            prev = g

        s1 = ema(np.array(ssim_vals, dtype=float), self.ema_alpha)
        s2 = ema(np.array(flow_vals, dtype=float), self.ema_alpha)
        # normalize flow to 0..1
        s2 = (s2 - s2.min()) / (s2.ptp() + 1e-8)

        score = (1 - self.w_flow) * s1 + self.w_flow * s2
        thr = robust_threshold(score, self.k_mad)
        cand = peak_pick(score, thr, self.min_spacing)

        # If embeddings provided, diversify; otherwise return peaks
        picked = diversify_with_embeddings(embs, cand, 0.12) if embs is not None else cand
        return KFResult(indices=picked, score=score)

# ---------- 3) Windowed k-Center (coverage control) ----------
class WindowKCenterKF(BaseKeyframeSelector):
    name = "kcenter"
    def __init__(self, window=150, k=3, delta=0.12):
        self.window = window; self.k = k; self.delta = delta

    def select(self, embs: np.ndarray, frames=None, meta=None) -> KFResult:
        T = len(embs); idxs: List[int] = []; score = np.zeros(T, dtype=float)
        for a in range(0, T, self.window):
            b = min(T, a + self.window)
            X = embs[a:b]

            if len(X) == 0:
                continue
            centers = [0]
            for _ in range(1, min(self.k, len(X))):
                sims = X @ X[centers].T                  # cosine
                dist = 1.0 - np.max(sims, axis=1)        # farthest-first
                centers.append(int(np.argmax(dist)))

            win_idx = [a + c for c in centers]
            idxs.extend(sorted(win_idx))
            score[win_idx] = 1.0

        idxs = diversify_with_embeddings(embs, sorted(idxs), self.delta)
        return KFResult(indices=sorted(idxs), score=score)
