from __future__ import annotations
import numpy as np, cv2
from typing import List
from .preselect_base import BasePreselector, PreselectResult

def ema(x_prev: float, x: float, a: float) -> float:
    return a * x + (1.0 - a) * x_prev

def robust_params(vals: List[float]):
    arr = np.array(vals, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) + 1e-8
    return med, mad

def adaptive_gate(x: float, med: float, mad: float, k: float) -> bool:
    return x > (med + 1.4826 * k * mad)

def enforce_spacing(keep_idx: List[int], idx: int, min_spacing: int) -> bool:
    return (len(keep_idx) == 0) or (idx - keep_idx[-1] >= min_spacing)

# ---------------- 1) Frame difference (fast, robust-ish) ----------------
class FrameDiffPreselector(BasePreselector):
    """
    Novelty = mean absolute diff on grayscale (or small resize). Very fast CPU.
    Good first-pass filter for CCTV.
    """
    def __init__(self, k_mad: float = 2.5, min_spacing: int = 6, ema_alpha: float = 0.2,
                 downscale: int = 2, keep_top1_per_window: int | None = 150):
        self.k_mad, self.min_spacing, self.ema_alpha = k_mad, min_spacing, ema_alpha
        self.downscale, self.keep_top1_per_window = downscale, keep_top1_per_window

    def start(self):
        self.prev = None
        self.idx = -1
        self.scores: List[float] = []
        self.keep: List[int] = []
        self.ema_score = 0.0
        self.window_best = (-1, -1.0)  # (idx, score)

    def _score(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            g = cv2.resize(g, (g.shape[1]//self.downscale, g.shape[0]//self.downscale))
        if self.prev is None:
            self.prev = g
            return 0.0
        diff = cv2.absdiff(self.prev, g)
        s = float(np.mean(diff)) / 255.0
        self.prev = g
        return s

    def process(self, frame, idx: int) -> bool:
        self.idx = idx
        raw = self._score(frame)
        self.ema_score = ema(self.ema_score, raw, self.ema_alpha) if idx > 0 else raw
        self.scores.append(self.ema_score)

        # update window best
        if self.keep_top1_per_window:
            if self.window_best[1] < self.ema_score:
                self.window_best = (idx, self.ema_score)
            if (idx + 1) % self.keep_top1_per_window == 0:
                # enforce at least 1 per window
                if len(self.keep) == 0 or self.keep[-1] != self.window_best[0]:
                    if enforce_spacing(self.keep, self.window_best[0], self.min_spacing):
                        self.keep.append(self.window_best[0])
                self.window_best = (-1, -1.0)

        # online adaptive threshold with warmup
        # lightweight: use a short buffer for robust stats
        if len(self.scores) < 30:
            return False
        med, mad = robust_params(self.scores[-30:])  # recent-only for adaptivity
        is_peak = adaptive_gate(self.ema_score, med, mad, self.k_mad)

        if is_peak and enforce_spacing(self.keep, idx, self.min_spacing):
            # avoid double adding same index as window-best
            if not self.keep or self.keep[-1] != idx:
                self.keep.append(idx)
            return True
        return False

    def finalize(self) -> PreselectResult:
        # flush partial window
        if self.keep_top1_per_window and self.window_best[0] >= 0:
            if enforce_spacing(self.keep, self.window_best[0], self.min_spacing):
                self.keep.append(self.window_best[0])
        scores = np.array(self.scores, dtype=float)
        self.keep = sorted(set(self.keep))
        return PreselectResult(indices=self.keep, scores=scores)

# ---------------- 2) SSIM drop (uses opencv-contrib if available) ----------------
class SSIMPreselector(BasePreselector):
    def __init__(self, k_mad=2.5, min_spacing=6, ema_alpha=0.2, keep_top1_per_window: int | None = 150):
        self.k_mad, self.min_spacing, self.ema_alpha = k_mad, min_spacing, ema_alpha
        self.keep_top1_per_window = keep_top1_per_window

    def start(self):
        self.prev_g = None
        self.idx = -1
        self.scores: List[float] = []
        self.keep: List[int] = []
        self.ema_score = 0.0
        self.have_quality = hasattr(cv2, "quality") and hasattr(cv2.quality, "QualitySSIM_compute")
        self.window_best = (-1, -1.0)

    def process(self, frame, idx: int) -> bool:
        self.idx = idx
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_g is None:
            self.prev_g = g
            self.scores.append(0.0)
            return False

        if self.have_quality:
            ssim = float(cv2.quality.QualitySSIM_compute(self.prev_g, g)[0][0])
            novelty = 1.0 - ssim
        else:
            diff = (g.astype(np.float32) - self.prev_g.astype(np.float32)) / 255.0
            novelty = float(np.mean(np.abs(diff)))  # proxy if SSIM unavailable
        self.prev_g = g

        self.ema_score = ema(self.ema_score, novelty, self.ema_alpha)
        self.scores.append(self.ema_score)

        if self.keep_top1_per_window:
            if self.window_best[1] < self.ema_score:
                self.window_best = (idx, self.ema_score)
            if (idx + 1) % self.keep_top1_per_window == 0:
                if len(self.keep) == 0 or self.keep[-1] != self.window_best[0]:
                    if idx - (self.keep[-1] if self.keep else -1e9) >= self.min_spacing:
                        self.keep.append(self.window_best[0])
                self.window_best = (-1, -1.0)

        if len(self.scores) < 30:
            return False
        med, mad = robust_params(self.scores[-30:])
        is_peak = adaptive_gate(self.ema_score, med, mad, self.k_mad)
        if is_peak and enforce_spacing(self.keep, idx, self.min_spacing):
            if not self.keep or self.keep[-1] != idx:
                self.keep.append(idx)
            return True
        return False

    def finalize(self) -> PreselectResult:
        if self.keep_top1_per_window and self.window_best[0] >= 0:
            if enforce_spacing(self.keep, self.window_best[0], self.min_spacing):
                self.keep.append(self.window_best[0])
        return PreselectResult(indices=sorted(set(self.keep)), scores=np.array(self.scores, dtype=float))

# ---------------- 3) Background subtraction area ----------------
class MOG2Preselector(BasePreselector):
    def __init__(self, k_mad=2.5, min_spacing=6, ema_alpha=0.2, keep_top1_per_window: int | None = 150):
        self.k_mad, self.min_spacing, self.ema_alpha = k_mad, min_spacing, ema_alpha
        self.keep_top1_per_window = keep_top1_per_window

    def start(self):
        self.bgs = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)
        self.idx = -1
        self.scores: List[float] = []
        self.keep: List[int] = []
        self.ema_score = 0.0
        self.window_best = (-1, -1.0)

    def process(self, frame, idx: int) -> bool:
        self.idx = idx
        fg = self.bgs.apply(frame)
        # binarize, ignore shadows (value 127)
        moving = (fg == 255).astype(np.float32)
        ratio = float(np.mean(moving))  # 0..1
        self.ema_score = ema(self.ema_score, ratio, self.ema_alpha)
        self.scores.append(self.ema_score)

        if self.keep_top1_per_window:
            if self.window_best[1] < self.ema_score:
                self.window_best = (idx, self.ema_score)
            if (idx + 1) % self.keep_top1_per_window == 0:
                if enforce_spacing(self.keep, self.window_best[0], self.min_spacing):
                    self.keep.append(self.window_best[0])
                self.window_best = (-1, -1.0)

        if len(self.scores) < 60:  # warm-up for bgs
            return False
        med, mad = robust_params(self.scores[-60:])
        if adaptive_gate(self.ema_score, med, mad, self.k_mad) and enforce_spacing(self.keep, idx, self.min_spacing):
            if not self.keep or self.keep[-1] != idx:
                self.keep.append(idx)
            return True
        return False

    def finalize(self) -> PreselectResult:
        if self.keep_top1_per_window and self.window_best[0] >= 0:
            if enforce_spacing(self.keep, self.window_best[0], self.min_spacing):
                self.keep.append(self.window_best[0])
        return PreselectResult(indices=sorted(set(self.keep)), scores=np.array(self.scores, dtype=float))

# ---------------- 4) Lightweight optical flow magnitude ----------------
class FlowPreselector(BasePreselector):
    def __init__(self, k_mad=2.5, min_spacing=6, ema_alpha=0.2, pyr_scale=0.5, win_size=15, keep_top1_per_window: int | None = 150):
        self.k_mad, self.min_spacing, self.ema_alpha = k_mad, min_spacing, ema_alpha
        self.pyr_scale, self.win_size = pyr_scale, win_size
        self.keep_top1_per_window = keep_top1_per_window

    def start(self):
        self.prev = None
        self.idx = -1
        self.scores: List[float] = []
        self.keep: List[int] = []
        self.ema_score = 0.0
        self.window_best = (-1, -1.0)

    def process(self, frame, idx: int) -> bool:
        self.idx = idx
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev is None:
            self.prev = g; self.scores.append(0.0); return False
        flow = cv2.calcOpticalFlowFarneback(self.prev, g, None, self.pyr_scale, 3, self.win_size, 3, 5, 1.1, 0)
        mag = np.mean(np.linalg.norm(flow, axis=-1))
        self.prev = g

        self.ema_score = ema(self.ema_score, mag, self.ema_alpha)
        self.scores.append(self.ema_score)

        if self.keep_top1_per_window:
            if self.window_best[1] < self.ema_score:
                self.window_best = (idx, self.ema_score)
            if (idx + 1) % self.keep_top1_per_window == 0:
                if enforce_spacing(self.keep, self.window_best[0], self.min_spacing):
                    self.keep.append(self.window_best[0])
                self.window_best = (-1, -1.0)

        if len(self.scores) < 30:
            return False
        med, mad = robust_params(self.scores[-30:])
        if adaptive_gate(self.ema_score, med, mad, self.k_mad) and enforce_spacing(self.keep, idx, self.min_spacing):
            if not self.keep or self.keep[-1] != idx:
                self.keep.append(idx)
            return True
        return False

    def finalize(self) -> PreselectResult:
        if self.keep_top1_per_window and self.window_best[0] >= 0:
            if enforce_spacing(self.keep, self.window_best[0], self.min_spacing):
                self.keep.append(self.window_best[0])
        return PreselectResult(indices=sorted(set(self.keep)), scores=np.array(self.scores, dtype=float))
