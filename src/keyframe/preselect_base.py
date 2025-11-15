# src/keyframe/preselect_base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class PreselectResult:
    """
    Result of a pre-embedding keyframe selection pass.

    Attributes
    ----------
    indices : List[int]
        Zero-based indices (in the sampled stream) of frames to KEEP.
    scores : np.ndarray
        Per-frame novelty/importance scores (shape: [T,]) produced by the preselector
        for debugging/plots. If the method is purely binary and doesn't compute a
        score, return an array of zeros with length T.
    """
    indices: List[int]
    scores: np.ndarray


class BasePreselector(ABC):
    """
    Streaming preselector interface for picking keyframes directly from decoded video.

    Usage pattern
    -------------
    pre = YourPreselector(...)
    pre.start()
    for idx, frame in enumerate(video_frames):   # frame: np.ndarray (H, W, 3) BGR
        keep = pre.process(frame, idx)           # decide KEEP/skip online (O(1))
        if keep:
            # send frame to embedder, save frame, etc.
            pass
    result = pre.finalize()                      # flush window heuristics, return indices/scores

    Notes
    -----
    - Methods should be fast, CPU-friendly, and not require deep models.
    - Implementations should internally handle:
        * optional smoothing (e.g., EMA),
        * adaptive thresholds (e.g., median + MAD),
        * min-spacing debouncing,
        * optional "top-1 per window" fallback to guarantee local coverage.
    """

    @abstractmethod
    def start(self) -> None:
        """Initialize/reset internal state before streaming frames."""
        raise NotImplementedError

    @abstractmethod
    def process(self, frame: np.ndarray, idx: int) -> bool:
        """
        Consume one frame and return True if it should be KEPT, else False.

        Parameters
        ----------
        frame : np.ndarray
            BGR image array (H, W, 3), dtype uint8.
        idx : int
            Zero-based index within the sampled frame stream (monotonic).

        Returns
        -------
        bool
            True if this frame is selected as a keyframe, otherwise False.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> PreselectResult:
        """
        Finalize selection after the stream ends (flush window-best, etc.).

        Returns
        -------
        PreselectResult
            Selected indices and the per-frame scores used by the method.
        """
        raise NotImplementedError
