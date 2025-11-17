"""Utility modules for LensDB."""

from typing import Tuple
import cv2
import numpy as np
import torch
from PIL import Image


def get_best_device() -> torch.device:
    """
    Selects the best available device in order:
    1. CUDA GPU
    2. Apple MPS (Metal Performance Shaders)
    3. CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def resize_image(frame: np.ndarray, input_size: Tuple[int, int]) -> Image:
    resized = cv2.resize(frame, input_size, interpolation=cv2.INTER_LINEAR)
    image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    return image


__all__ = [
    'get_best_device',
    'resize_image'
]

