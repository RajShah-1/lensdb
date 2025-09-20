from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from src.utils import get_best_device, resize_image

# Input resolution for embedders
CLIP_INPUT_SIZE = (224, 224)  # (width, height)

class Embedder(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed(self, frames: list[np.ndarray]) -> np.ndarray:
        """Convert a frame (numpy array) into an embedding vector."""
        pass

class DummyEmbedder(Embedder):
    """Example embedder: returns mean pixel values as embedding."""
    def embed(self, frames) -> np.ndarray:
        resized = [
            resize_image(f, CLIP_INPUT_SIZE)
            for f in frames
        ]
        arr = np.stack(resized, axis=0)
        embs = arr.mean(axis=(1, 2))  # mean over H,W → (batch, 3)
        return embs

@dataclass
class EmbedderConfig:
    processor_name: str
    model_name: str

CLIP_VIT_B32 = EmbedderConfig("openai/clip-vit-base-patch32",
                              "openai/clip-vit-base-patch32")
MOBILE_CLIP_VIT_PATCH16 = EmbedderConfig("apple/mobileclip-vit-base-patch16",
                                         "apple/mobileclip-vit-base-patch16")
class CLIPEmbedder(Embedder):
    """CLIP based Embedder (image -> semantic vector)."""
    def __init__(self, cfg : EmbedderConfig):
        self.device = get_best_device()
        from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
        self.processor = AutoProcessor.from_pretrained(cfg.processor_name)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(cfg.model_name)
        self.model = self.model.to(self.device, dtype=torch.float32)

    def embed(self, frames) -> np.ndarray:
        images = [resize_image(f, CLIP_INPUT_SIZE) for f in frames]
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        embs = outputs.cpu().numpy()
        return embs / np.linalg.norm(embs, axis=1, keepdims=True)
