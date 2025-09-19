from abc import ABC, abstractmethod
import numpy as np


class Embedder(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed(self, frame) -> np.ndarray:
        """Convert a frame (numpy array) into an embedding vector."""
        pass


class DummyEmbedder(Embedder):
    """Example embedder: returns mean pixel values as embedding."""

    def embed(self, frame) -> np.ndarray:
        return frame.mean(axis=(0, 1))  # RGB mean → (3,)
