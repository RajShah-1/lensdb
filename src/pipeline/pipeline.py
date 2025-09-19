from .video_reader import VideoReader
from .frame_sampler import FrameSampler
from src.embeddings.embedder import Embedder


class VideoPipeline:
    """End-to-end pipeline: read video → sample frames → embed."""

    def __init__(self, video_path: str, sampler: FrameSampler, embedder: Embedder):
        self.video_path = video_path
        self.sampler = sampler
        self.embedder = embedder

    def run(self):
        reader = VideoReader(self.video_path)
        embeddings = []
        for frame in self.sampler.sample(reader):
            emb = self.embedder.embed(frame)
            embeddings.append(emb)
        return embeddings
