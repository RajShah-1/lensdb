from pipeline.frame_sampler import FrameSampler
from src.embeddings.embedder import DummyEmbedder
from pipeline.pipeline import VideoPipeline


def main():
    video_path = "videos/demo.mp4"

    sampler = FrameSampler(every_n_frames=30)
    embedder = DummyEmbedder()
    pipeline = VideoPipeline(video_path, sampler, embedder)

    embeddings = pipeline.run()
    print(f"Extracted {len(embeddings)} embeddings")
    print(embeddings[:5])

if __name__ == "__main__":
    main()
