from pipeline.pipeline import VideoPipeline
from src.embeddings.embedder import CLIPEmbedder, CLIP_VIT_B32

def main():
    video_path = "videos/demo.mp4"
    embedder = CLIPEmbedder(CLIP_VIT_B32)
    pipeline = VideoPipeline(video_path, embedder)
    embeddings = pipeline.run(save=True)
    print(f"Extracted {len(embeddings)} embeddings")

if __name__ == "__main__":
    main()
