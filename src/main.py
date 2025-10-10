from pipeline.pipeline import VideoPipeline
from src.embeddings.embedder import CLIPEmbedder, CLIP_VIT_B32
from src.pipeline.detection_pipeline import DetectionPipeline
from src.detectors.object_detector import ObjectDetector


def gen_embeddings():
    video_path = "videos/demo.mp4"
    embedder = CLIPEmbedder(CLIP_VIT_B32)
    pipeline = VideoPipeline(video_path, embedder)
    embeddings = pipeline.run(save=True)
    print(f"Extracted {len(embeddings)} embeddings")

def run_detection():
    video_path = "videos/demo.mp4"
    detector = ObjectDetector(model_name="yolov8l")
    pipeline = DetectionPipeline(video_path, detector)
    counts = pipeline.run(save=True)

if __name__ == "__main__":
    # gen_embeddings()
    run_detection()
