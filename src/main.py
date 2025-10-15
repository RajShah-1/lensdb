from pathlib import Path

from pipeline.pipeline import VideoPipeline
from src.embeddings.embedder import CLIPEmbedder, CLIP_VIT_B32
from src.models.model_configs import MEDIUM, SMALL
from src.pipeline.detection_pipeline import DetectionPipeline
from src.detectors.object_detector import ObjectDetector
from src.training.train_pipeline import finetune_on_virat, pretrain_on_coco


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

def run_detection_on_dir(videos_dir: str, model_name: str, annotated: bool):
    videos = sorted(Path(videos_dir).glob("*.mp4"))
    print(f"Found {len(videos)} videos in {videos_dir}")

    detector = ObjectDetector(model_name=model_name, save_annotated=annotated)

    for vid in videos:
        print(f"\n=== Processing {vid.name} ===")
        out_dir = Path("data/VIRAT") / vid.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline = DetectionPipeline(str(vid), detector, out_dir=out_dir)
        pipeline.run(save=True)

def gen_embeddings_for_dir(videos_dir: str, embedder_config=None):
    """
    Generate embeddings for all videos in a directory.
    """
    videos = sorted(Path(videos_dir).glob("*.mp4"))
    print(f"Found {len(videos)} videos in {videos_dir}")
    
    if embedder_config is None:
        embedder_config = CLIP_VIT_B32
    
    embedder = CLIPEmbedder(embedder_config)
    
    for vid in videos:
        print(f"\n=== Processing {vid.name} ===")
        out_dir = Path("data/VIRAT") / vid.stem / "embeddings"
        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline = VideoPipeline(str(vid), embedder, out_dir=out_dir)
        pipeline.run(save=True)

if __name__ == "__main__":
    # gen_embeddings()
    # run_detection()
    # run_detection_on_dir("/storage/ice1/8/3/rshah647/VIRATGround/videos_original", 
    #                      "yolo11x.pt", 
    #                      False)
    # gen_embeddings_for_dir("/storage/ice1/8/3/rshah647/VIRATGround/videos_original", 
    #                        CLIP_VIT_B32)

    # pretrain_on_coco(
    #     coco_dir="data/coco",  # root dir of coco (with train2017, val2017 subdirs)
    #     target="car",
    #     model_config=SMALL
    # )

    finetune_on_virat(
        data_dir="data/VIRAT",        # root of all video folders
        target="car",
        pretrained_checkpoint=None,   # None = from scratch
        train_ratio=0.4, # 40% train / 60% test
        model_config=MEDIUM
    )



