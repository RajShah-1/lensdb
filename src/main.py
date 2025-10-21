from pathlib import Path

from pipeline.pipeline import VideoPipeline
from src.embeddings.embedder import CLIPEmbedder, CLIP_VIT_B32
from src.models.model_configs import MEDIUM, SMALL
from src.pipeline.detection_pipeline import DetectionPipeline
from src.detectors.object_detector import ObjectDetector
from src.training.train_pipeline import finetune_on_virat, pretrain_on_coco
from src.indexing.faiss_index import FAISSIndex
from src.query.semantic_query import SemanticQueryPipeline, simple_query


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

def build_index(data_dir="data/VIRAT"):
    """Build FAISS index for semantic queries."""
    index = FAISSIndex(data_dir)
    index.build()

def run_semantic_query():
    """
    Run a semantic query using FAISS + MLP pipeline.
    
    Example: Find frames with cars (count >= 2)
    """
    results = simple_query(
        data_dir="data/VIRAT",
        checkpoint_path="models/checkpoints/car_virat_finetuned.pth",
        model_config=MEDIUM,
        text_query="car",
        min_count=2.0,              # Only frames with 2+ cars
        similarity_threshold=0.2     # Cosine similarity threshold
    )
    return results

def run_advanced_query():
    """
    More advanced query with custom predicates.
    """
    pipeline = SemanticQueryPipeline(
        data_dir="data/VIRAT",
        checkpoint_path="models/checkpoints/car_virat_finetuned.pth",
        model_config=MEDIUM,
        prefilter_threshold=0.25,
        top_k=1000  # Max 1000 frames
    )
    
    # Custom predicate: between 1 and 5 cars
    results = pipeline.query(
        text_query="car",
        count_predicate=lambda count: 1 <= count <= 5
    )
    
    # Save results
    pipeline.query_and_save(
        text_query="car",
        count_predicate=lambda count: count >= 2,
        output_dir="data/_query_results/cars_2plus"
    )
    
    return results


if __name__ == "__main__":
    # ========================================
    # STEP 1: Generate embeddings for videos
    # ========================================
    # gen_embeddings()  # Single video
    # gen_embeddings_for_dir("/path/to/videos", CLIP_VIT_B32)  # Batch
    
    # ========================================
    # STEP 2: Generate ground truth counts (optional, for training)
    # ========================================
    # run_detection()  # Single video
    # run_detection_on_dir("/path/to/videos", "yolo11x.pt", False)  # Batch
    
    # ========================================
    # STEP 3: Train count predictor
    # ========================================
    # Option A: Pretrain on COCO then finetune
    # pretrain_on_coco(
    #     coco_dir="data/coco",
    #     target="car",
    #     model_config=SMALL
    # )
    
    # Option B: Train directly on VIRAT (recommended if you have VIRAT data)
    # finetune_on_virat(
    #     data_dir="data/VIRAT",
    #     target="car",
    #     pretrained_checkpoint=None,   # None = from scratch
    #     train_ratio=0.4,              # 40% train / 60% test
    #     model_config=MEDIUM
    # )
    
    # ========================================
    # STEP 4: Build FAISS index
    # ========================================
    # build_index("data/VIRAT")
    
    # ========================================
    # STEP 5: Run semantic queries
    # ========================================
    # Simple query
    # run_semantic_query()
    
    # Advanced query with custom predicates
    # run_advanced_query()
    
    pass



