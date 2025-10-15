import csv
from pathlib import Path

import cv2

from src.detectors.object_detector import ObjectDetector
from src.pipeline.video_reader import VideoReader

BATCH_SIZE = 16
SAVE_INTERVAL = 100  # Save every Nth frame

class DetectionPipeline:
    """Pipeline: read video → run object detection → save counts + frames."""

    def __init__(self, video_path: str, detector: ObjectDetector, out_dir: str | None = None):
        self.video_path = Path(video_path)
        self.detector = detector
        self.out_dir = Path(out_dir) if out_dir else Path("data/VIRAT") / self.video_path.stem
        self.frames_dir = self.out_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = SAVE_INTERVAL  # Save every Nth frame

    def run(self, save: bool = True):
        reader = VideoReader(str(self.video_path), "time",
                             0,
                             1,
                             BATCH_SIZE)
        counts_file = self.out_dir / "counts.csv"

        with open(counts_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "car_count", "people_count", "car_bboxes", "people_bboxes"])

            frame_counter = 0
            total_processed = 0

            for batch in reader:
                for frame in batch:
                    frame_path = None
                    if save and frame_counter % self.save_interval == 0:
                        frame_path = self.frames_dir / f"frame_{frame_counter:05d}.jpg"
                    
                    det_counts = self.detector.detect(frame, frame_path=frame_path)
                    writer.writerow([
                        frame_counter,
                        det_counts["car_count"],
                        det_counts["people_count"],
                        det_counts["car_bboxes"],
                        det_counts["people_bboxes"],
                    ])

                    if save and frame_path is not None:
                        cv2.imwrite(str(frame_path), frame)

                    frame_counter += 1
                total_processed += len(batch)

                if total_processed % 100 == 0:
                    print(f"Processed {total_processed} frames")

        print(f"Saved detection counts → {counts_file}")
        print(f"Saved frames under → {self.frames_dir}")
