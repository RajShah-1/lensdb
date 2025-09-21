import numpy as np
import torch

class ObjectDetector:
    """
    YOLOv8-based object detector for cars & people.
    """
    def __init__(self, model_name: str = "yolov8n.pt", device=None):
        from ultralytics import YOLO
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_name)
        self.model.fuse()
        self.target_classes = {"person": 0, "car": 2}  # COCO IDs

    def detect(self, frame: np.ndarray) -> dict:
        """Run detection on a single frame → return counts dict."""
        results = self.model.predict(frame, verbose=False, device=self.device)[0]
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)

        return {
            "car_count": int(np.sum(cls_ids == self.target_classes["car"])),
            "people_count": int(np.sum(cls_ids == self.target_classes["person"])),
        }
