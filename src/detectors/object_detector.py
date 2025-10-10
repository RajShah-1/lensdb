import numpy as np
import torch
import cv2
from pathlib import Path
import json

class ObjectDetector:
    """
    YOLOv8-based detector for cars & people.
    Saves optional annotated frames and bbox data.
    """
    def __init__(self, model_name, device=None, save_annotated=False):
        from ultralytics import YOLO
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_name)
        self.model.fuse()
        self.target_classes = {"person": 0, "car": 2}
        self.save_annotated = save_annotated

    def detect(self, frame: np.ndarray, frame_path: Path | None = None) -> dict:
        results = self.model.predict(frame, verbose=False, device=self.device)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)

        car_boxes = boxes[cls_ids == self.target_classes["car"]]
        people_boxes = boxes[cls_ids == self.target_classes["person"]]

        # Optionally draw boxes
        if self.save_annotated and frame_path is not None:
            annotated = frame.copy()
            for (x1, y1, x2, y2), cid in zip(boxes, cls_ids):
                color = (0, 255, 0) if cid == 2 else (255, 0, 0)
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.imwrite(str(frame_path), annotated)

        # Return counts and bbox JSON for CSV
        return {
            "car_count": len(car_boxes),
            "people_count": len(people_boxes),
            "car_bboxes": json.dumps(car_boxes.tolist()),
            "people_bboxes": json.dumps(people_boxes.tolist()),
        }
