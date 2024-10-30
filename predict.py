from uuid import uuid4
import os
from typing import Any

from cog import Input, Path, BasePredictor
import supervision as sv
from ultralytics import YOLO
import numpy as np
from loguru import logger
import cv2


class Predictor(BasePredictor):
    def setup(self):
        self.model = YOLO("yolo11n.pt")
        logger.info("Model loaded")

    def predict(  # pyright: ignore
            self,
            image: Path = Input(description="Grayscale input image"),
            confidence_threshold: float = Input(
                description="Confidence threshold",
                ge=0.2,
                le=0.95,
                default=0.5,
            ),
    ) -> Path:
        # Load image
        img = cv2.imread(str(image))
        results = self.model(img, conf=confidence_threshold)
        detections = sv.Detections.from_ultralytics(results[0])
        # # Initialize a box annotator
        box_annotator = sv.BoxAnnotator()
        img = box_annotator.annotate(scene=img, detections=detections)
        output_path = os.path.join("/tmp", f"out-{str(uuid4())}.jpg")
        cv2.imwrite(output_path, img)
        return Path(output_path)
