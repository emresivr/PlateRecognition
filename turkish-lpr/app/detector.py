"""
app/detector.py — License plate detection using YOLOv8.

Wraps the Ultralytics YOLO model to detect license plates in camera frames.
Designed so any YOLOv8-compatible .pt file can be swapped in — just change
DETECTOR_MODEL_PATH in .env.

Usage:
    from app.detector import PlateDetector
    detector = PlateDetector(settings)
    detections = detector.detect(frame)
    annotated = detector.draw(frame, detections)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("turkish_lpr.detector")


# ─────────────────────────────────────────────────────────────────────────────
# Detection result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single detected license plate bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str = "license_plate"

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> dict:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": round(self.confidence, 4),
            "class_name": self.class_name,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Plate Detector
# ─────────────────────────────────────────────────────────────────────────────

class PlateDetector:
    """
    YOLOv8-based license plate detector.

    Loads any Ultralytics-compatible .pt model.  The default is the
    keremberke pretrained plate model, but a locally fine-tuned model
    can replace it by changing DETECTOR_MODEL_PATH in .env.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float | None = None,
    ):
        """
        Args:
            model_path:  Path to a YOLOv8 .pt file.  If None, reads from config.
            confidence:  Minimum detection confidence.  If None, reads from config.
        """
        # Lazy import — ultralytics is heavy; don't slow down unrelated commands
        from ultralytics import YOLO

        from app.config import get_settings

        settings = get_settings()
        self._model_path = Path(model_path or settings.detector_model_abs_path)
        self._confidence = confidence if confidence is not None else settings.detector_confidence

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self._model_path}\n"
                f"Run: python -m app.main download-model"
            )

        logger.info("Loading YOLO model from %s", self._model_path)
        self._model = YOLO(str(self._model_path))
        logger.info(
            "Model loaded — confidence threshold: %.2f", self._confidence
        )

    # ── Core detection ──────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run plate detection on a single frame.

        Args:
            frame: BGR image (numpy array from OpenCV).

        Returns:
            List of Detection objects, sorted by confidence (highest first).
        """
        results = self._model(frame, conf=self._confidence, verbose=False)

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names.get(cls_id, "license_plate")

                detections.append(Detection(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    confidence=conf,
                    class_name=cls_name,
                ))

        # Sort by confidence, highest first
        detections.sort(key=lambda d: d.confidence, reverse=True)

        logger.info(
            "Detected %d plate(s) in frame (conf ≥ %.2f)",
            len(detections),
            self._confidence,
        )
        return detections

    # ── Visualization ───────────────────────────────────────────────────────

    @staticmethod
    def draw(
        frame: np.ndarray,
        detections: list[Detection],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        font_scale: float = 0.6,
    ) -> np.ndarray:
        """
        Draw bounding boxes and confidence labels on a frame.

        Args:
            frame:       BGR image (will NOT be modified in-place).
            detections:  List of Detection objects.
            color:       BGR color for boxes and text.
            thickness:   Line thickness in pixels.
            font_scale:  Font scale for the confidence label.

        Returns:
            A copy of the frame with annotations drawn.
        """
        annotated = frame.copy()

        for i, det in enumerate(detections):
            # Draw the bounding box
            cv2.rectangle(
                annotated,
                (det.x1, det.y1),
                (det.x2, det.y2),
                color,
                thickness,
            )

            # Label: "PLATE 87.3%"
            label = f"PLATE {det.confidence:.1%}"
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )

            # Background rectangle for readability
            label_y = max(det.y1 - 8, label_size[1] + 4)
            cv2.rectangle(
                annotated,
                (det.x1, label_y - label_size[1] - 4),
                (det.x1 + label_size[0] + 4, label_y + 4),
                color,
                cv2.FILLED,
            )

            # Text (black on colored background)
            cv2.putText(
                annotated,
                label,
                (det.x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return annotated
