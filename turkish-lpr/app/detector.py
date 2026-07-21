"""
app/detector.py — License plate detection + OCR using FastALPR.

FastALPR bundles both detection (open-image-models) and OCR (fast-plate-ocr)
in a single ONNX-based framework.  A single `detect()` call returns both
bounding boxes and raw OCR text.

The detector model and OCR model can be swapped by changing
DETECTOR_MODEL_NAME and OCR_MODEL_NAME in .env.

Usage:
    from app.detector import PlateDetector
    detector = PlateDetector()
    detections = detector.detect(frame)
    annotated = PlateDetector.draw(frame, detections)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("turkish_lpr.detector")


# ─────────────────────────────────────────────────────────────────────────────
# Detection result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single detected license plate with bounding box and OCR result."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float                  # detection confidence
    ocr_text: str = ""                 # raw OCR text from FastALPR
    ocr_confidence: float = 0.0       # OCR confidence score
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
            "ocr_text": self.ocr_text,
            "ocr_confidence": round(self.ocr_confidence, 4),
            "class_name": self.class_name,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Plate Detector (FastALPR)
# ─────────────────────────────────────────────────────────────────────────────

class PlateDetector:
    """
    License plate detector + OCR using FastALPR.

    FastALPR uses ONNX models that are auto-downloaded on first use.
    No manual model download step is needed.

    To swap models, change DETECTOR_MODEL_NAME / OCR_MODEL_NAME in .env.
    """

    def __init__(
        self,
        detector_model: str | None = None,
        ocr_model: str | None = None,
        confidence: float | None = None,
    ):
        """
        Args:
            detector_model:  FastALPR detector model name.  If None, reads from config.
            ocr_model:       FastALPR OCR model name.  If None, reads from config.
            confidence:      Minimum detection confidence.  If None, reads from config.
        """
        # Lazy import — fast_alpr pulls in onnxruntime which is heavy
        from fast_alpr import ALPR

        from app.config import get_settings

        settings = get_settings()
        self._detector_model = detector_model or settings.detector_model_name
        self._ocr_model = ocr_model or settings.ocr_model_name
        self._confidence = confidence if confidence is not None else settings.detector_confidence

        logger.info(
            "Initializing FastALPR — detector=%s, ocr=%s",
            self._detector_model,
            self._ocr_model,
        )

        self._alpr = ALPR(
            detector_model=self._detector_model,
            ocr_model=self._ocr_model,
        )

        logger.info(
            "FastALPR ready — confidence threshold: %.2f", self._confidence
        )

    # ── Core detection ──────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run plate detection + OCR on a single frame.

        Args:
            frame: BGR image (numpy array from OpenCV).

        Returns:
            List of Detection objects, sorted by confidence (highest first).
        """
        # FastALPR expects RGB input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self._alpr.predict(frame_rgb)

        detections: list[Detection] = []
        for result in results:
            det = result.detection
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = det.box.x1, det.box.y1, det.box.x2, det.box.y2
            det_conf = float(det.confidence)

            # Filter by confidence threshold
            if det_conf < self._confidence:
                continue

            ocr_text = ""
            ocr_conf = 0.0
            
            if result.ocr:
                ocr_text = result.ocr.text
                # confidence might be a single float or a list of floats per character
                if isinstance(result.ocr.confidence, list):
                    ocr_conf = sum(result.ocr.confidence) / len(result.ocr.confidence) if result.ocr.confidence else 0.0
                else:
                    ocr_conf = float(result.ocr.confidence)

            detections.append(Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=det_conf,
                ocr_text=ocr_text,
                ocr_confidence=ocr_conf,
            ))

        # Sort by detection confidence, highest first
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
        Draw bounding boxes, confidence labels, and OCR text on a frame.

        Args:
            frame:       BGR image (will NOT be modified in-place).
            detections:  List of Detection objects.
            color:       BGR color for boxes and text.
            thickness:   Line thickness in pixels.
            font_scale:  Font scale for labels.

        Returns:
            A copy of the frame with annotations drawn.
        """
        annotated = frame.copy()

        for det in detections:
            # Draw the bounding box
            cv2.rectangle(
                annotated,
                (det.x1, det.y1),
                (det.x2, det.y2),
                color,
                thickness,
            )

            # Label: "34ABC123 87.3%" (OCR text + detection confidence)
            if det.ocr_text:
                label = f"{det.ocr_text} {det.confidence:.1%}"
            else:
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
