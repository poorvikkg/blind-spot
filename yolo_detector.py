"""
=============================================================================
  BLIND SPOT DETECTION SYSTEM — Optional YOLO Classifier
  File: yolo_detector.py
=============================================================================
  Drop-in upgrade that replaces the contour heuristic with YOLOv8-nano
  vehicle classification. Only activated when config.USE_YOLO = True.

  Model download (first run only, ~6 MB):
      pip install ultralytics
      python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

  COCO class IDs relevant to vehicles:
      2  = car
      3  = motorcycle
      5  = bus
      7  = truck

  Performance targets:
      Laptop GPU  → 30+ FPS
      Raspberry Pi 4 (CPU) → 4–8 FPS (use yolov8n.pt + YOLO_INPUT_SIZE=256)
=============================================================================
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np

import config
from logger_system import get_logger

log = get_logger("YOLODetector")


class YOLODetector:
    """
    Wraps Ultralytics YOLOv8 for vehicle detection in the ROI crop.

    The detector receives the *already-cropped ROI frame* (BGR numpy array)
    and returns a list of bounding boxes  [(x, y, w, h), ...] for detected
    vehicles, directly compatible with the CentroidTracker interface.
    """

    def __init__(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(config.YOLO_MODEL_PATH)
            log.info("YOLOv8 model loaded from '%s'", config.YOLO_MODEL_PATH)
        except ImportError:
            log.error(
                "ultralytics package not installed. "
                "Run: pip install ultralytics"
            )
            raise
        except Exception as exc:
            log.error("Failed to load YOLO model: %s", exc)
            raise

    def detect(self, roi_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Run inference on a cropped ROI frame.

        Args:
            roi_frame: BGR image (the ROI crop, not the full frame).

        Returns:
            List of (x, y, w, h) bounding boxes (ROI-relative coordinates)
            for each detected vehicle that passes the confidence threshold.
        """
        results = self._model(
            roi_frame,
            imgsz=config.YOLO_INPUT_SIZE,
            conf=config.YOLO_CONFIDENCE,
            classes=config.YOLO_CLASSES,
            verbose=False,
        )

        bboxes: List[Tuple[int, int, int, int]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                # xyxy → x, y, w, h
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bboxes.append((x1, y1, x2 - x1, y2 - y1))

        log.debug("YOLO detected %d vehicles in ROI", len(bboxes))
        return bboxes

    def annotate(
        self,
        roi_frame: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        class_names: List[str] | None = None,
    ) -> np.ndarray:
        """
        Draw YOLO bounding boxes with confidence labels onto the ROI frame.

        Args:
            roi_frame:   The ROI crop (will be modified in place).
            bboxes:      List of (x, y, w, h) from detect().
            class_names: Optional list of COCO class names for labelling.

        Returns:
            Annotated ROI frame (same array, modified in place).
        """
        import cv2  # lazy import — only used if YOLO is enabled

        labels = class_names or ["car", "motorcycle", "bus", "truck"]
        for (x, y, w, h) in bboxes:
            cv2.rectangle(roi_frame, (x, y), (x + w, y + h), config.COLOR_BBOX, 2)
            cv2.putText(
                roi_frame, "VEHICLE",
                (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                config.COLOR_BBOX, 1, cv2.LINE_AA,
            )
        return roi_frame
