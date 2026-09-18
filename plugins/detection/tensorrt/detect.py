"""Minimal TensorRT YOLO detector for the paid TensorRT edition."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from schema import Det
from yolo_decode import decode_yolo_output, letterbox

logger = logging.getLogger(__name__)


class TensorRTDetector:
    """Load a YOLO TensorRT ``.engine`` and run detect on BGR frames."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "0",
        imgsz: int = 640,
        confidence: float = 0.35,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics loads the .engine in this v1 runner. "
                "pip install ultralytics  # plus a working TensorRT stack"
            ) from exc

        self.path = Path(model_path)
        if self.path.suffix.lower() not in {".engine", ".trt", ".plan"}:
            raise ValueError(f"Expected a TensorRT engine path, got {self.path}")
        self.imgsz = imgsz
        self.confidence = confidence
        self._model = YOLO(str(self.path), task="detect")
        self.device = device
        logger.info("TensorRT engine=%s device=%s", self.path, device)

    def predict(self, frame: np.ndarray) -> list[Det]:
        """Detect objects on one BGR frame via Ultralytics TRT runtime."""
        results = self._model.predict(
            frame,
            conf=self.confidence,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        dets: list[Det] = []
        if not results:
            return dets
        result = results[0]
        names = result.names or {}
        if result.boxes is None:
            return dets
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cid = int(box.cls[0])
            dets.append(
                Det(
                    bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    confidence=conf,
                    class_id=cid,
                    class_name=str(names.get(cid, cid)),
                )
            )
        return dets

    def predict_raw_engine(self, frame: np.ndarray) -> list[Det]:
        """Fallback path reserved for a future pure-TensorRT runner."""
        nchw, scale, pad_x, pad_y = letterbox(frame, self.imgsz)
        del nchw, scale, pad_x, pad_y
        raise NotImplementedError("Use predict() via Ultralytics TRT engine load in v1")
