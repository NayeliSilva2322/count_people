"""Minimal ONNX Runtime YOLO detector for the paid ORT edition."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from schema import Det
from yolo_decode import decode_yolo_output, letterbox

logger = logging.getLogger(__name__)


class OrtDetector:
    """Load a YOLO ``.onnx`` and run detect on BGR frames."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        imgsz: int = 640,
        confidence: float = 0.35,
        names: list[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for the ort edition. "
                "pip install onnxruntime  # or onnxruntime-gpu on Linux"
            ) from exc

        self.path = Path(model_path)
        self.imgsz = imgsz
        self.confidence = confidence
        self.names = list(names) if names else None
        providers = ["CPUExecutionProvider"]
        if device not in {"cpu", ""}:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                logger.warning("CUDA EP unavailable; using CPU ORT")
        self._session = ort.InferenceSession(str(self.path), providers=providers)
        inputs = self._session.get_inputs()
        self._input_name = inputs[0].name if inputs else "images"
        shape = inputs[0].shape if inputs else None
        if shape is not None and len(shape) == 4:
            if isinstance(shape[2], int) and isinstance(shape[3], int):
                self.imgsz = int(shape[2])
        outputs = self._session.get_outputs()
        self._output_name = outputs[0].name if outputs else None
        if self.names:
            logger.info(
                "ORT model=%s providers=%s names=%s",
                self.path,
                self._session.get_providers(),
                self.names,
            )
        else:
            logger.info(
                "ORT model=%s providers=%s (COCO name fallback — set run.names for custom)",
                self.path,
                self._session.get_providers(),
            )

    def predict(self, frame: np.ndarray) -> list[Det]:
        """Detect objects on one BGR frame."""
        nchw, scale, pad_x, pad_y = letterbox(frame, self.imgsz)
        feeds = {self._input_name: nchw}
        if self._output_name:
            outs = self._session.run([self._output_name], feeds)
        else:
            outs = self._session.run(None, feeds)
        return decode_yolo_output(
            outs[0],
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            confidence=self.confidence,
            names=self.names,
        )
