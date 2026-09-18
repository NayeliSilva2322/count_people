"""OpenCV drawing for plugin detections."""

from __future__ import annotations

import cv2
import numpy as np

from schema import Det

_COLORS = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 165, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 255),
    (0, 255, 255),
    (255, 0, 128),
]


def color_for_class(class_id: int) -> tuple[int, int, int]:
    """Stable BGR color from class id."""
    return _COLORS[int(class_id) % len(_COLORS)]


def draw_dets(
    frame: np.ndarray,
    dets: list[Det],
    *,
    thickness: int = 2,
) -> np.ndarray:
    """Draw boxes and class labels on a BGR frame."""
    vis = frame
    for det in dets:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        color = color_for_class(det.class_id)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        label = det.class_name or str(det.class_id)
        label = f"{label} {det.confidence:.2f}"
        cv2.putText(
            vis,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def draw_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw HUD lines top-left (plain green text, no panel or outline)."""
    cleaned = [str(line) for line in lines if str(line).strip()]
    if not cleaned:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    x0, y0 = 10, 22
    line_gap = 20

    for i, text in enumerate(cleaned):
        cv2.putText(
            frame,
            text,
            (x0, y0 + i * line_gap),
            font,
            scale,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def scale_for_imshow(frame: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Resize a BGR frame for preview only (does not affect infer)."""
    value = float(scale)
    if value <= 0:
        raise ValueError(f"imshow_scale must be > 0, got {scale!r}")
    if abs(value - 1.0) < 1e-6:
        return frame
    height, width = frame.shape[:2]
    new_w = max(1, int(round(width * value)))
    new_h = max(1, int(round(height * value)))
    interp = cv2.INTER_AREA if value < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (new_w, new_h), interpolation=interp)


def imshow_scaled(window: str, frame: np.ndarray, scale: float = 1.0) -> None:
    """``cv2.imshow`` with optional display scale."""
    cv2.imshow(window, scale_for_imshow(frame, scale))
