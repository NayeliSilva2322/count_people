"""Standalone OpenCV capture loop: detect_fn(frame) -> dets → draw."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import cv2
import numpy as np

from draw import draw_dets, draw_overlay, imshow_scaled
from ipc_loop import filter_dets
from schema import Det

logger = logging.getLogger(__name__)


def open_capture(source: str | int) -> cv2.VideoCapture:
    """Open webcam index or path/URL."""
    raw = source
    if isinstance(source, str) and source.isdigit():
        raw = int(source)
    cap = cv2.VideoCapture(raw)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source!r}")
    return cap


def run_standalone(
    source: str | int,
    detect_fn: Callable[[np.ndarray], list[Det]],
    *,
    confidence: float,
    classes: list[int | str] | None,
    imshow: bool,
    imshow_scale: float = 1.0,
    infer_label: str = "infer=local",
) -> None:
    """Read frames, detect, preview."""
    cap = open_capture(source)
    fps_value = 0.0
    window = "plugins-detection:standalone"
    logger.info("Standalone source=%s", source)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            started = time.perf_counter()
            dets = filter_dets(
                detect_fn(frame),
                confidence=confidence,
                classes=classes,
            )
            elapsed = time.perf_counter() - started
            fps_value = 0.9 * fps_value + 0.1 / max(1e-6, elapsed)
            vis = draw_dets(frame.copy(), dets)
            vis = draw_overlay(
                vis,
                [
                    infer_label,
                    f"dets={len(dets)}  fps={fps_value:.1f}",
                ],
            )
            if imshow:
                imshow_scaled(window, vis, imshow_scale)
                if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted (Ctrl+C)")
    finally:
        cap.release()
        if imshow:
            cv2.destroyAllWindows()
