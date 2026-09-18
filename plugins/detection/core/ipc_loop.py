"""Attach IpcSlot + InferSlot and draw detections (CPU IPC)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import cv2
import numpy as np

from common.media.infer_ipc import InferSlot
from common.media.ipc import IpcSlot
# Pack ships only ipc.py + infer_ipc.py — do not import other media modules.

from draw import draw_dets, draw_overlay, imshow_scaled
from schema import Det

logger = logging.getLogger(__name__)

_COCO = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def rows_to_dets(
    rows: list[tuple[float, float, float, float, float, int]],
) -> list[Det]:
    """Convert InferPacket.rows to plugin Det list."""
    out: list[Det] = []
    for x1, y1, x2, y2, conf, class_id in rows:
        cid = int(class_id)
        name = _COCO[cid] if 0 <= cid < len(_COCO) else str(cid)
        out.append(
            Det(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(conf),
                class_id=cid,
                class_name=name,
            )
        )
    return out


def filter_dets(
    dets: list[Det],
    *,
    confidence: float,
    classes: list[int | str] | None,
) -> list[Det]:
    """Filter by confidence and optional class id / name allow-list."""
    allow_ids: set[int] | None = None
    allow_names: set[str] | None = None
    if classes:
        allow_ids = {c for c in classes if isinstance(c, int)}
        allow_names = {str(c).lower() for c in classes if not isinstance(c, int)}
        if not allow_ids:
            allow_ids = None
        if not allow_names:
            allow_names = None

    kept: list[Det] = []
    for det in dets:
        if det.confidence < confidence:
            continue
        if allow_ids is not None and det.class_id in allow_ids:
            kept.append(det)
            continue
        if allow_names is not None and det.class_name.lower() in allow_names:
            kept.append(det)
            continue
        if allow_ids is None and allow_names is None:
            kept.append(det)
    return kept


def run_ipc(
    camera_id: str,
    *,
    confidence: float,
    classes: list[int | str] | None,
    imshow: bool,
    imshow_scale: float = 1.0,
    infer_label: str = "infer=topic-02",
    on_frame: Callable[[np.ndarray, list[Det]], np.ndarray] | None = None,
) -> None:
    """Plugin loop: InferSlot + IpcSlot → filter → draw.

    Does not load a local detector. Topic 01 and 02 must already be running.
    """
    frames = IpcSlot.attach(camera_id)
    infer = InferSlot.attach(camera_id)
    last_infer = 0
    fps_value = 0.0
    window = f"plugins-detection:{camera_id}"
    logger.info("IPC attach camera_id=%s (IpcSlot + InferSlot)", camera_id)
    try:
        while True:
            if infer.seq == last_infer:
                infer.wait(last_infer, 0.05)
                if infer.seq == last_infer:
                    continue
            packet = infer.latest()
            if packet is None:
                continue
            last_infer = packet.infer_seq
            deadline = time.monotonic() + 0.1
            while frames.seq < packet.frame_seq and time.monotonic() < deadline:
                frames.wait(frames.seq, 0.02)
            frame = frames.latest()
            if frame is None:
                continue
            frame = frame.copy()
            dets = filter_dets(
                rows_to_dets(packet.rows),
                confidence=confidence,
                classes=classes,
            )
            started = time.perf_counter()
            if on_frame is not None:
                vis = on_frame(frame, dets)
            else:
                vis = draw_dets(frame.copy(), dets)
                fps_value = 0.9 * fps_value + 0.1 / max(1e-6, time.perf_counter() - started)
                vis = draw_overlay(
                    vis,
                    [
                        infer_label,
                        f"dets={len(dets)}  fps={fps_value:.1f}",
                        f"ipc={camera_id}",
                    ],
                )
            if imshow:
                imshow_scaled(window, vis, imshow_scale)
                if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted (Ctrl+C)")
    finally:
        infer.close()
        frames.close()
        if imshow:
            cv2.destroyAllWindows()
