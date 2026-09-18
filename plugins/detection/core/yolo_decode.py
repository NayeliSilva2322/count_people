"""Minimal YOLO letterbox + decode helpers for ORT/TRT plugin editions."""

from __future__ import annotations

import numpy as np

from schema import Det

# COCO-80 short list for class names (same as ipc_loop).
COCO_NAMES = [
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
]


def letterbox(
    image: np.ndarray,
    imgsz: int = 640,
    pad_value: float = 114.0,
) -> tuple[np.ndarray, float, float, float]:
    """Resize BGR with pad to square; return NCHW float32 RGB/255, scale, pad_x, pad_y."""
    height, width = image.shape[:2]
    scale = min(imgsz / height, imgsz / width)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    resized = image
    if (new_w, new_h) != (width, height):
        import cv2

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (imgsz - new_w) / 2
    pad_y = (imgsz - new_h) / 2
    canvas = np.full((imgsz, imgsz, 3), pad_value, dtype=np.float32)
    x0, y0 = int(round(pad_x)), int(round(pad_y))
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized.astype(np.float32)
    rgb = canvas[:, :, ::-1] / 255.0
    nchw = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)
    return nchw, scale, pad_x, pad_y


def scale_xyxy(
    box: tuple[float, float, float, float],
    *,
    scale: float,
    pad_x: float,
    pad_y: float,
) -> tuple[float, float, float, float]:
    """Map letterboxed xyxy back to original frame."""
    x1, y1, x2, y2 = box
    return (
        (x1 - pad_x) / scale,
        (y1 - pad_y) / scale,
        (x2 - pad_x) / scale,
        (y2 - pad_y) / scale,
    )


def decode_yolo_output(
    array: np.ndarray,
    *,
    scale: float,
    pad_x: float,
    pad_y: float,
    confidence: float,
    names: list[str] | None = None,
) -> list[Det]:
    """Decode YOLO ORT/TRT output to Det list.

    Supports end-to-end NMS export ``(n, 6)`` = x1,y1,x2,y2,conf,cls and
    raw ``(1, 84, N)`` / ``(1, N, 84)`` with a simple conf filter (no NMS).
    Prefer exporting with ``nms=True`` for production.
    """
    names = names or COCO_NAMES
    arr = np.asarray(array)
    dets: list[Det] = []

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[1] == 6:
        for row in arr:
            conf = float(row[4])
            if conf < confidence:
                continue
            cid = int(row[5])
            bbox = scale_xyxy(
                (float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
            )
            name = names[cid] if 0 <= cid < len(names) else str(cid)
            dets.append(Det(bbox=bbox, confidence=conf, class_id=cid, class_name=name))
        return dets

    # Raw YOLO: (84, N) or (N, 84) — cx,cy,w,h + class scores
    if arr.ndim == 2:
        if arr.shape[0] in {84, 85} or arr.shape[0] < arr.shape[1]:
            arr = arr.T
        for row in arr:
            scores = row[4:]
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf < confidence:
                continue
            cx, cy, w, h = map(float, row[:4])
            bbox = scale_xyxy(
                (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
            )
            name = names[cid] if 0 <= cid < len(names) else str(cid)
            dets.append(Det(bbox=bbox, confidence=conf, class_id=cid, class_name=name))
    return dets
