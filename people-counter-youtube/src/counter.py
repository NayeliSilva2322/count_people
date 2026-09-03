
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from .utils import get_logger


log = get_logger()

def draw_boxes(frame, boxes, color=(0, 200, 255), thickness=1, font_scale=0.5):
    """Rectángulo completo, línea fina, con la probabilidad de cada detección."""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, f"{conf:.2f}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    return frame

def count_people(
    video_path: Path,
    model_name: str,
    conf: float,
    img_size: int,
    frame_skip: int,
    person_class_id: int,
    save_evidence_path: Path = None,
    save_video_path: Path = None,
    show_live: bool = True,
) -> dict:
    """Recorre el video mostrando en vivo el conteo de personas detectadas por frame
    y se queda con el aforo máximo simultáneo (más personas en un mismo frame)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Cargando {model_name} en {device}")
    model = YOLO(model_name)
    if device == "cuda":
        model.to(device).half()  # FP16: casi el doble de rápido en GPU

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay_ms = max(1, int(1000 / fps))  # ritmo real del video en la ventana en vivo

    writer = None
    if save_video_path:
        writer = cv2.VideoWriter(str(save_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    max_count, max_frame_img = 0, None
    last_count, display_frame = 0, None
    frame_idx, processed = 0, 0
    r = None  # guardamos el último resultado para reusar boxes en frames salteados

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_skip == 0:
            r = model.track(
                frame,
                imgsz=img_size,
                conf=conf,
                iou=0.75,
                classes=[person_class_id],
                device=device,
                half=(device == "cuda"),
                persist=True,
                tracker="botsort.yaml",
                verbose=False,
            )[0]
            last_count = len(r.boxes)
            processed += 1
            if last_count > max_count:
                max_count, max_frame_img = last_count, draw_boxes(frame.copy(), r.boxes)

        display_frame = draw_boxes(frame.copy(), r.boxes) if r is not None else frame
        cv2.putText(display_frame, f"Personas: {last_count}  (max: {max_count})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if writer:
            writer.write(display_frame)
        if show_live:
            cv2.imshow("People Counter - presiona 'q' para salir", display_frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if show_live:
        cv2.destroyAllWindows()
    if save_evidence_path and max_frame_img is not None:
        cv2.imwrite(str(save_evidence_path), max_frame_img)
        log.info(f"Evidencia del aforo máximo guardada en {save_evidence_path}")

    return {"max_count": max_count, "frames_processed": processed, "total_frames": frame_idx}