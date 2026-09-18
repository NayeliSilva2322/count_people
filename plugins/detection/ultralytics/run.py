"""Detection plugin — Ultralytics edition (FREE).

IPC mode attaches topic 01+02 MediaBus and draws InferSlot boxes.
Standalone uses YOLO.predict (default yolo26s.pt).

    python plugins/detection/ultralytics/run.py --ipc cam0 --imshow
    python plugins/detection/ultralytics/run.py \
      --source data/videos/trackdm1.mp4 --model models/detection/yolo26s.pt --imshow
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_EDITION_ROOT = Path(__file__).resolve().parent
_CORE = _EDITION_ROOT.parent / "core"
_REPO = _EDITION_ROOT.parents[2]
for path in (_CORE, _REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import add_common_args, load_yaml, merge_run_config  # noqa: E402
from ipc_loop import run_ipc  # noqa: E402
from schema import Det  # noqa: E402
from standalone import run_standalone  # noqa: E402

logger = logging.getLogger(__name__)

EDITION = "ultralytics"
LICENSE = "free"


def _detect_ultralytics(model_path: str, device: str, confidence: float, imgsz: int = 640):
    from ultralytics import YOLO

    model = YOLO(model_path)

    def detect(frame) -> list[Det]:
        results = model.predict(
            frame,
            conf=confidence,
            device=device,
            imgsz=imgsz,
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

    return detect


def main() -> None:
    """CLI entry for the Ultralytics (free) detection plugin."""
    parser = argparse.ArgumentParser(description="plugins/detection ultralytics (free)")
    add_common_args(parser)
    args = parser.parse_args()
    config_path = Path(args.config) if args.config else _EDITION_ROOT / "configs" / "run.yaml"
    yaml_data = load_yaml(config_path) if config_path.is_file() else {}
    cfg = merge_run_config(yaml_data, args, edition=EDITION, license_tier=LICENSE)
    run = cfg["run"]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.info("edition=%s license=%s", cfg["edition"], cfg["license"])

    device = str(run.get("device") or "cpu")
    ipc = run.get("ipc")
    source = run.get("source")
    if ipc:
        run_ipc(
            str(ipc),
            confidence=float(run["confidence"]),
            classes=run.get("classes"),
            imshow=bool(run.get("imshow")),
            imshow_scale=float(run.get("imshow_scale", 1.0)),
            infer_label="infer=topic-02 (boxes from InferSlot)",
        )
        return
    if source is None or str(source) == "":
        raise SystemExit(
            "Set --ipc cam0 or --source <webcam|file|rtsp>.\n"
            "  python plugins/detection/ultralytics/run.py --ipc cam0 --imshow\n"
            "  python plugins/detection/ultralytics/run.py --source 0 --imshow"
        )
    model = run.get("model") or "models/detection/yolo26s.pt"
    detect = _detect_ultralytics(
        str(model), device, float(run["confidence"]), int(run.get("imgsz", 640))
    )
    run_standalone(
        source if not str(source).isdigit() else int(source),
        detect,
        confidence=float(run["confidence"]),
        classes=run.get("classes"),
        imshow=bool(run.get("imshow")),
        imshow_scale=float(run.get("imshow_scale", 1.0)),
        infer_label=f"infer={EDITION} device={device}",
    )


if __name__ == "__main__":
    main()
