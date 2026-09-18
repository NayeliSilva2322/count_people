"""Detection plugin — ONNX Runtime edition (PAID).

    python plugins/detection/ort/run.py --ipc cam0 --imshow
    python plugins/detection/ort/run.py --source 0 --model models/detection/yolo26s.onnx --imshow
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_EDITION_ROOT = Path(__file__).resolve().parent
_CORE = _EDITION_ROOT.parent / "core"
_REPO = _EDITION_ROOT.parents[2]
for path in (_CORE, _REPO, _EDITION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import add_common_args, load_yaml, merge_run_config  # noqa: E402
from detect import OrtDetector  # noqa: E402
from ipc_loop import run_ipc  # noqa: E402
from standalone import run_standalone  # noqa: E402

logger = logging.getLogger(__name__)

EDITION = "ort"
LICENSE = "paid"


def main() -> None:
    """CLI entry for the ORT (paid) detection plugin."""
    parser = argparse.ArgumentParser(description="plugins/detection ort (paid)")
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
            "  python plugins/detection/ort/run.py --ipc cam0 --imshow\n"
            "  python plugins/detection/ort/run.py --source 0 --model model.onnx --imshow"
        )
    model = run.get("model")
    if not model:
        raise SystemExit("ORT standalone requires --model path/to/model.onnx")
    detector = OrtDetector(
        model,
        device=device,
        confidence=float(run["confidence"]),
        imgsz=int(run.get("imgsz", 640)),
        names=run.get("names"),
    )
    run_standalone(
        source if not str(source).isdigit() else int(source),
        detector.predict,
        confidence=float(run["confidence"]),
        classes=run.get("classes"),
        imshow=bool(run.get("imshow")),
        imshow_scale=float(run.get("imshow_scale", 1.0)),
        infer_label=f"infer={EDITION} device={device}",
    )


if __name__ == "__main__":
    main()
