"""Load plugin YAML and merge CLI overrides."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML mapping (empty dict if missing keys)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add IPC / source / detect flags shared by all editions."""
    parser.add_argument("--config", type=str, default=None, help="Path to run.yaml")
    parser.add_argument("--ipc", type=str, default=None, help="MediaBus camera_id (e.g. cam0)")
    parser.add_argument("--source", type=str, default=None, help="Webcam index, file, or RTSP")
    parser.add_argument("--model", type=str, default=None, help="Infer weights (standalone only)")
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--classes", nargs="*", default=None, help="Class names or ids")
    parser.add_argument("--imshow", action="store_true", default=None)
    parser.add_argument("--no-imshow", action="store_false", dest="imshow")
    parser.add_argument(
        "--imshow-scale",
        type=float,
        default=None,
        help="Display scale for imshow only (e.g. 0.5); does not affect infer",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Ultralytics infer size (standalone). Match --imgsz when exporting ONNX/engine",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Infer device: cpu|0|1|… (standalone; ignored with --ipc)",
    )
    return parser


def merge_run_config(
    yaml_data: dict[str, Any],
    args: argparse.Namespace,
    *,
    edition: str,
    license_tier: str,
) -> dict[str, Any]:
    """Merge YAML + CLI into a flat runtime dict."""
    run = dict(yaml_data.get("run") or {})

    if args.ipc is not None:
        run["ipc"] = args.ipc
    if args.source is not None:
        run["source"] = args.source
    if args.model is not None:
        run["model"] = args.model
    if args.confidence is not None:
        run["confidence"] = args.confidence
    if args.classes is not None:
        run["classes"] = _parse_classes(args.classes)
    if args.imshow is not None:
        run["imshow"] = bool(args.imshow)
    if getattr(args, "imshow_scale", None) is not None:
        run["imshow_scale"] = float(args.imshow_scale)
    if getattr(args, "imgsz", None) is not None:
        run["imgsz"] = int(args.imgsz)
    if args.device is not None:
        run["device"] = args.device

    run.setdefault("ipc", None)
    run.setdefault("source", None)
    run.setdefault("model", None)
    run.setdefault("confidence", 0.35)
    run.setdefault("classes", None)
    run.setdefault("names", None)
    run.setdefault("imshow", False)
    run.setdefault("imshow_scale", 1.0)
    run["imshow_scale"] = _normalize_imshow_scale(run.get("imshow_scale"))
    run.setdefault("imgsz", 640)
    run["imgsz"] = int(run.get("imgsz") or 640)
    run.setdefault("device", "cpu")
    run["names"] = _normalize_names(run.get("names"))

    return {
        "edition": yaml_data.get("edition", edition),
        "license": yaml_data.get("license", license_tier),
        "run": run,
    }


def _normalize_names(value: Any) -> list[str] | None:
    """Parse ``run.names`` as an ordered label list (class_id → name)."""
    if value is None or value == "":
        return None
    if not isinstance(value, list) or not value:
        raise ValueError("run.names must be a non-empty YAML list, e.g. [Fire, Smoke]")
    return [str(item) for item in value]


def _parse_classes(values: list[str]) -> list[int | str]:
    out: list[int | str] = []
    for raw in values:
        try:
            out.append(int(raw))
        except ValueError:
            out.append(raw)
    return out


def _normalize_imshow_scale(value: Any) -> float:
    """Validate display-only scale (> 0)."""
    scale = 1.0 if value is None else float(value)
    if scale <= 0:
        raise ValueError(f"run.imshow_scale must be > 0, got {value!r}")
    return scale
