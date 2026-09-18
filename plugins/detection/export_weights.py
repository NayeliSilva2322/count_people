#!/usr/bin/env python3
"""Export ``yolo26s.pt`` (or any Ultralytics ``.pt``) to ``.onnx`` / ``.engine``.

Used by the ORT and TensorRT editions for standalone detect::

    python plugins/detection/export_weights.py --format onnx
    python plugins/detection/export_weights.py --format engine --device 0
    python plugins/detection/export_weights.py --format onnx,engine --device 0

Requires: Ultralytics (``pip install -e ".[ml]"`` or pack ``requirements.txt``).
Engine export needs a real NVIDIA TensorRT Python package on a CUDA GPU.

IMPORTANT: --imgsz is baked into .onnx / .engine. Export with the same
imgsz you use at runtime (run.imgsz / --imgsz). Mismatch → wrong boxes
or ORT/TRT shape errors.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_PT = _REPO / "models" / "detection" / "yolo26s.pt"


def _ensure_tensorrt_ready() -> None:
    """Make ``tensorrt.__version__`` available before Ultralytics export.

    Ultralytics calls ``check_version(trt.__version__, …)``. Some TensorRT pip
    / wheel installs expose bindings but omit ``__version__``. A local
    ``tensorrt.py`` script can also shadow the real package.
    """
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError(
            "TensorRT Python package is missing.\n"
            "  Match Torch CUDA (python -c \"import torch; print(torch.version.cuda)\"):\n"
            "    pip install -U pip wheel\n"
            "    pip install --extra-index-url https://pypi.nvidia.com tensorrt-cu13\n"
            "    # or tensorrt-cu12 for CUDA 12.x\n"
            "  Do not use bare `pip install tensorrt` (hangs on tiny .tar.gz without NVIDIA index).\n"
            "  Verify: python -c \"import tensorrt as t; print(t.__file__, t.__version__)\""
        ) from exc

    trt_file = getattr(trt, "__file__", None) or ""
    if trt_file.endswith("tensorrt.py") or Path(trt_file).name == "tensorrt.py":
        raise ImportError(
            f"Importing a local script, not NVIDIA TensorRT: {trt_file}\n"
            "Rename/move that file (do not name scripts tensorrt.py), then retry."
        )

    if hasattr(trt, "__version__") and trt.__version__:
        logger.info("TensorRT %s (%s)", trt.__version__, trt_file)
        return

    version = None
    try:
        import importlib.metadata as md

        for dist_name in ("tensorrt", "tensorrt_cu12", "tensorrt_cu13", "nvidia-tensorrt"):
            try:
                version = md.version(dist_name)
                break
            except md.PackageNotFoundError:
                continue
    except Exception:
        version = None

    if not version:
        raise ImportError(
            "module 'tensorrt' has no attribute '__version__' and package "
            "metadata was not found. Reinstall from the NVIDIA index:\n"
            "  pip uninstall -y tensorrt tensorrt_cu12 tensorrt_cu13 "
            "tensorrt_libs tensorrt_bindings "
            "tensorrt_cu12_libs tensorrt_cu12_bindings "
            "tensorrt_cu13_libs tensorrt_cu13_bindings\n"
            "  pip install -U pip wheel\n"
            "  pip install --extra-index-url https://pypi.nvidia.com tensorrt-cu13\n"
            "  python -c \"import tensorrt as t; print(t.__file__, t.__version__)\""
        )

    trt.__version__ = version
    logger.warning(
        "Patched tensorrt.__version__=%s (missing on this install). file=%s",
        version,
        trt_file,
    )


def export_weights(
    pt_path: Path,
    *,
    formats: list[str],
    imgsz: int = 640,
    device: str = "cpu",
) -> dict[str, Path]:
    """Export a Ultralytics checkpoint; return format → output path."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required to export. pip install -e '.[ml]'"
        ) from exc

    if not pt_path.is_file():
        raise FileNotFoundError(
            f"Weights not found: {pt_path}. Place yolo26s.pt under models/detection/"
        )

    need_engine = any(str(f).strip().lower() == "engine" for f in formats)
    if need_engine:
        _ensure_tensorrt_ready()
        if str(device) in {"cpu", "CPU"}:
            logger.warning(
                "Engine export on device=%s usually fails; use --device 0",
                device,
            )

    model = YOLO(str(pt_path))
    exported: dict[str, Path] = {}
    for fmt in formats:
        key = str(fmt).strip().lower()
        if key not in {"onnx", "engine"}:
            raise ValueError(f"Unsupported format {fmt!r}. Use onnx or engine.")
        kwargs: dict = {"format": key, "imgsz": imgsz, "device": device}
        if key == "onnx":
            kwargs["nms"] = True
        logger.info("export %s → %s (imgsz=%s device=%s)", pt_path, key, imgsz, device)
        out = model.export(**kwargs)
        exported[key] = Path(str(out))
        logger.info("wrote %s", exported[key])
    return exported


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export detection weights to ONNX and/or TensorRT engine"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=_DEFAULT_PT,
        help=f"Ultralytics .pt path (default: {_DEFAULT_PT})",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        help="Comma list: onnx | engine | onnx,engine (default: onnx)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input size baked into ONNX/engine. Must match run.imgsz / --imgsz at infer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Export device (cpu for onnx; 0 / cuda:0 for engine)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    formats = [p.strip() for p in str(args.format).split(",") if p.strip()]
    pt = Path(args.model)
    if not pt.is_file():
        candidate = _REPO / args.model
        if candidate.is_file():
            pt = candidate
    try:
        exported = export_weights(
            pt.resolve(),
            formats=formats,
            imgsz=args.imgsz,
            device=str(args.device),
        )
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    for key, path in exported.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
