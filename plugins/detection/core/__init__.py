"""Shared helpers for edition run.py entry points."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CORE = Path(__file__).resolve().parent
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def ensure_mode(run: dict) -> str:
    """Return ``ipc`` or ``standalone``; exit with help text if neither is set."""
    ipc = run.get("ipc")
    source = run.get("source")
    if ipc:
        return "ipc"
    if source is not None and str(source) != "":
        return "standalone"
    raise SystemExit(
        "Set --ipc cam0 (attach topic 01+02) or --source <webcam|file|rtsp>.\n"
        "  IPC:  python topics/01-capture-runtime/run.py --source 0 --no-consumer\n"
        "        python topics/02-inference-runtime/run.py --ipc cam0\n"
        "        python plugins/detection/<edition>/run.py --ipc cam0 --imshow\n"
        "  Solo: python plugins/detection/<edition>/run.py --source 0 --imshow"
    )
