"""Plugin-local detection result type."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Det:
    """One detection box (xyxy)."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str = ""
