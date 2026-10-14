"""Shared validation and crop metadata helpers for agronomic estimators."""

from __future__ import annotations

import math
from typing import Any

from src.recommendation.engine import load_crop_metadata


def crop_details(crop: str) -> dict[str, Any]:
    crops = load_crop_metadata()
    if crop not in crops:
        raise ValueError(f"Unsupported crop {crop!r}. Choose one of: {', '.join(crops)}.")
    return crops[crop]


def validate_finite_range(
    name: str, value: float, minimum: float, maximum: float
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be a finite number from {minimum} to {maximum}.")
