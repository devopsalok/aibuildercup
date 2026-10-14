"""Explainable crop-stress screening estimates for supported crop metadata."""

from __future__ import annotations

from dataclasses import dataclass

from src.recommendation.engine import FieldData

from .crop_model_utils import crop_details, validate_finite_range


@dataclass(frozen=True)
class StressEstimate:
    score: int
    level: str
    water_stress: int
    vegetation_stress: int
    heat_stress: int
    critical_growth_stage: bool
    explanation: str


def estimate_stress(
    field: FieldData,
    *,
    heat_onset_c: float = 35.0,
    severe_heat_c: float = 45.0,
) -> StressEstimate:
    """Combine moisture, NDVI, and heat indicators into a 0-100 screening score."""
    crop = crop_details(field.crop)
    validate_finite_range("Heat onset (C)", heat_onset_c, -30, 60)
    validate_finite_range("Severe heat (C)", severe_heat_c, -30, 60)
    if severe_heat_c <= heat_onset_c:
        raise ValueError("Severe heat temperature must be greater than heat onset temperature.")

    critical_stages = crop.get("critical_stages", [])
    is_critical_stage = field.growth_stage in critical_stages
    moisture_trigger = 35.0 if is_critical_stage else 25.0
    water_stress = round(
        100 * max(0.0, min(1.0, (moisture_trigger - field.soil_moisture_pct) / moisture_trigger))
    )
    vegetation_stress = round(
        100 * max(0.0, min(1.0, (0.65 - field.ndvi) / (0.65 - 0.30)))
    )
    heat_stress = round(
        100
        * max(
            0.0,
            min(1.0, (field.temperature_c - heat_onset_c) / (severe_heat_c - heat_onset_c)),
        )
    )
    score = round(0.45 * water_stress + 0.35 * vegetation_stress + 0.20 * heat_stress)
    level = "Low" if score < 25 else "Moderate" if score < 50 else "High" if score < 75 else "Critical"
    return StressEstimate(
        score=score,
        level=level,
        water_stress=water_stress,
        vegetation_stress=vegetation_stress,
        heat_stress=heat_stress,
        critical_growth_stage=is_critical_stage,
        explanation=(
            "Composite of soil-water, NDVI, and air-temperature screening indicators. "
            "It is not a measured physiological stress value or disease diagnosis."
        ),
    )