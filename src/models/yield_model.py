"""Linear stress-adjusted yield scenarios using user-supplied local calibration."""

from __future__ import annotations

from dataclasses import dataclass

from .crop_model_utils import crop_details, validate_finite_range


@dataclass(frozen=True)
class YieldEstimate:
    baseline_yield_kg_ha: float
    estimated_yield_kg_ha: float
    estimated_field_yield_kg: float
    stress_score: float
    estimated_loss_pct: float
    yield_loss_at_full_stress_pct: float


def estimate_yield(
    crop: str,
    *,
    baseline_yield_kg_ha: float,
    stress_score: float,
    yield_loss_at_full_stress_pct: float,
    area_ha: float = 1.0,
) -> YieldEstimate:
    """Calculate a configurable yield scenario; this is not a trained yield forecast.

    The baseline must come from the user's local records or an agronomic source.
    The maximum-stress loss percentage is an explicit user-calibrated assumption.
    """
    crop_details(crop)
    validate_finite_range("Baseline yield (kg/ha)", baseline_yield_kg_ha, 0.001, 1_000_000)
    validate_finite_range("Stress score", stress_score, 0, 100)
    validate_finite_range("Yield loss at full stress (%)", yield_loss_at_full_stress_pct, 0, 100)
    validate_finite_range("Field area (ha)", area_ha, 0.000001, 1_000_000)

    estimated_loss_pct = stress_score * yield_loss_at_full_stress_pct / 100
    estimated_yield = baseline_yield_kg_ha * (1 - estimated_loss_pct / 100)
    return YieldEstimate(
        baseline_yield_kg_ha=baseline_yield_kg_ha,
        estimated_yield_kg_ha=estimated_yield,
        estimated_field_yield_kg=estimated_yield * area_ha,
        stress_score=stress_score,
        estimated_loss_pct=estimated_loss_pct,
        yield_loss_at_full_stress_pct=yield_loss_at_full_stress_pct,
    )