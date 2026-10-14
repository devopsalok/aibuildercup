"""Transparent irrigation requirement estimator for supported crop metadata."""

from __future__ import annotations

from dataclasses import dataclass

from src.recommendation.engine import FieldData

from .crop_model_utils import crop_details, validate_finite_range


@dataclass(frozen=True)
class IrrigationEstimate:
    recommendation: str
    trigger_moisture_pct: float
    target_moisture_pct: float
    effective_rainfall_mm: float
    estimated_net_requirement_mm: float
    recommended_depth_mm: float
    water_volume_liters: float
    reason: str


def estimate_irrigation(
    field: FieldData,
    *,
    root_depth_m: float,
    available_water_capacity_mm_per_m: float,
    irrigation_available: bool = True,
    irrigation_efficiency_pct: float = 85.0,
    target_moisture_pct: float = 80.0,
    rainfall_effectiveness_pct: float = 80.0,
) -> IrrigationEstimate:
    """Estimate a refill depth from relative plant-available water and forecast rain.

    Soil moisture is interpreted as percent of plant-available water, not volumetric
    water content or sensor-specific percent saturation.
    """
    crop = crop_details(field.crop)
    validate_finite_range("Root depth (m)", root_depth_m, 0.05, 5.0)
    validate_finite_range(
        "Available water capacity (mm/m)",
        available_water_capacity_mm_per_m,
        1.0,
        1000.0,
    )
    validate_finite_range("Irrigation efficiency (%)", irrigation_efficiency_pct, 1, 100)
    validate_finite_range("Target moisture (%)", target_moisture_pct, 1, 100)
    validate_finite_range("Rainfall effectiveness (%)", rainfall_effectiveness_pct, 0, 100)
    if not isinstance(irrigation_available, bool):
        raise ValueError("Irrigation availability must be a boolean.")

    critical_stages = crop.get("critical_stages", [])
    is_critical_stage = field.growth_stage in critical_stages
    trigger = 35.0 if is_critical_stage else 25.0
    effective_rainfall = (
        field.rainfall_mm_24h
        * field.rainfall_probability_24h
        * rainfall_effectiveness_pct
        / 100
    )
    soil_water_deficit = (
        max(0.0, target_moisture_pct - field.soil_moisture_pct)
        / 100
        * available_water_capacity_mm_per_m
        * root_depth_m
        / (irrigation_efficiency_pct / 100)
    )
    net_requirement = max(0.0, soil_water_deficit - effective_rainfall)
    triggered = field.soil_moisture_pct < trigger
    recommended_depth = (
        net_requirement
        if triggered and field.rainfall_probability_24h < 0.7 and irrigation_available
        else 0.0
    )

    if not irrigation_available:
        recommendation = "Irrigation unavailable; monitor"
        reason = "The field data indicate a possible refill need, but irrigation is unavailable."
    elif not triggered:
        recommendation = "No immediate irrigation indicated"
        reason = f"Soil moisture is at or above the {trigger:g}% trigger for this stage."
    elif field.rainfall_probability_24h >= 0.7 and net_requirement == 0:
        recommendation = "Delay; forecast rain may cover the deficit"
        reason = "Effective forecast rainfall is estimated to cover the soil-water refill deficit."
    elif field.rainfall_probability_24h >= 0.7:
        recommendation = "Reassess after forecast rain"
        reason = "Soil moisture is below the stage trigger; reassess after expected rainfall."
    elif net_requirement > 0:
        recommendation = "Irrigation indicated"
        reason = "Soil moisture is below the stage trigger and estimated effective rainfall does not cover the refill deficit."
    else:
        recommendation = "No irrigation depth estimated"
        reason = "The estimated refill deficit is covered by effective forecast rainfall."

    water_volume = recommended_depth * field.area_ha * 10_000
    return IrrigationEstimate(
        recommendation=recommendation,
        trigger_moisture_pct=trigger,
        target_moisture_pct=target_moisture_pct,
        effective_rainfall_mm=effective_rainfall,
        estimated_net_requirement_mm=net_requirement,
        recommended_depth_mm=recommended_depth,
        water_volume_liters=water_volume,
        reason=reason,
    )