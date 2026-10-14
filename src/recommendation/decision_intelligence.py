"""Explainable, multi-domain agricultural decision synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.models.irrigation_model import IrrigationEstimate
from src.models.stress_model import StressEstimate
from src.models.yield_model import YieldEstimate
from src.recommendation.engine import Assessment, FieldData, risk_level

from src.models.crop_model_utils import crop_details, validate_finite_range


DECISION_DOMAINS = (
    "climate",
    "crop_stress",
    "disease",
    "economics",
    "fertilizers",
    "irrigation",
    "pests",
    "soils",
    "sustainability",
    "yields",
)


@dataclass(frozen=True)
class DecisionContext:
    """Optional measurements required for economic, soil, and pest decisions."""

    soil_ph: float | None = None
    soil_ec_ds_m: float | None = None
    organic_matter_pct: float | None = None
    pest_observed: bool | None = None
    pest_affected_area_pct: float | None = None
    baseline_yield_kg_ha: float | None = None
    yield_loss_at_full_stress_pct: float | None = None
    market_price_per_kg: float | None = None
    production_cost_per_ha: float | None = None


@dataclass(frozen=True)
class DomainDecision:
    domain: str
    status: str
    priority: str
    score: int | None
    recommendation: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class DecisionReport:
    crop: str
    overall_priority: str
    decisions: tuple[DomainDecision, ...]


def build_decision_intelligence(
    field: FieldData,
    assessment: Assessment,
    stress: StressEstimate,
    irrigation: IrrigationEstimate,
    yield_scenario: YieldEstimate | None,
    context: DecisionContext = DecisionContext(),
    *,
    disease_probability: float | None = None,
) -> DecisionReport:
    """Synthesize ten decision domains without treating screening scores as probabilities."""
    crop = crop_details(field.crop)
    if not isinstance(context.pest_observed, (bool, type(None))):
        raise ValueError("Pest observation must be true, false, or not recorded.")
    if context.pest_affected_area_pct is not None:
        validate_finite_range("Pest-affected area (%)", context.pest_affected_area_pct, 0, 100)
    if disease_probability is not None:
        if field.crop != "tomato":
            raise ValueError("Disease image probabilities are supported only for tomato.")
        validate_finite_range("Disease model probability", disease_probability, 0, 1)
    for name, value, maximum in (
        ("Soil pH", context.soil_ph, 14),
        ("Soil electrical conductivity (dS/m)", context.soil_ec_ds_m, 100),
        ("Soil organic matter (%)", context.organic_matter_pct, 100),
        ("Baseline yield (kg/ha)", context.baseline_yield_kg_ha, 1_000_000),
        ("Yield loss at full stress (%)", context.yield_loss_at_full_stress_pct, 100),
        ("Market price per kg", context.market_price_per_kg, 1_000_000),
        ("Production cost per ha", context.production_cost_per_ha, 1_000_000_000),
    ):
        if value is not None:
            validate_finite_range(name, value, 0, maximum)

    decisions = [
        _climate_decision(field, assessment, crop),
        _stress_decision(stress),
        _disease_decision(field, assessment, disease_probability, crop),
        _economics_decision(field, context, yield_scenario),
        _fertilizer_decision(field, assessment),
        _irrigation_decision(irrigation),
        _pest_decision(context),
        _soil_decision(field, context),
        _sustainability_decision(irrigation, yield_scenario, context),
        _yield_decision(stress, yield_scenario),
    ]
    priority_order = {"Critical": 4, "High": 3, "Moderate": 2, "Low": 1, "Unassessed": 0}
    overall = max(
        (decision.priority for decision in decisions),
        key=lambda priority: priority_order[priority],
    )
    return DecisionReport(crop=field.crop, overall_priority=overall, decisions=tuple(decisions))


def _decision(
    domain: str,
    score: int | None,
    recommendation: str,
    evidence: tuple[str, ...],
    *,
    status: str | None = None,
) -> DomainDecision:
    if score is None:
        return DomainDecision(
            domain=domain,
            status=status or "More data needed",
            priority="Unassessed",
            score=None,
            recommendation=recommendation,
            evidence=evidence,
        )
    bounded_score = max(0, min(100, int(score)))
    return DomainDecision(
        domain=domain,
        status=status or risk_level(bounded_score),
        priority=risk_level(bounded_score),
        score=bounded_score,
        recommendation=recommendation,
        evidence=evidence,
    )


def _climate_decision(
    field: FieldData, assessment: Assessment, crop: Mapping[str, object]
) -> DomainDecision:
    heat = min(100, max(0, round((field.temperature_c - 30) * 12.5)))
    cold = min(100, max(0, round((10 - field.temperature_c) * 10)))
    wet = min(100, max(0, round(field.rainfall_mm_24h / 75 * 100)))
    drought = min(100, max(0, round((35 - field.soil_moisture_pct) / 35 * 100)))
    climate_rule_score = assessment.scores.get("climate", 0)
    score = max(heat, cold, wet, drought, climate_rule_score)
    risks = set(crop.get("major_risks", []))
    sensitive = (
        (heat > 0 and any("heat" in risk or "temperature" in risk for risk in risks))
        or (drought > 0 and any("drought" in risk or "water" in risk for risk in risks))
        or (wet > 0 and any("flood" in risk or "waterlog" in risk for risk in risks))
    )
    if sensitive:
        score = min(100, round(score * 1.1))
    evidence = (
        f"Temperature: {field.temperature_c:g} C",
        f"Forecast rain: {field.rainfall_probability_24h:.0%}, {field.rainfall_mm_24h:g} mm",
        f"Soil moisture: {field.soil_moisture_pct:g}% plant-available water",
        f"Crop metadata indicates related hazard sensitivity: {'yes' if sensitive else 'not indicated'}",
    )
    recommendation = (
        "Prepare for the crop-relevant weather hazard; check the local forecast and field drainage."
        if score >= 50
        else "Continue weather and field monitoring; reassess after forecast changes."
    )
    return _decision("Climate", score, recommendation, evidence)


def _stress_decision(stress: StressEstimate) -> DomainDecision:
    return _decision(
        "Crop stress",
        stress.score,
        "Inspect affected zones and compare with prior observations to identify the likely cause.",
        (
            f"Water stress indicator: {stress.water_stress}/100",
            f"Vegetation indicator: {stress.vegetation_stress}/100",
            f"Heat indicator: {stress.heat_stress}/100",
            stress.explanation,
        ),
    )


def _disease_decision(
    field: FieldData,
    assessment: Assessment,
    disease_probability: float | None,
    crop: Mapping[str, object],
) -> DomainDecision:
    rule_score = assessment.scores.get("disease", 0)
    if disease_probability is not None:
        score = max(rule_score, round(disease_probability * 100))
        evidence = (
            f"Image-model disease score: {disease_probability:.1%} (tomato-only, not a diagnosis)",
            f"Symptoms reported: {'yes' if field.symptoms_reported else 'no'}",
        )
        recommendation = (
            "Inspect affected plants and seek a validated diagnosis before treatment."
            if score >= 50 or field.symptoms_reported
            else "Continue scouting; the image screen does not rule disease out."
        )
    else:
        score = rule_score if rule_score > 0 else None
        evidence = (
            f"Symptoms reported: {'yes' if field.symptoms_reported else 'no'}",
            "No compatible image result was supplied; weather favorability is only a risk indicator.",
        )
        recommendation = (
            "Inspect and confirm symptoms with a qualified local source before treatment."
            if field.symptoms_reported
            else "Scout the crop; no image-based disease assessment was supplied."
        )
    if score is None and "disease" not in crop.get("major_risks", []):
        return _decision(
            "Disease",
            None,
            "No disease risk is configured for this crop; continue routine scouting.",
            evidence,
        )
    return _decision("Disease", score, recommendation, evidence)


def _economics_decision(
    field: FieldData,
    context: DecisionContext,
    yield_scenario: YieldEstimate | None,
) -> DomainDecision:
    if (
        yield_scenario is None
        or context.market_price_per_kg is None
        or context.production_cost_per_ha is None
    ):
        missing = []
        if context.baseline_yield_kg_ha is None:
            missing.append("locally supported baseline yield")
        if context.market_price_per_kg is None:
            missing.append("market price")
        if context.production_cost_per_ha is None:
            missing.append("production cost per hectare")
        return _decision(
            "Economics",
            None,
            f"Enter {', '.join(missing)} to estimate revenue and gross margin.",
            ("Prices and costs should be timestamped and region-specific.",),
        )
    revenue = yield_scenario.estimated_field_yield_kg * context.market_price_per_kg
    costs = context.production_cost_per_ha * field.area_ha
    margin = revenue - costs
    score = 0 if margin >= 0 else min(100, round(abs(margin) / max(costs, 1) * 100))
    return _decision(
        "Economics",
        score,
        "Review the yield, price, and cost assumptions before comparing intervention economics.",
        (
            f"Estimated gross revenue: {revenue:,.2f} cost units",
            f"Estimated production cost: {costs:,.2f} cost units",
            f"Estimated gross margin: {margin:,.2f} cost units",
        ),
        status="Positive margin scenario" if margin >= 0 else "Negative margin scenario",
    )


def _fertilizer_decision(field: FieldData, assessment: Assessment) -> DomainDecision:
    score = assessment.scores.get("nutrient", 0)
    evidence = (
        f"Soil N: {field.nitrogen_mg_kg:g} mg/kg",
        f"Soil P: {field.phosphorus_mg_kg:g} mg/kg",
        f"Soil K: {field.potassium_mg_kg:g} mg/kg",
    )
    recommendation = (
        "Review the soil test with local crop- and soil-specific guidance; do not infer an application rate from this screen."
        if score > 0
        else "Use a current soil test and crop-specific nutrient plan; avoid fertilizer application based on this screen alone."
    )
    return _decision("Fertilizers", score, recommendation, evidence)


def _irrigation_decision(irrigation: IrrigationEstimate) -> DomainDecision:
    recommendation_label = irrigation.recommendation.lower()
    if "unavailable" in recommendation_label:
        score = min(100, max(65, round(irrigation.estimated_net_requirement_mm * 2)))
        recommendation = "Irrigation is unavailable; monitor soil moisture and consider feasible water access."
    elif "reassess after" in recommendation_label:
        score = 50
        recommendation = irrigation.reason
    elif irrigation.recommended_depth_mm > 0:
        score = min(100, 35 + round(irrigation.recommended_depth_mm * 2))
        recommendation = irrigation.reason
    else:
        score = 0
        recommendation = irrigation.reason
    return _decision(
        "Irrigation",
        score,
        recommendation,
        (
            f"Estimated net refill need: {irrigation.estimated_net_requirement_mm:.1f} mm",
            f"Estimated recommended depth: {irrigation.recommended_depth_mm:.1f} mm",
            irrigation.recommendation,
        ),
    )


def _pest_decision(context: DecisionContext) -> DomainDecision:
    if context.pest_observed is None:
        return _decision(
            "Pests",
            None,
            "Record pest scouting observations and severity to assess this domain.",
            ("Pest presence was not recorded.",),
        )
    if not context.pest_observed:
        return _decision(
            "Pests",
            0,
            "Continue routine scouting; no observation reported does not establish pest absence.",
            ("No pest signs were reported during this assessment.",),
        )
    if context.pest_affected_area_pct is None:
        return _decision(
            "Pests",
            None,
            "Scout representative zones and record pest identity and severity; follow integrated pest management guidance.",
            ("Pest signs reported; affected area and identity not supplied.",),
        )
    score = round(context.pest_affected_area_pct)
    return _decision(
        "Pests",
        score,
        "Confirm pest identity, scout representative zones, and use locally validated integrated pest management thresholds.",
        (f"User-estimated pest-affected area: {context.pest_affected_area_pct:.0f}%.",),
    )


def _soil_decision(field: FieldData, context: DecisionContext) -> DomainDecision:
    evidence = [f"Soil moisture: {field.soil_moisture_pct:g}% plant-available water"]
    concerns = 0
    known = 1
    if context.soil_ph is not None:
        known += 1
        evidence.append(f"Soil pH: {context.soil_ph:g}")
        concerns += int(context.soil_ph < 5.5 or context.soil_ph > 7.5)
    if context.soil_ec_ds_m is not None:
        known += 1
        evidence.append(f"Soil EC: {context.soil_ec_ds_m:g} dS/m")
        concerns += int(context.soil_ec_ds_m >= 4)
    if context.organic_matter_pct is not None:
        known += 1
        evidence.append(f"Organic matter: {context.organic_matter_pct:g}%")
    if known == 1:
        return _decision(
            "Soils",
            None,
            "Add a soil test (pH, salinity/EC, and organic matter) for a more complete soil assessment.",
            tuple(evidence),
            status="Partial data",
        )
    score = min(100, concerns * 50 + (25 if context.organic_matter_pct is None else 0))
    recommendation = (
        "Review the soil-test result with local crop-specific guidance before amending."
        if concerns
        else "Retain soil-test records and interpret them using crop- and region-specific thresholds."
    )
    return _decision("Soils", score, recommendation, tuple(evidence))


def _sustainability_decision(
    irrigation: IrrigationEstimate,
    yield_scenario: YieldEstimate | None,
    context: DecisionContext,
) -> DomainDecision:
    evidence = [
        f"Estimated irrigation water: {irrigation.water_volume_liters:,.0f} L",
        "No pesticide or fertilizer use-rate data were supplied.",
    ]
    if yield_scenario is None:
        return _decision(
            "Sustainability",
            None,
            "Track water, nutrient, pesticide, and energy inputs alongside harvested yield to compare efficiency.",
            tuple(evidence),
            status="Partial data",
        )
    water_m3 = irrigation.water_volume_liters / 1000
    water_efficiency = (
        yield_scenario.estimated_field_yield_kg / water_m3 if water_m3 > 0 else None
    )
    if water_efficiency is not None:
        evidence.append(f"Scenario water productivity: {water_efficiency:.1f} kg/m3 of estimated irrigation")
    else:
        evidence.append("Water productivity is not defined when estimated irrigation is zero.")
    evidence.append(
        f"Production cost assumption: {context.production_cost_per_ha if context.production_cost_per_ha is not None else 'not supplied'}"
    )
    return _decision(
        "Sustainability",
        None,
        "Compare resource-use metrics over time and against local benchmarks; no sustainability score is inferred.",
        tuple(evidence),
        status="Metrics only",
    )


def _yield_decision(
    stress: StressEstimate,
    yield_scenario: YieldEstimate | None,
) -> DomainDecision:
    if yield_scenario is None:
        return _decision(
            "Yields",
            stress.score,
            "Provide a locally supported baseline yield and calibrated stress-loss assumption for a yield scenario.",
            (f"Composite stress screen: {stress.score}/100.",),
            status="Scenario not calculated",
        )
    return _decision(
        "Yields",
        stress.score,
        "Use this as a scenario only; compare with local field records and account for other limiting factors.",
        (
            f"Baseline yield: {yield_scenario.baseline_yield_kg_ha:,.0f} kg/ha",
            f"Stress-adjusted scenario: {yield_scenario.estimated_yield_kg_ha:,.0f} kg/ha",
            f"Assumed yield loss: {yield_scenario.estimated_loss_pct:.1f}%",
        ),
        status="Scenario estimate",
    )
