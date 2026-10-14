"""Export the configured crop metadata and decision rules as a flat CSV dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from src.recommendation.engine import (
    DEFAULT_CROP_METADATA,
    DEFAULT_RULES_DIR,
    RuleConfigurationError,
    RuleEngine,
    load_crop_metadata,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = (
    ROOT_DIR / "data" / "processed" / "crop_decision_rules_with_demo_scenarios.csv"
)
DATASET_COLUMNS = (
    "record_type",
    "crop",
    "crop_category",
    "domain",
    "major_crop_risks",
    "critical_growth_stages",
    "rule_id",
    "name",
    "conditions_json",
    "score",
    "action",
    "priority",
    "recommendation",
    "evidence",
    "source_file",
    "data_status",
)

DOMAIN_NAMES = {
    "climate": "climate",
    "crop_stress": "crop_stress",
    "disease": "disease_screening",
    "nutrient": "fertilizer_screening",
    "water_stress": "irrigation_screening",
}


def build_crop_rule_rows(
    rules_directory: Path = DEFAULT_RULES_DIR,
    crop_metadata_path: Path = DEFAULT_CROP_METADATA,
) -> list[dict[str, Any]]:
    """Expand applicable configured rules and scenarios across crop metadata."""
    engine = RuleEngine.from_directory(rules_directory)
    crops = load_crop_metadata(crop_metadata_path)
    source_by_rule: dict[str, str] = {}
    for path in sorted(rules_directory.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        for rule in document.get("rules", []):
            source_by_rule[rule["id"]] = path.name

    rows: list[dict[str, Any]] = []
    for crop_id, metadata in crops.items():
        shared = {
            "crop": crop_id,
            "crop_category": metadata.get("category", ""),
            "major_crop_risks": _json(metadata.get("major_risks", [])),
            "critical_growth_stages": _json(metadata.get("critical_stages", [])),
        }
        rows.append(
            _row(
                shared,
                record_type="crop_profile",
                domain="crop_profile",
                name=metadata.get("scientific_name", crop_id),
                recommendation=(
                    "Use these crop risk tags and critical stages to scope screening; "
                    "they are metadata, not a diagnosis or management prescription."
                ),
                evidence="knowledgebase/crops/metadata/crop_metadata.yml",
                source_file="crop_metadata.yml",
                data_status="metadata",
            )
        )

        for rule in engine.rules:
            if not _applies_to_crop(rule, crop_id):
                continue
            category = rule["category"]
            recommendation = " ".join(rule.get("explanation", []))
            if category == "disease":
                recommendation = (
                    f"{recommendation} This risk screen does not identify or confirm a disease."
                )
            action = rule.get("action", {})
            rows.append(
                _row(
                    shared,
                    record_type="screening_rule",
                    domain=DOMAIN_NAMES[category],
                    rule_id=rule["id"],
                    name=rule["name"],
                    conditions_json=_json(rule["conditions"]),
                    score=rule["score"],
                    action=action.get("type", ""),
                    priority=action.get("priority", ""),
                    recommendation=recommendation,
                    evidence=recommendation,
                    source_file=source_by_rule[rule["id"]],
                    data_status="configurable screening rule; not a calibrated probability",
                )
            )

        for scenario in engine.scenarios:
            rows.append(
                _row(
                    shared,
                    record_type="recommendation_scenario",
                    domain="irrigation_recommendation",
                    rule_id=scenario["id"],
                    name=scenario["name"],
                    conditions_json="[]",
                    action="compare_scenario",
                    recommendation=scenario["description"],
                    evidence=(
                        f"Irrigation assumption={scenario['irrigation_mm']} mm; "
                        f"water stress score reduction={scenario['water_stress_reduction']}."
                    ),
                    source_file="recommendation_rules.yaml",
                    data_status="demonstration scenario; validate locally",
                )
            )

        rows.extend(_synthetic_domain_examples(shared, crop_id))
    return rows


def export_crop_rule_dataset(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    rules_directory: Path = DEFAULT_RULES_DIR,
    crop_metadata_path: Path = DEFAULT_CROP_METADATA,
) -> Path:
    """Validate and write the crop-expanded rules as UTF-8 CSV."""
    rows = build_crop_rule_rows(rules_directory, crop_metadata_path)
    if not rows:
        raise RuleConfigurationError("No crop decision-rule records were generated.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DATASET_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _applies_to_crop(rule: dict[str, Any], crop: str) -> bool:
    crop_conditions = [
        condition
        for condition in rule["conditions"]
        if condition["field"] == "crop"
    ]
    return all(_condition_includes_crop(condition, crop) for condition in crop_conditions)


def _condition_includes_crop(condition: dict[str, Any], crop: str) -> bool:
    operator = condition["operator"]
    expected = condition["value"]
    if operator == "==":
        return expected == crop
    if operator == "!=":
        return expected != crop
    if operator == "in":
        return crop in expected
    return False


def _synthetic_domain_examples(shared: dict[str, Any], crop: str) -> list[dict[str, Any]]:
    baseline_yield_kg_ha = 5000.0
    stress_score = 20
    yield_loss_at_full_stress_pct = 40.0
    estimated_yield_kg_ha = baseline_yield_kg_ha * (
        1 - stress_score * yield_loss_at_full_stress_pct / 10_000
    )
    area_ha = 1.0
    market_price_per_kg = 2.0
    production_cost_per_ha = 7000.0
    revenue = estimated_yield_kg_ha * area_ha * market_price_per_kg
    total_cost = production_cost_per_ha * area_ha
    margin = revenue - total_cost
    irrigation_depth_mm = 100.0
    irrigation_water_m3 = irrigation_depth_mm * area_ha * 10
    water_productivity = estimated_yield_kg_ha * area_ha / irrigation_water_m3
    prefix = f"DEMO-{crop.upper()}"

    return [
        _row(
            shared,
            record_type="synthetic_demo_scenario",
            domain="disease_screening",
            rule_id=f"{prefix}-DISEASE",
            name="Illustrative humid-weather scouting scenario",
            conditions_json=_json(
                {
                    "temperature_c": 24,
                    "humidity_pct": 90,
                    "rainfall_mm_24h": 12,
                    "symptoms_reported": False,
                    "image_model_probability": None,
                }
            ),
            action="increase_scouting_only",
            priority="illustrative",
            recommendation=(
                "Synthetic example only: humid and wet conditions warrant scouting; "
                "no disease is identified and no treatment is recommended. This scenario "
                "does not identify or confirm a disease."
            ),
            evidence="Generated example inputs; no observed farm or pathogen data.",
            source_file="generated synthetic example",
            data_status="synthetic; illustrative only; excluded from assessment scores",
        ),
        _row(
            shared,
            record_type="synthetic_demo_scenario",
            domain="economics",
            rule_id=f"{prefix}-ECONOMICS",
            name="Illustrative gross-margin scenario",
            conditions_json=_json(
                {
                    "area_ha": area_ha,
                    "baseline_yield_kg_ha": baseline_yield_kg_ha,
                    "stress_score": stress_score,
                    "yield_loss_at_full_stress_pct": yield_loss_at_full_stress_pct,
                    "market_price_per_kg": market_price_per_kg,
                    "production_cost_per_ha": production_cost_per_ha,
                    "estimated_yield_kg_ha": estimated_yield_kg_ha,
                    "estimated_revenue": revenue,
                    "estimated_total_cost": total_cost,
                    "estimated_gross_margin": margin,
                    "currency": "illustrative cost unit; no currency assumed",
                }
            ),
            action="replace_with_local_costs_and_prices",
            priority="illustrative",
            recommendation=(
                "Synthetic example only: replace all yield, price, and cost assumptions "
                "with timestamped local records before decision use."
            ),
            evidence=(
                f"Illustrative revenue={revenue:.2f}; cost={total_cost:.2f}; "
                f"gross margin={margin:.2f} cost units."
            ),
            source_file="generated synthetic example",
            data_status="synthetic; illustrative only; excluded from assessment scores",
        ),
        _row(
            shared,
            record_type="synthetic_demo_scenario",
            domain="sustainability",
            rule_id=f"{prefix}-SUSTAINABILITY",
            name="Illustrative irrigation water-productivity scenario",
            conditions_json=_json(
                {
                    "area_ha": area_ha,
                    "estimated_yield_kg_ha": estimated_yield_kg_ha,
                    "irrigation_depth_mm": irrigation_depth_mm,
                    "irrigation_water_m3": irrigation_water_m3,
                    "water_productivity_kg_per_m3": water_productivity,
                    "fertilizer_use_kg": None,
                    "pesticide_use_kg": None,
                    "energy_use_kwh": None,
                }
            ),
            action="track_inputs_and_harvest",
            priority="illustrative",
            recommendation=(
                "Synthetic example only: water productivity excludes rainfall and other water sources; "
                "record fertilizer, pesticide, energy, and harvested yield for a complete comparison."
            ),
            evidence=(
                f"Illustrative irrigation={irrigation_water_m3:.0f} m3; "
                f"yield={estimated_yield_kg_ha:.0f} kg; water productivity="
                f"{water_productivity:.2f} kg/m3."
            ),
            source_file="generated synthetic example",
            data_status="synthetic; illustrative only; excluded from assessment scores",
        ),
    ]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _row(shared: dict[str, Any], **values: Any) -> dict[str, Any]:
    row = {column: "" for column in DATASET_COLUMNS}
    row.update(shared)
    row.update(values)
    return row


if __name__ == "__main__":
    destination = export_crop_rule_dataset()
    print(f"Wrote crop decision-rule dataset: {destination}")
