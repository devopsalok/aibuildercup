"""Rule-based field risk assessment and what-if action ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RULES_DIR = ROOT_DIR / "decision_rules"
DEFAULT_CROP_METADATA = ROOT_DIR / "knowledgebase" / "crops" / "metadata" / "crop_metadata.yml"
RISK_CATEGORIES = ("water_stress", "crop_stress", "climate", "disease", "nutrient")
OPERATORS = {">", ">=", "<", "<=", "==", "!=", "in"}
DEFAULT_WEIGHTS = {"risk": 0.8, "water": 0.1, "cost": 0.1}


class RuleConfigurationError(ValueError):
    """Raised when a decision-rule YAML file is missing or invalid."""


@dataclass(frozen=True)
class FieldData:
    crop: str
    growth_stage: str
    soil_moisture_pct: float
    temperature_c: float
    rainfall_probability_24h: float
    rainfall_mm_24h: float
    humidity_pct: float
    ndvi: float
    nitrogen_mg_kg: float
    phosphorus_mg_kg: float
    potassium_mg_kg: float
    symptoms_reported: bool
    area_ha: float
    water_cost_per_mm_ha: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.crop, str)
            or not isinstance(self.growth_stage, str)
            or not self.crop.strip()
            or not self.growth_stage.strip()
        ):
            raise ValueError("Crop and growth stage are required.")
        _validate_range("soil moisture", self.soil_moisture_pct, 0, 100)
        _validate_range("temperature", self.temperature_c, -30, 60)
        _validate_range("rainfall probability", self.rainfall_probability_24h, 0, 1)
        _validate_range("rainfall amount", self.rainfall_mm_24h, 0, 2000)
        _validate_range("humidity", self.humidity_pct, 0, 100)
        _validate_range("NDVI", self.ndvi, -1, 1)
        _validate_range("nitrogen", self.nitrogen_mg_kg, 0, 10000)
        _validate_range("phosphorus", self.phosphorus_mg_kg, 0, 10000)
        _validate_range("potassium", self.potassium_mg_kg, 0, 10000)
        _validate_range("field area", self.area_ha, 0.000001, 1000000)
        _validate_range("water cost", self.water_cost_per_mm_ha, 0, 1000000)
        if not isinstance(self.symptoms_reported, bool):
            raise ValueError("Symptoms reported must be a boolean.")


@dataclass(frozen=True)
class RuleFinding:
    rule_id: str
    category: str
    name: str
    score: int
    risk_level: str
    action: str
    priority: str
    explanations: tuple[str, ...]


@dataclass(frozen=True)
class Assessment:
    scores: dict[str, int]
    findings: tuple[RuleFinding, ...]


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    name: str
    irrigation_mm: float
    water_liters: float
    estimated_cost: float
    overall_risk_score: float
    ranking_score: float
    description: str


def _validate_range(label: str, value: float, minimum: float, maximum: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{label.capitalize()} must be between {minimum} and {maximum}.")


def risk_level(score: float) -> str:
    if score >= 75:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Moderate"
    return "Low"


def load_crop_metadata(path: Path = DEFAULT_CROP_METADATA) -> dict[str, dict[str, Any]]:
    document = _read_yaml(path)
    crops = document.get("crops")
    if not isinstance(crops, dict) or not crops:
        raise RuleConfigurationError(f"{path}: expected a non-empty 'crops' mapping.")
    return crops


class RuleEngine:
    def __init__(self, rules: list[dict[str, Any]], scenarios: list[dict[str, Any]]) -> None:
        self.rules = rules
        self.scenarios = scenarios

    @classmethod
    def from_directory(cls, directory: Path = DEFAULT_RULES_DIR) -> RuleEngine:
        if not directory.is_dir():
            raise RuleConfigurationError(f"Decision rules directory does not exist: {directory}")
        rules: list[dict[str, Any]] = []
        scenarios: list[dict[str, Any]] | None = None
        seen_rule_ids: set[str] = set()
        seen_scenario_ids: set[str] = set()
        for path in sorted(directory.glob("*.yaml")):
            document = _read_yaml(path)
            file_rules = document.get("rules", [])
            if not isinstance(file_rules, list):
                raise RuleConfigurationError(f"{path}: 'rules' must be a list.")
            for rule in file_rules:
                _validate_rule(rule, path)
                if rule["id"] in seen_rule_ids:
                    raise RuleConfigurationError(f"{path}: duplicate rule id {rule['id']!r}.")
                seen_rule_ids.add(rule["id"])
                rules.append(rule)
            if "scenarios" in document:
                if scenarios is not None:
                    raise RuleConfigurationError("Only one YAML file may define scenarios.")
                scenarios = document["scenarios"]
                if not isinstance(scenarios, list) or not scenarios:
                    raise RuleConfigurationError(f"{path}: 'scenarios' must be a non-empty list.")
                for scenario in scenarios:
                    _validate_scenario(scenario, path)
                    if scenario["id"] in seen_scenario_ids:
                        raise RuleConfigurationError(f"{path}: duplicate scenario id {scenario['id']!r}.")
                    seen_scenario_ids.add(scenario["id"])
        if not rules:
            raise RuleConfigurationError(f"No decision rules were found in {directory}.")
        if scenarios is None:
            raise RuleConfigurationError(f"No scenario definitions were found in {directory}.")
        return cls(rules, scenarios)

    def assess(self, field: FieldData) -> Assessment:
        values = vars(field)
        findings: list[RuleFinding] = []
        scores = {category: 0 for category in RISK_CATEGORIES}
        for rule in self.rules:
            if all(_condition_matches(condition, values) for condition in rule["conditions"]):
                score = int(rule["score"])
                action = rule.get("action", {})
                finding = RuleFinding(
                    rule_id=rule["id"],
                    category=rule["category"],
                    name=rule["name"],
                    score=score,
                    risk_level=risk_level(score),
                    action=action.get("type", "monitor"),
                    priority=action.get("priority", "medium"),
                    explanations=tuple(rule.get("explanation", [])),
                )
                findings.append(finding)
                scores[finding.category] = max(scores[finding.category], score)
        return Assessment(scores=scores, findings=tuple(findings))

    def rank_scenarios(
        self,
        field: FieldData,
        assessment: Assessment,
        irrigation_available: bool = True,
        weights: dict[str, float] | None = None,
    ) -> list[ScenarioResult]:
        selected_weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        if set(selected_weights) != set(DEFAULT_WEIGHTS):
            raise ValueError("Weights must contain exactly 'risk', 'water', and 'cost'.")
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or not 0 <= weight <= 1
            for weight in selected_weights.values()
        ):
            raise ValueError("Scenario weights must be finite numbers from 0 to 1.")
        weight_total = sum(selected_weights.values())
        if weight_total <= 0:
            raise ValueError("At least one scenario weight must be greater than zero.")
        selected_weights = {key: value / weight_total for key, value in selected_weights.items()}

        available_scenarios = [
            scenario
            for scenario in self.scenarios
            if irrigation_available or scenario["irrigation_mm"] == 0
        ]
        if not available_scenarios:
            raise RuleConfigurationError("No feasible scenarios remain for the selected field.")

        results: list[dict[str, Any]] = []
        max_water = max(scenario["irrigation_mm"] for scenario in available_scenarios)
        for scenario in available_scenarios:
            irrigation_mm = float(scenario["irrigation_mm"])
            water_stress = max(
                0,
                assessment.scores["water_stress"] - int(scenario["water_stress_reduction"]),
            )
            category_scores = dict(assessment.scores)
            category_scores["water_stress"] = water_stress
            overall_risk = max(category_scores.values())
            liters = irrigation_mm * field.area_ha * 10000
            cost = irrigation_mm * field.area_ha * field.water_cost_per_mm_ha
            results.append(
                {
                    "scenario": scenario,
                    "risk": overall_risk,
                    "liters": liters,
                    "cost": cost,
                    "water_fraction": irrigation_mm / max_water if max_water else 0,
                }
            )

        max_cost = max(result["cost"] for result in results)
        for result in results:
            risk_fraction = result["risk"] / 100
            cost_fraction = result["cost"] / max_cost if max_cost else 0
            result["ranking_score"] = (
                selected_weights["risk"] * risk_fraction
                + selected_weights["water"] * result["water_fraction"]
                + selected_weights["cost"] * cost_fraction
            )
        results.sort(
            key=lambda result: (
                result["ranking_score"],
                result["risk"],
                result["scenario"]["irrigation_mm"],
                result["scenario"]["id"],
            )
        )
        return [
            ScenarioResult(
                scenario_id=result["scenario"]["id"],
                name=result["scenario"]["name"],
                irrigation_mm=float(result["scenario"]["irrigation_mm"]),
                water_liters=round(result["liters"], 1),
                estimated_cost=round(result["cost"], 2),
                overall_risk_score=round(result["risk"], 1),
                ranking_score=round(result["ranking_score"], 4),
                description=result["scenario"]["description"],
            )
            for result in results
        ]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuleConfigurationError(f"Required YAML file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise RuleConfigurationError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(document, dict):
        raise RuleConfigurationError(f"{path}: expected a YAML mapping.")
    return document


def _validate_rule(rule: Any, path: Path) -> None:
    if not isinstance(rule, dict):
        raise RuleConfigurationError(f"{path}: each rule must be a mapping.")
    required = {"id", "name", "category", "conditions", "score"}
    if not required.issubset(rule):
        raise RuleConfigurationError(f"{path}: each rule requires {sorted(required)}.")
    if not isinstance(rule["id"], str) or not isinstance(rule["name"], str):
        raise RuleConfigurationError(f"{path}: rule id and name must be strings.")
    if rule["category"] not in RISK_CATEGORIES:
        raise RuleConfigurationError(
            f"{path}: rule {rule['id']} has unknown category {rule['category']!r}."
        )
    if isinstance(rule["score"], bool) or not isinstance(rule["score"], int) or not 0 <= rule["score"] <= 100:
        raise RuleConfigurationError(f"{path}: rule {rule['id']} score must be an integer from 0 to 100.")
    if not isinstance(rule["conditions"], list) or not rule["conditions"]:
        raise RuleConfigurationError(f"{path}: rule {rule['id']} needs at least one condition.")
    for condition in rule["conditions"]:
        if (
            not isinstance(condition, dict)
            or not {"field", "operator", "value"}.issubset(condition)
            or not isinstance(condition["field"], str)
            or not isinstance(condition["operator"], str)
            or condition["operator"] not in OPERATORS
            or condition["field"] not in FieldData.__dataclass_fields__
        ):
            raise RuleConfigurationError(f"{path}: rule {rule['id']} has an invalid condition.")
        if condition["operator"] == "in" and not isinstance(condition["value"], list):
            raise RuleConfigurationError(
                f"{path}: rule {rule['id']} 'in' conditions must use a list of values."
            )
    action = rule.get("action", {})
    if not isinstance(action, dict):
        raise RuleConfigurationError(f"{path}: rule {rule['id']} action must be a mapping.")
    explanation = rule.get("explanation", [])
    if not isinstance(explanation, list) or not all(isinstance(item, str) for item in explanation):
        raise RuleConfigurationError(f"{path}: rule {rule['id']} explanation must be a list of strings.")


def _validate_scenario(scenario: Any, path: Path) -> None:
    required = {"id", "name", "irrigation_mm", "water_stress_reduction", "description"}
    if not isinstance(scenario, dict) or not required.issubset(scenario):
        raise RuleConfigurationError(f"{path}: each scenario requires {sorted(required)}.")
    for key in ("irrigation_mm", "water_stress_reduction"):
        value = scenario[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise RuleConfigurationError(f"{path}: scenario {key} must be a finite non-negative number.")
    if not all(isinstance(scenario[key], str) for key in ("id", "name", "description")):
        raise RuleConfigurationError(f"{path}: scenario id, name, and description must be strings.")


def _condition_matches(condition: dict[str, Any], values: dict[str, Any]) -> bool:
    field = condition["field"]
    if field not in values:
        raise RuleConfigurationError(f"Rule refers to unknown field {field!r}.")
    actual = values[field]
    expected = condition["value"]
    operator = condition["operator"]
    try:
        if operator == ">":
            return actual > expected
        if operator == ">=":
            return actual >= expected
        if operator == "<":
            return actual < expected
        if operator == "<=":
            return actual <= expected
        if operator == "==":
            return actual == expected
        if operator == "!=":
            return actual != expected
        return actual in expected
    except TypeError as error:
        raise RuleConfigurationError(
            f"Rule condition for {field!r} compares incompatible values."
        ) from error