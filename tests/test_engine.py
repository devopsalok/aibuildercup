import unittest
from pathlib import Path

from src.recommendation.engine import (
    FieldData,
    RuleConfigurationError,
    RuleEngine,
    load_crop_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


def make_field(**overrides: object) -> FieldData:
    values: dict[str, object] = {
        "crop": "wheat",
        "growth_stage": "tillering",
        "soil_moisture_pct": 12.0,
        "temperature_c": 28.0,
        "rainfall_probability_24h": 0.2,
        "rainfall_mm_24h": 0.0,
        "humidity_pct": 60.0,
        "ndvi": 0.65,
        "nitrogen_mg_kg": 35.0,
        "phosphorus_mg_kg": 18.0,
        "potassium_mg_kg": 140.0,
        "symptoms_reported": False,
        "area_ha": 1.0,
        "water_cost_per_mm_ha": 0.1,
    }
    values.update(overrides)
    return FieldData(**values)  # type: ignore[arg-type]


class RuleEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = RuleEngine.from_directory(ROOT / "decision_rules")

    def test_all_rules_and_crop_metadata_load(self) -> None:
        crops = load_crop_metadata(ROOT / "knowledgebase" / "crops" / "metadata" / "crop_metadata.yml")
        self.assertIn("wheat", crops)
        self.assertGreaterEqual(len(self.engine.rules), 10)
        self.assertEqual(len(self.engine.scenarios), 3)

    def test_assessment_matches_multiple_explainable_risks(self) -> None:
        result = self.engine.assess(make_field())
        self.assertEqual(result.scores["water_stress"], 90)
        self.assertEqual(result.scores["crop_stress"], 0)
        self.assertIn("IRR-001", {finding.rule_id for finding in result.findings})
        self.assertTrue(all(finding.explanations for finding in result.findings))

    def test_irrigation_scenarios_rank_and_calculate_water_and_cost(self) -> None:
        field = make_field()
        result = self.engine.assess(field)
        scenarios = self.engine.rank_scenarios(field, result)
        self.assertEqual(scenarios[0].name, "Reduced irrigation")
        reduced = next(item for item in scenarios if item.scenario_id == "reduced_irrigation")
        self.assertEqual(reduced.water_liters, 100000)
        self.assertEqual(reduced.estimated_cost, 1.0)
        self.assertLess(scenarios[0].ranking_score, scenarios[-1].ranking_score)

    def test_unavailable_irrigation_excludes_water_using_scenarios(self) -> None:
        field = make_field()
        scenarios = self.engine.rank_scenarios(field, self.engine.assess(field), irrigation_available=False)
        self.assertEqual([scenario.scenario_id for scenario in scenarios], ["delay_and_monitor"])

    def test_forecast_rain_is_explained_without_raising_stress_score(self) -> None:
        field = make_field(
            soil_moisture_pct=40,
            rainfall_probability_24h=0.8,
            rainfall_mm_24h=8,
        )
        result = self.engine.assess(field)
        rain_finding = next(item for item in result.findings if item.rule_id == "IRR-007")
        self.assertEqual(rain_finding.action, "consider_delaying_irrigation")
        self.assertEqual(result.scores["water_stress"], 0)

    def test_critical_growth_stage_rule_is_crop_specific(self) -> None:
        wheat = self.engine.assess(
            make_field(crop="wheat", growth_stage="tillering", soil_moisture_pct=30)
        )
        rice = self.engine.assess(
            make_field(crop="rice", growth_stage="tillering", soil_moisture_pct=30)
        )
        self.assertEqual(wheat.scores["water_stress"], 0)
        self.assertEqual(rice.scores["water_stress"], 70)

    def test_symptoms_trigger_inspection_not_diagnosis(self) -> None:
        result = self.engine.assess(make_field(symptoms_reported=True))
        finding = next(item for item in result.findings if item.rule_id == "DIS-002")
        self.assertEqual(finding.action, "inspect_and_confirm_symptoms")
        self.assertIn("before treatment", finding.explanations[0])

    def test_input_ranges_and_weights_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "Soil moisture"):
            make_field(soil_moisture_pct=101)
        with self.assertRaisesRegex(ValueError, "Soil moisture"):
            make_field(soil_moisture_pct=float("nan"))
        field = make_field()
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            self.engine.rank_scenarios(
                field,
                self.engine.assess(field),
                weights={"risk": 0, "water": 0, "cost": 0},
            )
        with self.assertRaisesRegex(ValueError, "from 0 to 1"):
            self.engine.rank_scenarios(
                field,
                self.engine.assess(field),
                weights={"risk": float("nan"), "water": 0.5, "cost": 0.5},
            )

    def test_invalid_rule_directory_fails_explicitly(self) -> None:
        with self.assertRaises(RuleConfigurationError):
            RuleEngine.from_directory(ROOT / "missing-rules")


if __name__ == "__main__":
    unittest.main()
