import unittest

from src.models.irrigation_model import estimate_irrigation
from src.models.stress_model import estimate_stress
from src.models.yield_model import estimate_yield
from src.recommendation.decision_intelligence import (
    DECISION_DOMAINS,
    DecisionContext,
    build_decision_intelligence,
)
from src.recommendation.engine import FieldData, RuleEngine, load_crop_metadata


def make_field(crop: str, **overrides: object) -> FieldData:
    values: dict[str, object] = {
        "crop": crop,
        "growth_stage": "vegetative",
        "soil_moisture_pct": 20.0,
        "temperature_c": 40.0,
        "rainfall_probability_24h": 0.1,
        "rainfall_mm_24h": 0.0,
        "humidity_pct": 90.0,
        "ndvi": 0.4,
        "nitrogen_mg_kg": 15.0,
        "phosphorus_mg_kg": 8.0,
        "potassium_mg_kg": 80.0,
        "symptoms_reported": False,
        "area_ha": 2.0,
        "water_cost_per_mm_ha": 0.1,
    }
    values.update(overrides)
    return FieldData(**values)  # type: ignore[arg-type]


class DecisionIntelligenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = RuleEngine.from_directory()

    def make_report(
        self,
        field: FieldData,
        context: DecisionContext = DecisionContext(),
        *,
        with_yield: bool = False,
        disease_probability: float | None = None,
    ):
        assessment = self.engine.assess(field)
        stress = estimate_stress(field)
        irrigation = estimate_irrigation(
            field,
            root_depth_m=0.6,
            available_water_capacity_mm_per_m=150,
        )
        yield_scenario = (
            estimate_yield(
                field.crop,
                baseline_yield_kg_ha=5000,
                stress_score=stress.score,
                yield_loss_at_full_stress_pct=50,
                area_ha=field.area_ha,
            )
            if with_yield
            else None
        )
        return build_decision_intelligence(
            field,
            assessment,
            stress,
            irrigation,
            yield_scenario,
            context,
            disease_probability=disease_probability,
        )

    def test_all_domains_are_returned_for_every_configured_crop(self) -> None:
        for crop in load_crop_metadata():
            with self.subTest(crop=crop):
                report = self.make_report(make_field(crop))
                self.assertEqual(
                    tuple(
                        decision.domain.lower().replace(" ", "_")
                        for decision in report.decisions
                    ),
                    DECISION_DOMAINS,
                )
                self.assertIn(report.overall_priority, {"Low", "Moderate", "High", "Critical"})

    def test_missing_data_is_explicit_and_never_represented_as_zero_risk(self) -> None:
        report = self.make_report(make_field("tomato"))
        decisions = {decision.domain: decision for decision in report.decisions}
        self.assertEqual(decisions["Economics"].priority, "Unassessed")
        self.assertEqual(decisions["Soils"].status, "Partial data")
        self.assertEqual(decisions["Pests"].priority, "Unassessed")
        self.assertIsNone(decisions["Disease"].score)
        self.assertEqual(decisions["Sustainability"].status, "Partial data")

    def test_soil_pest_and_economic_inputs_feed_recommendations(self) -> None:
        context = DecisionContext(
            soil_ph=5.0,
            soil_ec_ds_m=5.0,
            organic_matter_pct=2.0,
            pest_observed=True,
            pest_affected_area_pct=70,
            baseline_yield_kg_ha=5000,
            yield_loss_at_full_stress_pct=50,
            market_price_per_kg=2,
            production_cost_per_ha=10_000,
        )
        report = self.make_report(make_field("maize"), context, with_yield=True)
        decisions = {decision.domain: decision for decision in report.decisions}
        self.assertEqual(decisions["Soils"].score, 100)
        self.assertEqual(decisions["Pests"].score, 70)
        self.assertIn("gross margin", " ".join(decisions["Economics"].evidence).lower())
        self.assertIn("validated integrated pest management", decisions["Pests"].recommendation)

    def test_image_disease_probability_is_included_as_screening_evidence(self) -> None:
        report = self.make_report(make_field("tomato"), disease_probability=0.82)
        disease = next(item for item in report.decisions if item.domain == "Disease")
        self.assertEqual(disease.score, 82)
        self.assertIn("not a diagnosis", disease.evidence[0])

    def test_invalid_crop_and_optional_measurements_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported crop"):
            self.make_report(make_field("not-a-crop"))
        with self.assertRaisesRegex(ValueError, "Soil pH"):
            self.make_report(make_field("tomato"), DecisionContext(soil_ph=15))
        with self.assertRaisesRegex(ValueError, "Pest-affected area"):
            self.make_report(
                make_field("tomato"),
                DecisionContext(pest_observed=True, pest_affected_area_pct=101),
            )

    def test_tomato_image_probability_cannot_be_applied_to_other_crops(self) -> None:
        with self.assertRaisesRegex(ValueError, "only for tomato"):
            self.make_report(make_field("wheat"), disease_probability=0.9)


if __name__ == "__main__":
    unittest.main()
