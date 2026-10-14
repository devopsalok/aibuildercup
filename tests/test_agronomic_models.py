import unittest

from src.models.irrigation_model import estimate_irrigation
from src.models.stress_model import estimate_stress
from src.models.yield_model import estimate_yield
from src.recommendation.engine import FieldData, load_crop_metadata


def make_field(crop: str, **overrides: object) -> FieldData:
    values: dict[str, object] = {
        "crop": crop,
        "growth_stage": "vegetative",
        "soil_moisture_pct": 20.0,
        "temperature_c": 40.0,
        "rainfall_probability_24h": 0.0,
        "rainfall_mm_24h": 0.0,
        "humidity_pct": 60.0,
        "ndvi": 0.4,
        "nitrogen_mg_kg": 35.0,
        "phosphorus_mg_kg": 18.0,
        "potassium_mg_kg": 140.0,
        "symptoms_reported": False,
        "area_ha": 2.0,
        "water_cost_per_mm_ha": 0.1,
    }
    values.update(overrides)
    return FieldData(**values)  # type: ignore[arg-type]


class AgronomicModelTests(unittest.TestCase):
    def test_estimators_accept_every_configured_crop(self) -> None:
        crops = load_crop_metadata()
        for crop, details in crops.items():
            with self.subTest(crop=crop):
                stages = details.get("critical_stages", [])
                field = make_field(crop, growth_stage=stages[0] if stages else "vegetative")
                stress = estimate_stress(field)
                irrigation = estimate_irrigation(
                    field,
                    root_depth_m=0.6,
                    available_water_capacity_mm_per_m=150,
                )
                yield_scenario = estimate_yield(
                    crop,
                    baseline_yield_kg_ha=5000,
                    stress_score=stress.score,
                    yield_loss_at_full_stress_pct=50,
                    area_ha=field.area_ha,
                )
                self.assertGreaterEqual(stress.score, 0)
                self.assertLessEqual(stress.score, 100)
                self.assertGreaterEqual(irrigation.recommended_depth_mm, 0)
                self.assertGreaterEqual(yield_scenario.estimated_yield_kg_ha, 0)

    def test_irrigation_uses_critical_stage_and_field_area(self) -> None:
        field = make_field(
            "wheat",
            growth_stage="crown_root_initiation",
            soil_moisture_pct=20,
        )
        result = estimate_irrigation(
            field,
            root_depth_m=0.5,
            available_water_capacity_mm_per_m=100,
            irrigation_efficiency_pct=100,
        )
        self.assertEqual(result.trigger_moisture_pct, 35)
        self.assertEqual(result.recommended_depth_mm, 30)
        self.assertEqual(result.water_volume_liters, 600_000)

    def test_forecast_rain_reduces_estimated_irrigation(self) -> None:
        field = make_field(
            "tomato",
            growth_stage="vegetative",
            soil_moisture_pct=20,
            rainfall_probability_24h=0.5,
            rainfall_mm_24h=10,
        )
        result = estimate_irrigation(
            field,
            root_depth_m=0.5,
            available_water_capacity_mm_per_m=100,
            irrigation_efficiency_pct=100,
            rainfall_effectiveness_pct=80,
        )
        self.assertEqual(result.effective_rainfall_mm, 4)
        self.assertEqual(result.recommended_depth_mm, 26)

    def test_no_irrigation_is_applied_when_unavailable_or_not_triggered(self) -> None:
        field = make_field("rice", soil_moisture_pct=50)
        not_triggered = estimate_irrigation(
            field,
            root_depth_m=0.5,
            available_water_capacity_mm_per_m=100,
        )
        unavailable = estimate_irrigation(
            make_field("rice", soil_moisture_pct=10),
            root_depth_m=0.5,
            available_water_capacity_mm_per_m=100,
            irrigation_available=False,
        )
        self.assertEqual(not_triggered.recommended_depth_mm, 0)
        self.assertEqual(unavailable.recommended_depth_mm, 0)

    def test_stress_estimate_is_bounded_and_explainable(self) -> None:
        low = estimate_stress(
            make_field("maize", soil_moisture_pct=80, ndvi=0.8, temperature_c=25)
        )
        high = estimate_stress(
            make_field("maize", soil_moisture_pct=0, ndvi=0.1, temperature_c=50)
        )
        self.assertEqual(low.score, 0)
        self.assertEqual(high.score, 100)
        self.assertIn("not a measured", high.explanation)

    def test_yield_estimate_uses_explicit_linear_assumptions(self) -> None:
        result = estimate_yield(
            "tomato",
            baseline_yield_kg_ha=8000,
            stress_score=40,
            yield_loss_at_full_stress_pct=50,
            area_ha=2,
        )
        self.assertEqual(result.estimated_loss_pct, 20)
        self.assertEqual(result.estimated_yield_kg_ha, 6400)
        self.assertEqual(result.estimated_field_yield_kg, 12_800)

    def test_invalid_crop_and_inputs_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported crop"):
            estimate_yield(
                "unknown",
                baseline_yield_kg_ha=1000,
                stress_score=20,
                yield_loss_at_full_stress_pct=50,
            )
        with self.assertRaisesRegex(ValueError, "Root depth"):
            estimate_irrigation(
                make_field("wheat"),
                root_depth_m=float("nan"),
                available_water_capacity_mm_per_m=100,
            )
        with self.assertRaisesRegex(ValueError, "greater than heat onset"):
            estimate_stress(make_field("wheat"), heat_onset_c=40, severe_heat_c=35)


if __name__ == "__main__":
    unittest.main()
