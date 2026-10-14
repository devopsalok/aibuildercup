import csv
import tempfile
import unittest
from pathlib import Path

from src.recommendation.export_crop_rule_dataset import (
    DATASET_COLUMNS,
    build_crop_rule_rows,
    export_crop_rule_dataset,
)
from src.recommendation.engine import load_crop_metadata


class CropRuleDatasetTests(unittest.TestCase):
    def test_rows_cover_all_crops_and_expected_domains(self) -> None:
        rows = build_crop_rule_rows()
        crop_ids = set(load_crop_metadata())
        self.assertEqual({row["crop"] for row in rows}, crop_ids)
        records_by_crop = {
            crop: [row for row in rows if row["crop"] == crop]
            for crop in crop_ids
        }
        for crop, crop_rows in records_by_crop.items():
            with self.subTest(crop=crop):
                domains = {row["domain"] for row in crop_rows}
                self.assertTrue(
                    {
                        "crop_profile",
                        "climate",
                        "crop_stress",
                        "disease_screening",
                        "fertilizer_screening",
                        "irrigation_screening",
                        "irrigation_recommendation",
                        "economics",
                        "sustainability",
                    }.issubset(domains)
                )

    def test_crop_specific_rules_are_not_expanded_to_other_crops(self) -> None:
        rows = build_crop_rule_rows()
        wheat_only = [
            row for row in rows
            if row["rule_id"] == "IRR-003"
        ]
        self.assertEqual({row["crop"] for row in wheat_only}, {"wheat"})
        rice_rule = [row for row in rows if row["rule_id"] == "IRR-004"]
        self.assertEqual({row["crop"] for row in rice_rule}, {"rice"})

    def test_export_has_stable_columns_and_is_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = export_crop_rule_dataset(Path(temp_dir) / "crop_rules.csv")
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(tuple(reader.fieldnames or ()), DATASET_COLUMNS)
                rows = list(reader)
            self.assertGreater(len(rows), 100)
            self.assertTrue(all(row["recommendation"] for row in rows))
            demo_rows = [
                row for row in rows
                if row["record_type"] == "synthetic_demo_scenario"
            ]
            self.assertEqual(len(demo_rows), 3 * len(load_crop_metadata()))
            self.assertTrue(
                all(
                    row["score"] == ""
                    and "excluded from assessment scores" in row["data_status"]
                    and row["source_file"] == "generated synthetic example"
                    for row in demo_rows
                )
            )
            disease_rows = [row for row in rows if row["domain"] == "disease_screening"]
            self.assertTrue(disease_rows)
            self.assertTrue(
                all(
                    "does not identify or confirm a disease" in row["recommendation"].lower()
                    for row in disease_rows
                )
            )


if __name__ == "__main__":
    unittest.main()
