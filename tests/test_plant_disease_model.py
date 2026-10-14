import unittest
import csv
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from src.models.plant_disease_model import (
    FEATURE_COUNT,
    image_features,
    load_disease_model,
    predict_disease_image,
)

ROOT = Path(__file__).resolve().parents[1]


class PlantDiseaseModelTests(unittest.TestCase):
    def test_image_features_have_fixed_size_and_finite_values(self) -> None:
        stream = BytesIO()
        Image.new("RGB", (80, 60), color=(40, 150, 60)).save(stream, format="PNG")
        features = image_features(stream.getvalue())
        self.assertEqual(features.shape, (FEATURE_COUNT,))
        self.assertTrue(np.isfinite(features).all())

    def test_invalid_image_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a readable image"):
            image_features(b"not an image")

    def test_empty_image_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            image_features(b"")

    def test_public_labeled_subset_preserves_official_splits(self) -> None:
        manifest_path = ROOT / "data" / "processed" / "plantvillage_tomato_subset.csv"
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(rows)
        self.assertEqual({row["target_name"] for row in rows}, {"healthy", "diseased"})
        self.assertEqual({row["split"] for row in rows}, {"train", "test"})
        self.assertTrue(all(row["license"] == "CC BY-SA 3.0" for row in rows))
        leaf_splits: dict[str, set[str]] = {}
        for row in rows:
            leaf_splits.setdefault(row["leaf_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in leaf_splits.values()))

    def test_trained_model_returns_bounded_disease_probability(self) -> None:
        stream = BytesIO()
        Image.new("RGB", (80, 60), color=(40, 150, 60)).save(stream, format="PNG")
        result = predict_disease_image(stream.getvalue(), load_disease_model())
        self.assertIn(result["label"], {"healthy", "diseased"})
        self.assertGreaterEqual(result["disease_probability"], 0.0)
        self.assertLessEqual(result["disease_probability"], 1.0)
        self.assertAlmostEqual(
            result["disease_probability"] + result["healthy_probability"], 1.0, places=5
        )


if __name__ == "__main__":
    unittest.main()
