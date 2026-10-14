"""Image features and inference for the small PlantVillage tomato model."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "artifacts" / "plantvillage_tomato.joblib"
IMAGE_SIZE = 64
CELL_SIZE = 8
ORIENTATION_BINS = 9
COLOR_BINS = 8
FEATURE_COUNT = (IMAGE_SIZE // CELL_SIZE) ** 2 * ORIENTATION_BINS + 3 * COLOR_BINS


def image_features(image_bytes: bytes) -> np.ndarray:
    """Return a fixed-size HOG and color-histogram representation for a leaf image."""
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            rgb = np.asarray(source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)
    except (OSError, ValueError) as error:
        raise ValueError("The uploaded file is not a readable image.") from error

    gray = rgb.mean(axis=2) / 255.0
    gradient_y, gradient_x = np.gradient(gray)
    magnitude = np.hypot(gradient_x, gradient_y)
    orientation = np.mod(np.degrees(np.arctan2(gradient_y, gradient_x)), 180.0)
    cell_histograms: list[float] = []
    bin_edges = np.linspace(0.0, 180.0, ORIENTATION_BINS + 1)

    for y in range(0, IMAGE_SIZE, CELL_SIZE):
        for x in range(0, IMAGE_SIZE, CELL_SIZE):
            cell_orientation = orientation[y : y + CELL_SIZE, x : x + CELL_SIZE]
            cell_magnitude = magnitude[y : y + CELL_SIZE, x : x + CELL_SIZE]
            histogram, _ = np.histogram(
                cell_orientation, bins=bin_edges, weights=cell_magnitude
            )
            cell_histograms.extend(histogram.astype(np.float32).tolist())

    color_histograms: list[float] = []
    for channel in range(3):
        histogram, _ = np.histogram(rgb[:, :, channel], bins=COLOR_BINS, range=(0, 256))
        normalized = histogram.astype(np.float32) / float(IMAGE_SIZE * IMAGE_SIZE)
        color_histograms.extend(normalized.tolist())

    return np.asarray(cell_histograms + color_histograms, dtype=np.float32)


def load_disease_model(path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained tomato disease model not found at {path}. "
            "Run python -m src.models.train_plantvillage_model first."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "metadata" not in bundle:
        raise ValueError(f"Model artifact has an invalid format: {path}")
    return bundle


def predict_disease_image(image_bytes: bytes, bundle: dict[str, Any]) -> dict[str, float | str]:
    features = image_features(image_bytes).reshape(1, -1)
    model = bundle["model"]
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    disease_index = classes.index(1)
    healthy_index = classes.index(0)
    disease_probability = float(probabilities[disease_index])
    return {
        "label": "diseased" if disease_probability >= 0.5 else "healthy",
        "disease_probability": disease_probability,
        "healthy_probability": float(probabilities[healthy_index]),
    }
