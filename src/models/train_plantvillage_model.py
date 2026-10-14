"""Download a small official PlantVillage tomato subset and train a binary image model."""

from __future__ import annotations

import csv
import json
import random
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.models.plant_disease_model import (
    DEFAULT_MODEL_PATH,
    image_features,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT_DIR / "data" / "processed" / "plantvillage_tomato_subset.csv"
METRICS_PATH = ROOT_DIR / "models" / "artifacts" / "plantvillage_tomato_metrics.json"
DATASET_URL = "https://huggingface.co/datasets/mohanty/PlantVillage"
SPLITS_URL = f"{DATASET_URL}/raw/main/splits/color_{{split}}.txt"
IMAGE_URL_BASE = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/"
LICENSE = "CC BY-SA 3.0"
RANDOM_SEED = 42
HEALTHY_TRAIN_COUNT = 50
HEALTHY_TEST_COUNT = 20
DISEASED_TRAIN_PER_CLASS = 5
DISEASED_TEST_PER_CLASS = 2


def _download_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "AgriShield-ML-demo/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Could not download public dataset metadata from {url}: {error}") from error


def _read_split(split: str) -> list[str]:
    paths = [line.strip() for line in _download_text(SPLITS_URL.format(split=split)).splitlines()]
    tomato_paths = [
        path
        for path in paths
        if len(path.split("/")) == 4 and path.split("/")[2].startswith("Tomato___")
    ]
    if not tomato_paths:
        raise RuntimeError(f"The PlantVillage {split} split contains no tomato images.")
    return tomato_paths


def _select_paths(paths: list[str], split: str) -> list[dict[str, Any]]:
    by_class: dict[str, list[str]] = {}
    for path in paths:
        label = path.split("/")[2]
        by_class.setdefault(label, []).append(path)

    randomizer = random.Random(RANDOM_SEED + (0 if split == "train" else 1))
    healthy_class = "Tomato___healthy"
    healthy_count = HEALTHY_TRAIN_COUNT if split == "train" else HEALTHY_TEST_COUNT
    disease_count = DISEASED_TRAIN_PER_CLASS if split == "train" else DISEASED_TEST_PER_CLASS
    if len(by_class.get(healthy_class, [])) < healthy_count:
        raise RuntimeError(f"Not enough healthy tomato images in the official {split} split.")

    selected: list[dict[str, Any]] = [
        {"path": path, "class_name": healthy_class, "target": 0, "split": split}
        for path in randomizer.sample(sorted(by_class[healthy_class]), healthy_count)
    ]
    disease_classes = sorted(
        name for name in by_class if name.startswith("Tomato___") and name != healthy_class
    )
    if not disease_classes:
        raise RuntimeError(f"No tomato disease classes found in the official {split} split.")
    for disease_class in disease_classes:
        class_paths = sorted(by_class[disease_class])
        if len(class_paths) < disease_count:
            raise RuntimeError(
                f"Not enough images for {disease_class} in the official {split} split."
            )
        selected.extend(
            {
                "path": path,
                "class_name": disease_class,
                "target": 1,
                "split": split,
            }
            for path in randomizer.sample(class_paths, disease_count)
        )
    return selected


def _image_url(source_path: str) -> str:
    return IMAGE_URL_BASE + quote(source_path, safe="/")


def _download_image(entry: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    url = _image_url(entry["path"])
    request = urllib.request.Request(url, headers={"User-Agent": "AgriShield-ML-demo/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            content = response.read(15 * 1024 * 1024 + 1)
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Could not download public image {url}: {error}") from error
    if len(content) > 15 * 1024 * 1024:
        raise RuntimeError(f"Image exceeds the 15 MB safety limit: {url}")
    return entry, content


def _manifest_row(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": Path(entry["path"]).name,
        "source_path": entry["path"],
        "image_url": _image_url(entry["path"]),
        "crop": "tomato",
        "source_class": entry["class_name"],
        "target": entry["target"],
        "target_name": "diseased" if entry["target"] else "healthy",
        "split": entry["split"],
        "leaf_id": Path(entry["path"]).name.split("___", maxsplit=1)[0],
        "dataset": "PlantVillage",
        "dataset_url": DATASET_URL,
        "license": LICENSE,
    }


def train() -> dict[str, Any]:
    entries = _select_paths(_read_split("train"), "train")
    entries.extend(_select_paths(_read_split("test"), "test"))
    entries.sort(key=lambda item: (item["split"], item["target"], item["path"]))

    image_contents: dict[str, bytes] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_download_image, entry) for entry in entries]
        for future in as_completed(futures):
            entry, content = future.result()
            image_contents[entry["path"]] = content

    manifest_rows = [_manifest_row(entry) for entry in entries]
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    train_entries = [entry for entry in entries if entry["split"] == "train"]
    test_entries = [entry for entry in entries if entry["split"] == "test"]
    x_train = np.vstack([image_features(image_contents[item["path"]]) for item in train_entries])
    y_train = np.asarray([item["target"] for item in train_entries], dtype=np.int8)
    x_test = np.vstack([image_features(image_contents[item["path"]]) for item in test_entries])
    y_test = np.asarray([item["target"] for item in test_entries], dtype=np.int8)

    classifier = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            estimator=SVC(
                kernel="rbf",
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
            method="sigmoid",
            cv=3,
        ),
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    test_probabilities = classifier.predict_proba(x_test)[:, list(classifier.classes_).index(1)]

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["healthy", "diseased"],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "dataset": "PlantVillage tomato color images",
        "dataset_url": DATASET_URL,
        "license": LICENSE,
        "seed": RANDOM_SEED,
        "train_rows": int(len(train_entries)),
        "test_rows": int(len(test_entries)),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "test_disease_brier_score": float(np.mean((test_probabilities - y_test) ** 2)),
        "classification_report": report,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope_note": (
            "Binary tomato leaf-image screening on a small, controlled-background subset. "
            "Not a field-level disease diagnosis and not validated on farmer photographs."
        ),
    }

    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(
        {
            "model": classifier,
            "metadata": {
                "dataset": metrics["dataset"],
                "dataset_url": DATASET_URL,
                "license": LICENSE,
                "positive_class": "diseased",
                "input_feature_count": int(x_train.shape[1]),
                "train_rows": int(len(train_entries)),
                "test_rows": int(len(test_entries)),
                "random_seed": RANDOM_SEED,
            },
        },
        DEFAULT_MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    metrics = train()
    print(
        f"Trained on {metrics['train_rows']} official train images; "
        f"held-out accuracy={metrics['test_accuracy']:.3f}, "
        f"balanced accuracy={metrics['test_balanced_accuracy']:.3f}."
    )
    print(f"Labeled manifest: {MANIFEST_PATH}")
    print(f"Model: {DEFAULT_MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
