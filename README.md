# AgriShield

Rule-based agricultural field assessment and decision support.

## Run the prototype

Use Python 3.11 or 3.12, install the dependencies, and launch Streamlit from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run dasboard/app.py
```

The dashboard accepts crop and growth stage, soil moisture, weather, NDVI, soil-test N/P/K, field area,
symptom observations, and an estimated water/energy cost rate. It evaluates the YAML rules in
`decision_rules/`, simulates feasible irrigation scenarios, and ranks them by risk, water use, and
estimated cost. Scenario priority weights can be adjusted in the form.

## Local open-source crop assistant

The Streamlit app includes an optional chat assistant backed by Ollama and Qwen2.5 3B (Apache-2.0).
Install Ollama, then download the model and start the local Ollama service:

```powershell
ollama pull qwen2.5:3b
ollama serve
python -m streamlit run dasboard/app.py
```

The assistant sends requests only to the local Ollama endpoint (`http://127.0.0.1:11434` by default)
and retrieves relevant Markdown guidance from `knowledgebase/crops/decision_intelligence/`, the
selected crop profile, and the latest field assessment. The latest assessment is provided as chat
context; chat history is held in Streamlit session state. If Ollama or the model is unavailable, the
app shows an explicit error and setup guidance rather than generating a fallback answer.

Optional configuration:

```powershell
$env:AGUARD_OLLAMA_URL = "http://127.0.0.1:11434"
$env:AGUARD_OLLAMA_MODEL = "qwen2.5:3b"
```

LLM responses can be wrong. The assistant is instructed not to invent diagnoses, prices, thresholds,
or application rates, but always verify agronomic recommendations with local qualified guidance.

Run the rule-engine checks with:

```powershell
python -m unittest discover -s tests
```

## Crop decision-rule dataset

Generate the crop-expanded, traceable CSV dataset with:

```powershell
python -m src.recommendation.export_crop_rule_dataset
```

This writes `data/processed/crop_decision_rules_with_demo_scenarios.csv` from the 12 crop profiles and the active YAML
rules/scenarios. It includes crop profiles; climate, stress, disease-screening, fertilizer, and
irrigation-screening rules; and irrigation recommendation scenarios. Crop-scoped rules are only
expanded for their matching crop. Each row retains its rule ID, conditions, score, action, source
file, and a note about whether it is metadata, a configurable screening rule, or a demonstration
scenario.

The export also contains one clearly marked synthetic example per crop for disease scouting,
economics, and sustainability. These records demonstrate input/output shapes only: they have no
assessment score, are excluded from live decision assessments, and must not be treated as observed
farm data or local crop benchmarks. Replace their assumptions with verified observations before using
them for analysis.

This is a structured rule dataset, not labeled observations for ML training. The current sources do
not define exhaustive crop-specific pathogen catalogs, disease labels, or fertilizer application
rates; disease entries are broad screening rules and must not be interpreted as diagnoses or treatment
instructions. Update the YAML source rules and regenerate the CSV to keep it synchronized.

## Trained tomato-leaf disease model

Train a small, reproducible binary image classifier and create its labeled CSV manifest with:

```powershell
python -m src.models.train_plantvillage_model
```

The training script downloads a balanced sample of tomato leaf images from the official PlantVillage
train/test lists, preserving the published split. It writes the URL-backed labeled subset to
`data/processed/plantvillage_tomato_subset.csv`, the trained model to
`models/artifacts/plantvillage_tomato.joblib`, and held-out metrics to
`models/artifacts/plantvillage_tomato_metrics.json`. The dashboard uses this model only when a tomato
leaf image is uploaded. Disease is not scored from the weather/symptom placeholder rules.

PlantVillage is a controlled-background image dataset, not field observations. The small classifier
screens tomato images as healthy/diseased and is not calibrated or validated for farmer photographs;
it does not predict water, climate, crop-stage, or nutrient risks. Those risk scores continue to use
rules; transparent irrigation, stress, and yield prototype estimates are described below. Consult an
agronomist before making crop protection decisions.

PlantVillage data are licensed CC BY-SA 3.0. Attribution: Mohanty, Hughes, and Salathé (2016),
"Using deep learning for image-based plant disease detection," Frontiers in Plant Science,
DOI: 10.3389/fpls.2016.01419. Source: [PlantVillage dataset](https://huggingface.co/datasets/mohanty/PlantVillage).

## Rule and model assumptions

Rules use a list of `{field, operator, value}` conditions; all conditions in a rule must match.
Non-disease risk scores and thresholds are editable screening defaults, not probabilities or validated
crop-specific recommendations. The soil nutrient thresholds are generic placeholders and require local
units, calibration, and agronomic guidance. Reported symptoms prompt inspection only and do not confirm
disease.

The irrigation what-if scenarios use editable demonstration amounts and fixed water-stress score
reductions in `decision_rules/recommendation_rules.yaml`. Water use is calculated as irrigation depth
times field area (`1 mm * 1 ha = 10,000 L`); estimated cost uses the rate entered by the user. These
effects are not calibrated yield or soil-water predictions. Review assumptions against local field data
and an agronomist before operational use.

## Agronomic prototype estimators

`src/models/irrigation_model.py`, `src/models/stress_model.py`, and `src/models/yield_model.py`
provide transparent estimates for every crop in `knowledgebase/crops/metadata/crop_metadata.yml`.
The dashboard shows these estimates alongside the existing YAML screening rules and scenario
simulation:

- Irrigation estimates a refill depth using soil moisture as percent of plant-available water,
  user-entered root depth and available-water capacity, irrigation efficiency, and probability-weighted
  effective forecast rainfall. Trigger values and other defaults are demonstration settings, not
  universal recommendations.
- Crop stress combines soil-water, NDVI, and temperature indicators into a 0-100 screening score.
  Its default temperature thresholds and component weights are generic and should be locally calibrated.
- Yield is a linear scenario, not a trained forecast. It is only calculated when a locally supported
  baseline yield and an assumed yield loss at maximum stress are supplied.

These calculations are not trained or field-validated agronomic models. Soil sensor interpretation,
root depth, available-water capacity, temperature thresholds, and yield response vary by location,
soil, cultivar, and management. Validate inputs and outputs with local observations and an agronomist
before making operational decisions.

## Multi-domain decision intelligence

The dashboard synthesizes an explainable decision view for climate, crop stress, disease, economics,
fertilizers, irrigation, pests, soils, sustainability, and yields for every configured crop. It combines
the existing rule assessment and agronomic scenario estimators with optional field observations:

- Enter recent soil pH, electrical conductivity, and organic matter for more than the partial soil view.
- Record pest scouting and affected area; the system recommends scouting/IPM confirmation, not pesticide
  selection or an application rate.
- Add locally supported baseline yield, market price, and production cost to calculate yield and gross
  margin scenarios. Zero/blank economics inputs leave that domain unassessed.
- A tomato leaf photo contributes a disease-image screening score only for tomato. Other crops use
  reported symptoms and environmental favorability; neither confirms disease.

The overall attention priority ranks available indicators for review. Domain scores are screening
indices, not probabilities. The soil, climate, nutrient, pest, yield, and economic assumptions are
generic or user-supplied; missing measurements are reported as unassessed rather than treated as safe.
Timestamp and localize price/cost inputs, use current soil analyses, and validate thresholds and actions
with locally authoritative agronomic guidance. The system does not prescribe chemicals or fertilizer
doses. The current screen uses a single assessment snapshot; it does not yet calculate temporal trends,
historical climate risk, affected zones, or validated crop-specific pest/disease probabilities.
