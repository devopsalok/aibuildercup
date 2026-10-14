"""Streamlit interface for the rule-based AgriGuard decision workflow."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.plant_disease_model import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    load_disease_model,
    predict_disease_image,
)
from src.models.irrigation_model import estimate_irrigation  # noqa: E402
from src.models.stress_model import estimate_stress  # noqa: E402
from src.models.yield_model import estimate_yield  # noqa: E402
from src.assistant.local_crop_chat import CropChatError, ask_local_crop_assistant  # noqa: E402
from src.recommendation.engine import (  # noqa: E402
    FieldData,
    RuleConfigurationError,
    RuleEngine,
    load_crop_metadata,
    risk_level,
)
from src.recommendation.decision_intelligence import (  # noqa: E402
    DecisionContext,
    build_decision_intelligence,
)


st.set_page_config(page_title="AgriGuard | Field decision support", layout="wide")
st.session_state.setdefault("agri_chat_messages", [])
st.session_state.setdefault("agri_last_assessment_context", "")
st.session_state.setdefault("agri_last_assessment_crop", "")


@st.cache_resource
def load_engine() -> RuleEngine:
    return RuleEngine.from_directory()


@st.cache_data
def load_crops() -> dict:
    return load_crop_metadata()


@st.cache_resource
def load_tomato_disease_model() -> dict:
    return load_disease_model(DEFAULT_MODEL_PATH)


st.title("AgriGuard | Decision support")
st.caption("Assess field risks, compare irrigation scenarios, and review an explainable action.")
# st.info(
#     "Tomato leaf disease screening uses a trained image classifier. Water, climate, crop-stress, and "
#     "nutrient risk scores still use editable screening rules. Separate irrigation, stress, and yield "
#     "estimators are transparent prototypes, not locally validated or trained models. The image "
#     "classifier uses a small controlled-background dataset and is not a field diagnosis."
# )

try:
    engine = load_engine()
    crops = load_crops()
except RuleConfigurationError as error:
    st.error(f"Unable to start the decision engine: {error}")
    st.stop()

st.header("Field data input")
crop_ids = list(crops)
stage_options = sorted(
    {
        "germination",
        "seedling",
        "vegetative",
        "tillering",
        "crown_root_initiation",
        "panicle_initiation",
        "tasseling",
        "silking",
        "flowering",
        "boll_development",
        "grain_filling",
        "fruiting",
        "maturity",
    }
    | {
        stage
        for crop in crops.values()
        for stage in crop.get("critical_stages", [])
    }
)

with st.form("field_data"):
    left, right = st.columns(2)
    with left:
        crop = st.selectbox("Crop", crop_ids, format_func=lambda value: value.title())
        leaf_image = st.file_uploader(
            "Tomato leaf image (optional, trained disease model)",
            type=["jpg", "jpeg", "png"],
        )
        growth_stage = st.selectbox(
            "Growth stage",
            stage_options,
            format_func=lambda value: value.replace("_", " ").title(),
        )
        soil_moisture_pct = st.number_input(
            "Soil moisture (% of plant-available water)",
            min_value=0.0,
            max_value=100.0,
            value=32.0,
            step=1.0,
        )
        root_depth_m = st.number_input(
            "Effective root depth (m)", min_value=0.05, max_value=5.0, value=0.6, step=0.1
        )
        available_water_capacity = st.number_input(
            "Available water capacity (mm/m)",
            min_value=1.0,
            max_value=1000.0,
            value=150.0,
            step=10.0,
        )
        ndvi = st.number_input("NDVI (-1 to 1)", min_value=-1.0, max_value=1.0, value=0.62, step=0.01)
        symptoms_reported = st.checkbox("Field symptoms observed/reported", value=False)
        area_ha = st.number_input("Field area (hectares)", min_value=0.001, value=1.0, step=0.1)
    with right:
        temperature_c = st.number_input(
            "Air temperature (C)", min_value=-30.0, max_value=60.0, value=28.0, step=0.5
        )
        humidity_pct = st.number_input(
            "Relative humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=1.0
        )
        rainfall_probability = st.slider(
            "Rainfall probability in next 24h", min_value=0.0, max_value=1.0, value=0.3, step=0.05
        )
        rainfall_mm = st.number_input(
            "Expected rainfall in next 24h (mm)", min_value=0.0, max_value=2000.0, value=0.0, step=1.0
        )
        st.markdown("**Soil test nutrients (mg/kg)**")
        nutrient_columns = st.columns(3)
        nitrogen = nutrient_columns[0].number_input("N", min_value=0.0, value=35.0, step=1.0)
        phosphorus = nutrient_columns[1].number_input("P", min_value=0.0, value=18.0, step=1.0)
        potassium = nutrient_columns[2].number_input("K", min_value=0.0, value=140.0, step=1.0)
        with st.expander("Optional soil test details"):
            soil_test_available = st.checkbox("I have a recent soil test", value=False)
            soil_ph = st.number_input(
                "Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1
            )
            soil_ec_ds_m = st.number_input(
                "Soil electrical conductivity (dS/m)",
                min_value=0.0,
                max_value=100.0,
                value=1.0,
                step=0.1,
            )
            organic_matter_pct = st.number_input(
                "Soil organic matter (%)",
                min_value=0.0,
                max_value=100.0,
                value=2.0,
                step=0.1,
            )
        with st.expander("Optional pest scouting"):
            pest_reported = st.checkbox("Pest signs observed in the field", value=False)
            affected_area_value = st.slider(
                "Estimated affected area (%)",
                0,
                100,
                10,
            )
            pest_affected_area_pct = affected_area_value if pest_reported else None
        water_cost = st.number_input(
            "Water/energy cost per mm per hectare (cost units)",
            min_value=0.0,
            value=0.1,
            step=0.05,
        )
        irrigation_available = st.checkbox("Irrigation is available", value=True)

    with st.expander("Optional yield scenario (requires local calibration)"):
        baseline_yield_kg_ha = st.number_input(
            "Expected yield without current stress (kg/ha)",
            min_value=0.0,
            value=0.0,
            step=100.0,
            help="Enter a local historical or agronomist-provided yield. Leave at 0 to skip this estimate.",
        )
        yield_loss_at_full_stress_pct = st.number_input(
            "Expected yield loss at a stress score of 100 (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=5.0,
            help="This is a user-supplied linear scenario assumption, not a learned crop coefficient.",
        )
        market_price_per_kg = st.number_input(
            "Expected market price per kg (cost units)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Enter a recent local price; leave at 0 to skip the economic estimate.",
        )
        production_cost_per_ha = st.number_input(
            "Estimated total production cost per hectare (cost units)",
            min_value=0.0,
            value=0.0,
            step=100.0,
            help="Enter the estimated full-season cost; leave at 0 to skip the economic estimate.",
        )

    st.markdown("**Scenario ranking priorities**")
    weight_columns = st.columns(3)
    risk_weight = weight_columns[0].slider("Risk", 0.0, 1.0, 0.8, 0.05)
    water_weight = weight_columns[1].slider("Water use", 0.0, 1.0, 0.1, 0.05)
    cost_weight = weight_columns[2].slider("Cost", 0.0, 1.0, 0.1, 0.05)
    submitted = st.form_submit_button("Assess field and compare scenarios", type="primary")

if submitted:
    if risk_weight + water_weight + cost_weight == 0:
        st.error("Choose at least one positive scenario ranking priority.")
        st.stop()
    try:
        field = FieldData(
            crop=crop,
            growth_stage=growth_stage,
            soil_moisture_pct=soil_moisture_pct,
            temperature_c=temperature_c,
            rainfall_probability_24h=rainfall_probability,
            rainfall_mm_24h=rainfall_mm,
            humidity_pct=humidity_pct,
            ndvi=ndvi,
            nitrogen_mg_kg=nitrogen,
            phosphorus_mg_kg=phosphorus,
            potassium_mg_kg=potassium,
            symptoms_reported=symptoms_reported,
            area_ha=area_ha,
            water_cost_per_mm_ha=water_cost,
        )
        assessment = engine.assess(field)
        stress_prediction = estimate_stress(field)
        irrigation_prediction = estimate_irrigation(
            field,
            root_depth_m=root_depth_m,
            available_water_capacity_mm_per_m=available_water_capacity,
            irrigation_available=irrigation_available,
        )
        yield_prediction = (
            estimate_yield(
                crop,
                baseline_yield_kg_ha=baseline_yield_kg_ha,
                stress_score=stress_prediction.score,
                yield_loss_at_full_stress_pct=yield_loss_at_full_stress_pct,
                area_ha=area_ha,
            )
            if baseline_yield_kg_ha > 0
            else None
        )
        disease_prediction = None
        if leaf_image is not None:
            if crop != "tomato":
                raise ValueError("The trained image classifier is tomato-only; select Tomato or remove the image.")
            disease_prediction = predict_disease_image(
                leaf_image.getvalue(), load_tomato_disease_model()
            )
        decision_context = DecisionContext(
            soil_ph=soil_ph if soil_test_available else None,
            soil_ec_ds_m=soil_ec_ds_m if soil_test_available else None,
            organic_matter_pct=organic_matter_pct if soil_test_available else None,
            pest_observed=pest_reported,
            pest_affected_area_pct=pest_affected_area_pct,
            baseline_yield_kg_ha=baseline_yield_kg_ha if baseline_yield_kg_ha > 0 else None,
            yield_loss_at_full_stress_pct=(
                yield_loss_at_full_stress_pct if baseline_yield_kg_ha > 0 else None
            ),
            market_price_per_kg=market_price_per_kg if market_price_per_kg > 0 else None,
            production_cost_per_ha=(
                production_cost_per_ha if production_cost_per_ha > 0 else None
            ),
        )
        decision_report = build_decision_intelligence(
            field,
            assessment,
            stress_prediction,
            irrigation_prediction,
            yield_prediction,
            decision_context,
            disease_probability=(
                float(disease_prediction["disease_probability"])
                if disease_prediction is not None
                else None
            ),
        )
        st.session_state["agri_last_assessment_context"] = "\n".join(
            [
                f"Crop: {crop}; growth stage: {growth_stage}.",
                f"Soil moisture: {soil_moisture_pct:g}% plant-available water; "
                f"temperature: {temperature_c:g} C; rainfall forecast: "
                f"{rainfall_probability:.0%}, {rainfall_mm:g} mm.",
                f"NDVI: {ndvi:g}; field symptoms reported: {symptoms_reported}.",
                *[
                    (
                        f"{decision.domain}: {decision.status}, priority "
                        f"{decision.priority}, indicator {decision.score}; "
                        f"next step: {decision.recommendation}"
                    )
                    for decision in decision_report.decisions
                ],
            ]
        )
        st.session_state["agri_last_assessment_crop"] = crop
        risk_scores = {
            category: score
            for category, score in assessment.scores.items()
            if category != "disease"
        }
        if disease_prediction is not None:
            risk_scores["disease"] = int(round(float(disease_prediction["disease_probability"]) * 100))
        assessment = replace(
            assessment,
            scores=risk_scores,
            findings=tuple(finding for finding in assessment.findings if finding.category != "disease"),
        )
        scenarios = engine.rank_scenarios(
            field,
            assessment,
            irrigation_available=irrigation_available,
            weights={"risk": risk_weight, "water": water_weight, "cost": cost_weight},
        )
    except (FileNotFoundError, RuleConfigurationError, ValueError) as error:
        st.error(f"Assessment could not be completed: {error}")
        st.stop()

    st.header("Decision intelligence")
    st.metric("Overall attention priority", decision_report.overall_priority)
    st.caption(
        "Joined, explainable screening across ten domains. Scores are rule-based indicators or "
        "user-input scenarios—not calibrated probabilities. Missing inputs remain explicitly unassessed."
    )
    priority_order = {"Critical": 0, "High": 1, "Moderate": 2, "Low": 3, "Unassessed": 4}
    decision_rows = [
        {
            "Domain": decision.domain,
            "Status": decision.status,
            "Priority": decision.priority,
            "Indicator (0-100)": decision.score,
            "Recommended next step": decision.recommendation,
            "Evidence": " | ".join(decision.evidence),
        }
        for decision in sorted(
            decision_report.decisions,
            key=lambda item: (priority_order[item.priority], item.domain),
        )
    ]
    st.dataframe(
        pd.DataFrame(decision_rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Recommended next step": st.column_config.TextColumn(width="large"),
            "Evidence": st.column_config.TextColumn(width="large"),
        },
    )

    st.header("Crop-aware agronomic estimates")
    st.caption(
        "These transparent prototype calculations are not trained or locally validated models. "
        "Review their input assumptions with local soil and crop guidance before acting."
    )
    irrigation_column, stress_column, yield_column = st.columns(3)
    irrigation_column.metric(
        "Estimated irrigation",
        f"{irrigation_prediction.recommended_depth_mm:.1f} mm",
        help=irrigation_prediction.reason,
    )
    irrigation_column.caption(
        f"{irrigation_prediction.recommendation}; approximately "
        f"{irrigation_prediction.water_volume_liters:,.0f} L for this field."
    )
    stress_column.metric(
        "Composite crop-stress screen",
        f"{stress_prediction.score}/100",
        stress_prediction.level,
        help=stress_prediction.explanation,
    )
    if yield_prediction is None:
        yield_column.metric("Stress-adjusted yield scenario", "Not calculated")
        yield_column.caption("Enter a locally supported baseline yield to calculate this scenario.")
    else:
        yield_column.metric(
            "Stress-adjusted yield scenario",
            f"{yield_prediction.estimated_yield_kg_ha:,.0f} kg/ha",
            f"-{yield_prediction.estimated_loss_pct:.1f}% vs. baseline",
        )
        yield_column.caption(
            f"Estimated field total: {yield_prediction.estimated_field_yield_kg:,.0f} kg; "
            "uses the user-entered baseline and linear stress-loss assumption."
        )

    st.header("Risk screening results")
    risk_rows = [
        {
            "Risk area": "Tomato leaf disease (model)" if category == "disease"
            else category.replace("_", " ").title(),
            "Score": score,
            "Level": risk_level(score),
        }
        for category, score in assessment.scores.items()
    ]
    st.caption(
        "Disease score (when a tomato leaf image is supplied) is the trained model's estimated "
        "diseased-image probability. Other displayed scores are YAML rule-screening scores, not "
        "learned probabilities."
    )
    risk_frame = pd.DataFrame(risk_rows)
    chart_column, table_column = st.columns([1.5, 1])
    with chart_column:
        risk_values = np.asarray(risk_frame["Score"], dtype=float)
        risk_chart = px.bar(
            risk_frame,
            x="Risk area",
            y=risk_values,
            color="Level",
            category_orders={"Level": ["Low", "Moderate", "High", "Critical"]},
            range_y=[0, 100],
            title="Trained disease-model probability and rule-screening scores",
            labels={"Score": "Model probability / relative rule score (0-100)"},
        )
        st.plotly_chart(risk_chart, width="stretch")
    with table_column:
        st.dataframe(risk_frame, hide_index=True, width="stretch")
        if disease_prediction is None:
            st.info("Tomato disease model: not assessed. Upload a tomato leaf photo to run it.")
        else:
            st.metric(
                "Tomato image: model disease probability",
                f"{float(disease_prediction['disease_probability']):.1%}",
                help=(
                    "This is a model score on a lab-style image, not a calibrated field-disease "
                    "probability or diagnosis."
                ),
            )
            st.caption(
                f"Image classifier result: **{disease_prediction['label']}**. "
                "Confirm suspected disease with a qualified local agronomist."
            )
        if assessment.findings:
            finding_rows = [
                {
                    "Rule": finding.name,
                    "Risk area": finding.category.replace("_", " ").title(),
                    "Priority": finding.priority.title(),
                    "Suggested check": finding.action.replace("_", " ").title(),
                    "Why": " ".join(finding.explanations),
                }
                for finding in assessment.findings
            ]
            st.markdown("**Matched rules and reasons**")
            st.dataframe(pd.DataFrame(finding_rows), hide_index=True, width="stretch")
        else:
            st.success("No configured risk rules were triggered. Continue routine field monitoring.")

    st.header("3. What-if irrigation simulation")
    st.caption(
        "Only options compatible with irrigation availability are shown. "
        "Scenario effects and irrigation amounts come from editable YAML assumptions."
    )
    scenario_rows = [
        {
            "Scenario": scenario.name,
            "Irrigation (mm)": scenario.irrigation_mm,
            "Water (L)": scenario.water_liters,
            "Estimated cost": scenario.estimated_cost,
            "Overall risk": scenario.overall_risk_score,
            "Ranking score (lower is better)": scenario.ranking_score,
        }
        for scenario in scenarios
    ]
    scenario_frame = pd.DataFrame(scenario_rows)
    scenario_chart = px.bar(
        scenario_frame,
        x="Scenario",
        y="Ranking score (lower is better)",
        color="Scenario",
        title="Scenario comparison (lower combined score ranks higher)",
    )
    st.plotly_chart(scenario_chart, width="stretch")
    st.dataframe(scenario_frame, hide_index=True, width="stretch")

    st.header("Decision engine")
    recommended = scenarios[0]
    st.success(f"Recommended option to consider: **{recommended.name}**")
    st.write(recommended.description)
    st.write(
        f"Estimated use: **{recommended.water_liters:,.0f} L** "
        f"({recommended.irrigation_mm:g} mm over {area_ha:g} ha); "
        f"estimated cost: **{recommended.estimated_cost:,.2f} cost units**."
    )

    st.header("Farmer advisory")
    st.markdown(
        "- Recheck soil moisture and the local forecast before carrying out an irrigation action.\n"
        "- Reported symptoms or favorable weather indicate a need to inspect; they do not confirm disease.\n"
        "- Confirm nutrient results and any input decision with crop-, soil-, and region-specific guidance."
    )
    st.caption(
        "Assumptions: screening thresholds in decision_rules/*.yaml and fixed scenario risk reductions "
        "are demonstration defaults, not locally calibrated agronomic models. Water is calculated as "
        "irrigation depth * field area (1 mm over 1 ha = 10,000 L); estimated cost is depth * area * "
        "the user-entered cost rate. Validate thresholds, costs, and scenario responses with local data "
        "and an agronomist before operational use."
    )
else:
    st.info("Enter field observations and submit the form to generate an assessment and scenario ranking.")

st.divider()
st.header("Ask the local AgriGuard assistant")
st.caption(
    "Powered by Ollama and the local Qwen2.5 3B model. Your question and assessment context are "
    "sent only to the Ollama service configured on this machine."
)
st.markdown(
    "To enable it, install Ollama and run `ollama pull qwen2.5:3b`. Ollama must be running locally."
)
for chat_message in st.session_state["agri_chat_messages"]:
    with st.chat_message(chat_message["role"]):
        st.markdown(chat_message["content"])

chat_question = st.chat_input("Ask about this crop, the assessment, or local guidance")
if chat_question:
    st.session_state["agri_chat_messages"].append(
        {"role": "user", "content": chat_question}
    )
    with st.chat_message("user"):
        st.markdown(chat_question)
    with st.chat_message("assistant"):
        with st.spinner("Consulting the local crop guidance..."):
            try:
                answer = ask_local_crop_assistant(
                    chat_question,
                    crop=st.session_state["agri_last_assessment_crop"] or crop,
                    assessment_context=st.session_state["agri_last_assessment_context"],
                    history=st.session_state["agri_chat_messages"][:-1],
                )
            except (CropChatError, FileNotFoundError, ValueError) as error:
                st.error(f"Local assistant unavailable: {error}")
            else:
                st.markdown(answer)
                st.session_state["agri_chat_messages"].append(
                    {"role": "assistant", "content": answer}
                )