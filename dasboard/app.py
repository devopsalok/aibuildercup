"""Streamlit interface for the rule-based AgriGuard decision workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation.engine import (  # noqa: E402
    FieldData,
    RuleConfigurationError,
    RuleEngine,
    load_crop_metadata,
    risk_level,
)


st.set_page_config(page_title="AgriGuard | Field decision support", layout="wide")


@st.cache_resource
def load_engine() -> RuleEngine:
    return RuleEngine.from_directory()


@st.cache_data
def load_crops() -> dict:
    return load_crop_metadata()


st.title("AgriGuard | Rule-based field decision support")
st.caption("Assess field risks, compare irrigation scenarios, and review an explainable action.")
st.info(
    "This first-phase prototype uses editable screening rules, not trained predictions. "
    "Scores are relative rule scores (0–100), not probabilities, diagnoses, or treatment instructions."
)

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
        growth_stage = st.selectbox(
            "Growth stage",
            stage_options,
            format_func=lambda value: value.replace("_", " ").title(),
        )
        soil_moisture_pct = st.number_input(
            "Soil moisture (%)", min_value=0.0, max_value=100.0, value=32.0, step=1.0
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
        water_cost = st.number_input(
            "Water/energy cost per mm per hectare (cost units)",
            min_value=0.0,
            value=0.1,
            step=0.05,
        )
        irrigation_available = st.checkbox("Irrigation is available", value=True)

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
        scenarios = engine.rank_scenarios(
            field,
            assessment,
            irrigation_available=irrigation_available,
            weights={"risk": risk_weight, "water": water_weight, "cost": cost_weight},
        )
    except (RuleConfigurationError, ValueError) as error:
        st.error(f"Assessment could not be completed: {error}")
        st.stop()

    st.header("Risk assessment")
    risk_rows = [
        {"Risk area": category.replace("_", " ").title(), "Score": score, "Level": risk_level(score)}
        for category, score in assessment.scores.items()
    ]
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
            title="Rule-based risk scores",
            labels={"Score": "Relative score (0-100)"},
        )
        st.plotly_chart(risk_chart, width="stretch")
    with table_column:
        st.dataframe(risk_frame, hide_index=True, width="stretch")
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