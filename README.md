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

Run the rule-engine checks with:

```powershell
python -m unittest discover -s tests
```

## Rule and model assumptions

Rules use a list of `{field, operator, value}` conditions; all conditions in a rule must match.
Risk scores and thresholds are editable screening defaults, not probabilities or validated crop-specific
recommendations. The soil nutrient thresholds are generic placeholders and require local units,
calibration, and agronomic guidance. Reported symptoms prompt inspection only and do not confirm disease.

The irrigation what-if scenarios use editable demonstration amounts and fixed water-stress score
reductions in `decision_rules/recommendation_rules.yaml`. Water use is calculated as irrigation depth
times field area (`1 mm * 1 ha = 10,000 L`); estimated cost uses the rate entered by the user. These
effects are not calibrated yield or soil-water predictions. Review assumptions against local field data
and an agronomist before operational use.
