# Insurance Claims Risk Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

An end-to-end machine learning pipeline for auto insurance claims that tackles two business problems in parallel: **claim severity prediction** (how much will a claim cost?) and **fraud detection** (is this claim fraudulent?). The project investigates whether the two problems share underlying drivers — and finds, through dual-method explainability, that they largely do not.

---

## Dataset

`insurance_claims.csv` — 1,000 auto insurance claims, 40 raw features

| Target | Type | Description |
|---|---|---|
| `total_claim_amount` | Regression | Expected payout — used for reserving |
| `fraud_reported` | Classification | Whether a claim is fraudulent |

Fraud rate: 24.7% (imbalanced but not extreme).

---

## Project Structure

```
Insurance_Project/
├── Data/                              # Raw and processed data
├── Outputs/                           # Model outputs and plots
├── eda.ipynb                          # Exploratory data analysis
├── feature_engineering.ipynb          # Feature creation and leakage-safe splits
├── severity_model.ipynb               # Regression model for claim amounts
├── fraud_model.ipynb                  # Classification model for fraud detection
├── explainability.ipynb               # SHAP feature importance analysis
├── business_impact..ipynb             # Dollar-value translation of results
└── insurance_risk_pipeline_report.md  # Full written report
```

---

## Pipeline Overview

### 1. Exploratory Data Analysis

- `total_claim_amount` is **bimodal** — a log-transform made skewness *worse* (-0.59 → -1.66), so it was not applied
- Bimodality is driven by `incident_severity` and `incident_type`, with a threshold effect from number of vehicles (1 vehicle → ~$46K; 2+ vehicles → ~$61–64K; no further change beyond 2)
- `incident_severity` is the strongest univariate fraud predictor: Major Damage claims have a **60.5% fraud rate** vs. 6.7–12.9% for all other severity tiers
- Several intuitive hypotheses were tested and rejected: policy-state mismatch (data artifact — only 9 matching rows), customer tenure, witness count, and police report availability all showed no meaningful signal

### 2. Feature Engineering

Two separate, **leakage-safe** feature sets were built:

- **Severity model** — excludes `injury_claim`, `property_claim`, and `vehicle_claim` (these sum to `total_claim_amount` by definition)
- **Fraud model** — retains claim component breakdowns since they don't leak the fraud label

Engineered features:
- `multi_vehicle_incident` — binary flag capturing the threshold effect found in EDA
- `severity_vehicle_interaction` — combined category of `incident_severity` × `multi_vehicle_incident`

**Dataset artifact caught and removed:** `insured_hobbies` contained a synthetic-data quirk where `chess` (82.6% fraud rate) and `cross-fit` (74.3%) were deterministically tied to fraud — far outside the natural 9–30% range for other hobbies. This was excluded from the fraud feature set entirely.

Train/test split was performed **before encoding** to prevent leakage from category frequency statistics.

### 3. Severity Model

Two model families were compared:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Gamma GLM (log link) | $14,267 | $10,647 | 0.694 |
| XGBoost (tuned) | $14,278 | $10,762 | 0.694 |

Both models, built on completely different statistical foundations, converge on the same performance ceiling (R² ≈ 0.69) — strong evidence this reflects genuine signal limits in the data. A 30-iteration hyperparameter search on XGBoost produced no improvement.

Final GLM formula: `total_claim_amount ~ incident_type + collision_type + bodily_injuries + witnesses + age`

**Business impact — reserving accuracy vs. naive baseline:**

| | Naive flat-average | Model |
|---|---|---|
| RMSE | $25,928 | $14,753 |
| Total reserving error (200 claims) | $4,013,666 | $2,187,140 |

→ **45.5% reduction** in total reserving error

### 4. Fraud Model

Three approaches were compared:

| Model | PR-AUC | F1 | Recall (fraud) | Precision (fraud) |
|---|---|---|---|---|
| Logistic Regression (scaled) | 0.528 | 0.63 | 71% | 56% |
| Random Forest | 0.515 | 0.56 | 49% | 65% |
| Isolation Forest (unsupervised) | — | 0.18 | — | — |

> Note: An initial Logistic Regression run without feature scaling performed *worse than random* (PR-AUC 0.29). Adding `StandardScaler` fixed this completely.

**Threshold tuning — dollar impact:**

| | Threshold 0.55 (F1-optimal) | Threshold 0.20 (high-recall) |
|---|---|---|
| Fraud caught ($) | $2,074,560 | $2,625,060 |
| Fraud missed ($) | $923,910 | $373,410 |
| False alarms (count) | 21 | 94 |
| Review cost (@ $500/review) | $10,500 | $47,000 |
| **Net benefit vs. no model** | $2,064,060 | **$2,578,060** |

The high-recall threshold (0.20) produces a larger net dollar benefit despite quadrupling false alarms, because the average claim (~$53K) dwarfs the $500 manual review cost. When missed fraud costs ~100× more than a false positive, **optimising for recall outperforms optimising for F1**.

### 5. Explainability (SHAP)

SHAP was used to directly test whether severity and fraud share root causes:

| Model | Top driver | Mean |SHAP| |
|---|---|---|
| Severity (XGBoost, TreeExplainer) | `collision_type_Unknown` | 16,464 (15× next feature) |
| Fraud (Logistic Regression, LinearExplainer) | `incident_severity_Major Damage` | 0.59 (2× next feature) |

Only `months_as_customer` and `age` appear in both models' top 10 — and neither ranks in the top 3 for either model. The dominant drivers are **completely disjoint**: severity is governed by incident mechanics (what physically happened); fraud is governed by claim categorisation and component size (how the claim was recorded).

**Conclusion:** a combined "risk score" would obscure this. Separate models are the correct architecture.

---

## Key Results Summary

| Stage | Outcome |
|---|---|
| EDA | Identified bimodal severity distribution and dominant fraud predictor; ruled out multiple plausible hypotheses with evidence |
| Feature engineering | Built leakage-safe feature sets; caught and removed a redundant feature and a dataset artifact |
| Severity model | GLM and XGBoost converge at R² ≈ 0.69; performance ceiling explained |
| Fraud model | Logistic Regression (scaled) outperforms Random Forest on recall; threshold tuning translated into real dollar tradeoffs |
| Explainability | SHAP confirms severity and fraud are driven by largely disjoint feature sets |
| Business impact | 45.5% reserving error reduction; $2.58M fraud detection net benefit at high-recall threshold |

---

## Limitations

- **Sample size:** 1,000 total claims (800 train / 200 test) is modest for fraud detection — only ~49 fraud cases in the test set
- **Synthetic data:** the `insured_hobbies` artifact is a known quirk in this teaching dataset; real deployments should validate feature importance against domain plausibility
- **Unexplained variance:** ~31% of claim severity variance is likely irreducible noise in this dataset
- **Business impact figures** rely on stated assumptions ($500/review cost, naive baseline definition) and are directionally informative, not validated forecasts
- **Detection-intensity confound:** `incident_severity` dominating the fraud model may partly reflect adjusters scrutinising large claims more closely, not purely a causal relationship

---

## Tech Stack

- Python (pandas, numpy, scikit-learn, statsmodels, xgboost)
- SHAP for model explainability
- Matplotlib / Seaborn for visualisation
- Jupyter Notebooks

---

## Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/anushaanilsingh/Insurance_Project.git
cd Insurance_Project
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn statsmodels xgboost shap matplotlib seaborn jupyter
```

**3. Run the notebooks in order**
```
eda.ipynb
feature_engineering.ipynb
severity_model.ipynb
fraud_model.ipynb
explainability.ipynb
business_impact..ipynb
```

> The notebooks are designed to be run sequentially — each one builds on outputs from the previous stage.

## Author

**Anusha Anil Singh**
- GitHub: [@anushaanilsingh](https://github.com/anushaanilsingh)

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) — you are free to use, modify, and distribute this project with attribution.

---

## Acknowledgements

- Dataset sourced from a commonly used synthetic auto insurance teaching dataset
- Actuarial modeling approach (Gamma GLM, log link) inspired by conventions in the motor claims pricing literature (e.g. freMTPL2sev benchmark dataset)
