# Insurance Claims Risk Pipeline: Severity Prediction & Fraud Detection

**Dataset:** `insurance_claims.csv` — 1,000 auto insurance claims, 40 raw features
**Targets:** `total_claim_amount` (severity, regression) and `fraud_reported` (fraud, classification)

---

## 1. Project Overview

This project builds an end-to-end claims risk pipeline on a single dataset, addressing two distinct business problems:

1. **Claim severity prediction** — estimating expected payout for reserving purposes
2. **Fraud detection** — flagging claims likely to be fraudulent before payout

Rather than treating these as two unrelated modeling exercises, the project investigates whether they share underlying drivers — and finds, with evidence from two independent explainability methods, that they largely do not. This shapes the final recommendation: **the two problems should be modeled and acted on separately, not combined into a single risk score.**

---

## 2. Exploratory Data Analysis

### 2.1 Data Cleaning
- Dropped a trailing junk export column (`_c39`)
- Converted `'?'` placeholder values (used in `collision_type`, `property_damage`, `police_report_available`) to proper `NaN`, then filled with an explicit `"Unknown"` category — since missingness here is structural (e.g. a vehicle theft has no `collision_type` by definition), not random

### 2.2 Severity Distribution
- `total_claim_amount` is **bimodal**, not simply right-skewed. A log-transform made skewness *worse* (-0.59 → -1.66), ruling out a naive transform.
- The bimodality is driven primarily by `incident_severity` and `incident_type`, with a secondary, non-linear effect from `number_of_vehicles_involved`: claim cost jumps sharply from 1 to 2+ vehicles (~$46K → ~$61–64K) but barely changes from 2 to 4 vehicles — a threshold effect, not a linear one.
- **Tested and rejected:** claim-amount-vs-severity mismatch as a fraud signal. Fraud rate was nearly identical (10.6% vs. 11.0%) between high- and low-amount "Minor Damage" claims.

### 2.3 Fraud Distribution
- Fraud rate: 24.7% — imbalanced but not extreme.
- **Headline finding:** `incident_severity` is by far the strongest univariate fraud predictor — Major Damage claims show a 60.5% fraud rate vs. 6.7–12.9% for all other severity tiers. No numeric feature correlates with fraud above 0.17.
- **Tested and rejected:**
  - `state_mismatch` (policy state ≠ incident state) — discarded as a data-construction artifact: only 9 of 1,000 rows have matching states, making the apparent signal noise from a tiny sample, not a real pattern.
  - Customer tenure (`months_as_customer`) — no meaningful difference between fraud (208.1) and non-fraud (202.6) cases.
  - Witnesses count and police report availability — both flat across fraud status.

---

## 3. Feature Engineering

- **Dropped identifiers/non-predictive columns:** `policy_number`, `policy_bind_date`, `insured_zip`, `incident_date`, `incident_location`, `auto_model`.
- **Engineered features:**
  - `multi_vehicle_incident` — binary flag (1 vehicle vs. 2+), reflecting the threshold effect found in EDA
  - `severity_vehicle_interaction` — combined category of `incident_severity` × `multi_vehicle_incident`
- **Leakage control — two separate feature sets:**
  - **Severity model:** excludes `injury_claim`, `property_claim`, `vehicle_claim` (these sum to `total_claim_amount` by definition — including them would be predicting a total from its own parts)
  - **Fraud model:** retains claim component breakdowns, since they don't leak the fraud label
- **Known dataset artifact identified and removed:** `insured_hobbies` contains a synthetic-data quirk where `chess` (82.6% fraud rate, n=46) and `cross-fit` (74.3%, n=35) are deterministically tied to fraud — far outside the natural 9–30% range of all other hobbies. This is a known generation artifact in this teaching dataset, not a real-world pattern, and was excluded from the fraud feature set entirely.
- **Train/test split performed before encoding** to prevent information leakage from category frequency statistics. Fraud split was stratified to preserve class balance (24.75%/24.5% train/test).

---

## 4. Severity Model

Two models were compared: a **Gamma GLM** (actuarial-standard baseline, log link) and **XGBoost** (gradient boosting).

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Gamma GLM | 14,267 | 10,647 | 0.694 |
| XGBoost (tuned) | 14,278 | 10,762 | 0.694 |

**Key modeling findings:**
- The two models, built on completely different statistical foundations, converge on the same performance ceiling (R² ≈ 0.69) — strong evidence this reflects genuine signal limits in the data, not a modeling shortfall.
- `multi_vehicle_incident` was found to be **perfectly redundant** with `incident_type` (every "Multi-vehicle Collision" record has `multi_vehicle_incident = 1`; every other incident type has `multi_vehicle_incident = 0`) and was removed from the GLM to avoid unstable, duplicated coefficients.
- `incident_severity` initially appeared significant in isolation, but **lost statistical significance** (all p-values > 0.12) once `incident_type` was properly controlled for — its apparent severity effect was confounded with incident type.
- Final GLM formula: `total_claim_amount ~ incident_type + collision_type + bodily_injuries + witnesses + age`
- A residual correlation check against all unused numeric features found nothing above 0.09 — ruling out a missed continuous driver.
- A 30-iteration hyperparameter search on XGBoost produced no improvement (0.6730 → 0.6937, matching the GLM exactly), confirming the result is data-limited, not tuning-limited.

**Conclusion:** Claim severity is driven almost entirely by `incident_type` (Parked Car and Vehicle Theft claims are dramatically lower than Collision claims) and `collision_type` (the "Unknown" category, itself a proxy for non-collision incident types). The remaining ~31% of variance is best characterized as irreducible/random variation in this dataset rather than a feature engineering gap.

**Methodological note:** the Gamma GLM with a log link used here follows the same modeling convention used in actuarial science literature for motor claim severity — for example, on the freMTPL2sev dataset (French Motor Third-Party Liability claims), a standard benchmark in the actuarial pricing literature. This project does not re-run the analysis on that dataset, since it lacks a fraud label and would require an entirely separate feature set, but the modeling approach (Gamma family, log link, categorical risk factors) was deliberately chosen to match established actuarial practice rather than treated as an arbitrary regression choice.

**Business framing — reserving accuracy vs. naive baseline:**

| | Naive flat-average reserve | Model |
|---|---|---|
| RMSE | $25,928 | $14,753 |
| Total absolute reserving error (200 test claims) | $4,013,666 | $2,187,140 |

The model reduces total reserving error by **45.5%** relative to reserving the historical average for every claim. *(Illustrative extrapolation to a hypothetical 10,000-claim book suggests a directional benefit on the order of tens of millions of dollars — this is a projection, not a validated forecast, and assumes the same claim distribution holds at scale.)*

---

## 5. Fraud Model

Three approaches were compared: **Logistic Regression**, **Random Forest**, and an unsupervised **Isolation Forest** cross-check.

| Model | PR-AUC | F1 | Recall (fraud) | Precision (fraud) |
|---|---|---|---|---|
| Logistic Regression (scaled) | 0.528 | 0.63 | 71% | 56% |
| Random Forest | 0.515 | 0.56 | 49% | 65% |
| Isolation Forest (unsupervised) | — | 0.18 | — | — |

**Key modeling findings:**
- An initial Logistic Regression run performed *worse than random* (PR-AUC 0.29) due to unscaled numeric features distorting gradient-based optimization. Adding `StandardScaler` to numeric inputs fixed this completely.
- Removing the `insured_hobbies` artifact changed Random Forest's PR-AUC only marginally (0.523 → 0.515), confirming the model's real performance was not substantially inflated by the artifact — but it was still removed to avoid reporting a misleading feature importance ranking.
- **Logistic Regression and Random Forest make different tradeoffs**, not just different scores: Logistic Regression catches more fraud (71% recall) at the cost of more false alarms; Random Forest is pickier (65% precision) but misses more fraud. This tradeoff — not a single "winning" model — is the more useful takeaway for a real deployment decision.

**Threshold tuning (Logistic Regression):**

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.20 (high recall) | 0.31 | 0.86 | 0.45 |
| 0.55 (F1-optimal) | 0.61 | 0.67 | 0.64 |

**Business framing — dollar impact at each threshold:**

| | Threshold 0.55 | Threshold 0.20 |
|---|---|---|
| Fraud caught ($) | $2,074,560 | $2,625,060 |
| Fraud missed ($) | $923,910 | $373,410 |
| False alarms (count) | 21 | 94 |
| Review cost (@ $500/review) | $10,500 | $47,000 |
| **Net benefit vs. no model** | $2,064,060 | **$2,578,060** |

**Key business insight:** the high-recall threshold (0.20) produces a *larger* net dollar benefit despite quadrupling false alarms, because the average claim amount (~$53K) dwarfs the assumed $500 manual review cost. When missed fraud costs roughly 100x more than a false-positive review, **optimizing for recall outperforms optimizing for F1 or precision** — a conclusion that only emerges once model performance is translated into business economics rather than left as abstract classification metrics.

*(Manual review cost of $500/claim is a stated assumption, not derived from the dataset, since no investigator-cost field is available. Results should be treated as directionally informative rather than exact.)*

---

## 6. Explainability (SHAP)

SHAP was used to directly test the project's original question: **do severity and fraud share root causes?**

- **Severity model (XGBoost, TreeExplainer):** top driver is `collision_type_Unknown` by a wide margin (mean |SHAP| = 16,464 — more than 15x the next feature), consistent with the GLM's findings. Other contributors: `policy_annual_premium`, `months_as_customer`, `age`, `bodily_injuries`.
- **Fraud model (Logistic Regression, LinearExplainer):** top driver is `incident_severity_Major Damage` (mean |SHAP| = 0.59 — more than 2x the next feature), followed by claim component amounts (`vehicle_claim`, `injury_claim`) and `multi_vehicle_incident`.
- **Overlap analysis:** only `months_as_customer` and `age` appear in both models' top 10 — and neither ranks in the top 3 for either model. The dominant drivers are **completely disjoint**: severity is explained by incident mechanics (what physically happened), while fraud is explained by claim categorization and component size (how the claim was recorded and how large it is).

**Conclusion:** This is strong, dual-method evidence (tree-based and linear explainability methods independently agree) that severity and fraud are governed by different underlying processes. A combined "risk score" approach would obscure this — separate models, as built here, are the better architecture.

**Caveat:** `incident_severity` dominating the fraud model may partly reflect *detection intensity* rather than a purely causal driver — adjusters likely scrutinize large claims more closely, which could inflate the observed association between severity and reported fraud. This is flagged rather than treated as a clean causal finding.

---

## 7. Limitations

- **Sample size:** 1,000 total claims (800 train / 200 test) is modest for fraud detection specifically, given only ~49 fraud cases in the test set — individual prediction changes move precision/recall noticeably, visible in the jaggedness of the precision-recall curve.
- **Synthetic dataset artifacts:** the `insured_hobbies` quirk (chess/cross-fit deterministically tied to fraud) is a known generation artifact in this teaching dataset and was excluded; it serves as a reminder to validate feature importance against domain plausibility before trusting it.
- **Severity ceiling:** ~31% of claim severity variance remains unexplained after extensive feature engineering, interaction testing, and hyperparameter tuning across two model families — most plausibly irreducible noise in this dataset rather than a missing feature.
- **Business impact figures rely on stated assumptions** (manual review cost, naive baseline definition, linear extrapolation to a larger book) and should be read as directionally informative, not as validated forecasts.
- **Possible detection-intensity confound** in the fraud model's reliance on `incident_severity`, as noted in Section 6.
- **Scope:** a second, real-world actuarial dataset (freMTPL2) was considered as a validation benchmark for the severity model but deliberately excluded — it has no fraud label and would require a separate feature engineering pipeline, which would dilute this project's focused two-target narrative on a single dataset rather than meaningfully strengthen it.

---

## 8. Summary

| Stage | Outcome |
|---|---|
| EDA | Identified bimodal severity distribution and dominant fraud predictor (`incident_severity`); ruled out multiple plausible-sounding hypotheses with evidence |
| Feature engineering | Built leakage-safe feature sets for each target; caught and removed a redundant feature and a dataset artifact |
| Severity model | GLM and XGBoost converge at R² ≈ 0.69; identified and explained the performance ceiling |
| Fraud model | Logistic Regression (scaled) outperforms Random Forest on recall; threshold tuning translated into real dollar tradeoffs |
| Explainability | SHAP confirms severity and fraud are driven by largely disjoint feature sets |
| Business impact | Quantified reserving improvement (45.5% error reduction) and fraud detection net benefit ($2.58M at the high-recall threshold) |

This project demonstrates a complete, defensible modeling workflow: every major decision — feature inclusion, model choice, threshold selection — is backed by a specific piece of evidence generated during the analysis, including several instances of testing and rejecting an initially plausible hypothesis.
