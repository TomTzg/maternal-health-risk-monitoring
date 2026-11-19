# Summary: Maternal Health Risk Monitoring

**Project**: Machine Learning for Maternal Risk Monitoring (610501 Production-ready ML)

**Deadline**: 2026-12-06 ｜ **Current commit**: `d6af9e7` (`master`, clean working tree)

**Status**: The pipeline runs end to end; model comparison, held-out evaluation, and three figures are complete.

---

## 1. Project Overview

### 1.1 Project Scope

Predict the maternal risk level (`low / mid / high`) from six routine physiological indicators (age, systolic BP, diastolic BP, blood glucose, body temperature, and heart rate), to support triage prioritization in resource-limited settings. **The model is for research demonstration only and is not a substitute for clinical diagnosis.**

### 1.2 Data and Method

- Data from Mendeley Maternal Health Risk Data (**6,103 records**, CC BY 4.0), with three nearly balanced classes.
- Group by a hash of the six-feature vector so that identical inputs never cross the train/test boundary or CV folds.
- A 20% held-out test set (**1,224 records**); the remaining **4,879 records** are used for 5-fold stratified group cross-validation.
- Compare Logistic Regression, RBF SVM, Gaussian NB, Decision Tree, Random Forest, Gradient Boosting, and a majority-class baseline; select by validation macro-F1.

### 1.3 Key Results

- Selected **Random Forest**: CV macro-F1 **0.9199**, versus Gradient Boosting at 0.9180. The gap is smaller than the fold-wise standard deviation, so the two are effectively tied; RF wins by the predefined rule.
- Held-out: **Accuracy 93.55%, Macro-F1 93.60%, high-risk recall 91.71%**.
- Errors concentrate on the **mid ↔ high** boundary; only 3 high-risk cases were predicted as low.
- The 30–39 age group has the highest error rate (13.50%); the <20 and 40+ groups have too few samples, so subgroup findings are exploratory only.

### 1.4 Engineering and Current Status

- Docker Compose runs preprocessing → training → evaluation → batch prediction with one command; random seed 42; pinned dependencies; three figures and complete metric files are available.
- Completed: data audit, grouped split, model comparison, held-out evaluation, error analysis, and figures.
- To regenerate:
  - `data/processed/train.csv`, `test.csv`: inputs for training and evaluation;
  - `artifacts/best_model.joblib`, `artifacts/model_metadata.json`: model and experiment metadata loaded by evaluation and prediction;
  - `reports/results/demo_predictions.csv`: batch prediction output from the demo.

---

## 2. Data

### 2.1 Dataset and Split

| Item | Content |
| --- | --- |
| Data file | `data/raw/Maternal Health Risk Data Set.csv` (199 KB) |
| Source | Mendeley Data, Version 1, DOI: 10.17632/ckhgtgdg4f.1, CC BY 4.0 |
| Records | **6,103** |
| Features (6) | `Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate` |
| Labels (3) | `low risk, mid risk, high risk` |
| Split | Train **4,879** / test **1,224** (20.06%), seed=42 |
| Leakage prevention | Hash-grouped by the six-dimensional feature vector; identical inputs fall only on the train or test side |

**Class distribution**:

| Class | Count | Share |
| --- | ---: | ---: |
| low risk | 2,001 | 32.8% |
| mid risk | 2,043 | 33.5% |
| high risk | 2,059 | 33.7% |

> The classes are close to 1:1:1, so SMOTE or other resampling is unnecessary.

### 2.2 Data Audit

| Audit item | Value |
| --- | ---: |
| Missing values | 0 |
| Unique feature-vector groups | 5,555 |
| Duplicate feature rows | 548 |
| **Groups with conflicting labels** | **311** |
| Rows in conflicting groups | 622 (about 10.2%) |
| Age outside 10–60 | 1 |
| Diastolic BP outside 40–150 | 1 |
| Body temperature outside 95–106 | 43 |
| Systolic ≤ Diastolic | 19 |
| **Rows removed** | **0** |

**Three data issues that must be explained:**

1. **Conflicting labels**: identical six-dimensional inputs map to different risk levels. This is label noise and sets a performance ceiling; the project retains all data and does not delete silently.
2. **Unresolved glucose unit**: the source header says `mg/dl`, but values of 3.5–8.9 are more consistent with the article's `mmol/L`. Original values are retained without conversion and listed as a limitation.
3. **Outliers are flagged, not removed**: 43 body-temperature, 1 age, and 1 diastolic BP values; additionally, 19 rows have systolic ≤ diastolic.

### 2.3 Experimental Results

**Cross-validation model comparison** (`reports/results/model_comparison.csv`):

| Rank | Model | Validation macro-F1 | ± std |
| ---: | --- | ---: | ---: |
| 1 | **Random Forest** | **0.9199** | 0.0079 |
| 2 | Gradient Boosting | 0.9180 | 0.0047 |
| 3 | Decision Tree | 0.9012 | 0.0078 |
| 4 | RBF SVM | 0.8846 | 0.0076 |
| 5 | Gaussian NB | 0.7637 | 0.0269 |
| 6 | Logistic Regression | 0.5536 | 0.0167 |
| 7 | Majority Baseline | 0.1693 | 0.0052 |

> RF leads GB by only **0.0019**, less than the fold-wise standard deviation; describe it as "wins by the rule, effectively tied".

**Held-out metrics** (`reports/results/test_metrics.json`):

| Metric | Value |
| --- | ---: |
| Accuracy | **93.55%** |
| Macro-F1 | **93.60%** |
| High-risk recall | **91.71%** |
| Misclassified samples | 79 / 1,224 (6.45%) |
| Batch latency | 11.43 ms / 1,224 rows (0.0093 ms/row) |

**Per-class metrics**:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| low risk | 0.9631 | 0.9924 | 0.9775 | 394 |
| mid risk | 0.9218 | 0.9005 | 0.9110 | 432 |
| high risk | 0.9217 | 0.9171 | 0.9194 | 398 |

**Confusion matrix**:

| True \ Predicted | low risk | mid risk | high risk |
| --- | ---: | ---: | ---: |
| low risk | **391** | 3 | 0 |
| mid risk | 12 | **389** | 31 |
| high risk | 3 | 30 | **365** |

> Most errors concentrate on **mid ↔ high**; only 3 high-risk cases were misclassified as low.

**Age-group errors**:

| Age group | Samples | Errors | Error rate | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| <20 | 46 | 2 | 4.35% | 0.3259 |
| 20–29 | 837 | 35 | 4.18% | 0.9541 |
| 30–39 | 311 | 42 | **13.50%** | 0.8918 |
| 40+ | 30 | 0 | 0.00% | 0.3333 |

> The <20 and 40+ groups have very few samples, so macro-F1 is unstable; treat these as exploratory findings only and do not compare the three macro-F1 values directly.

### 2.4 Result Files and Figures

| Path | Content | Reporting use |
| --- | --- | --- |
| `reports/figures/model_comparison_macro_f1.png` | CV comparison of 7 models (±1 std) | Model-selection slide |
| `reports/figures/confusion_matrix.png` | Held-out confusion matrix | Results slide |
| `reports/figures/age_group_error_rate.png` | Age-group error rate + 95% Wilson CI | Error-analysis slide |
| `reports/results/model_comparison.csv` | Raw model-comparison table | Report table |
| `reports/results/cv_*.csv` | Grid-search details for each model | Appendix/reference |
| `reports/results/test_metrics.json` | All held-out metrics | Source of report numbers |
| `reports/results/confusion_matrix.csv` | Confusion matrix | Report table |
| `reports/results/test_predictions.csv` | Row-level test predictions | Error analysis |
| `reports/results/age_group_errors.csv` | Age-group errors | Report table |
| `reports/results/data_audit.json` | Data audit | Data slide |

---

## 3. Reproduction Workflow

### 3.1 Prerequisites

- Docker Desktop / Docker Engine + Compose v2; no local Python required.
- Raw data is already at `data/raw/Maternal Health Risk Data Set.csv`.
- Default image mirror is `docker.1ms.run`; to use the official source:
  `PYTHON_IMAGE=python:3.11-slim docker compose build`.

### 3.2 One-Command Run

```bash
# Build the image
docker compose build

# Full pipeline: preprocessing → full-grid training → evaluation → demo
docker compose run --rm pipeline

# Quick smoke run (one candidate per model, still 5 folds; not for report numbers)
docker compose run --rm pipeline python -m src.pipeline --quick

# Unit tests
docker compose run --rm test
```

### 3.3 Stage-by-Stage Run (in order)

```bash
docker compose run --rm preprocess
docker compose run --rm train
docker compose run --rm evaluate
docker compose run --rm demo
```

Optional parallel search:

```bash
docker compose run --rm train python -m src.models.train --jobs 2
```

### 3.4 Custom Batch Prediction

```bash
docker compose run --rm demo python -m demo.run_demo \
  --input /app/data/raw/input.csv \
  --output /app/reports/results/predictions.csv
```

The default input is `demo/batch_input.csv` (three example rows).

### 3.5 Files Generated After Reproduction

| Path | Content |
| --- | --- |
| `data/processed/train.csv`, `test.csv` | Grouped split results (4,879 / 1,224) |
| `artifacts/best_model.joblib` | Selected model (full Pipeline) |
| `artifacts/model_metadata.json` | Selected model, validation score, seed, etc. |
| `reports/results/data_audit.json` | Data audit |
| `reports/results/model_comparison.csv` | Model comparison |
| `reports/results/cv_*.csv` | Grid-search details for each model |
| `reports/results/test_metrics.json` | Held-out metrics |
| `reports/results/confusion_matrix.csv` | Confusion matrix |
| `reports/results/test_predictions.csv` | Row-level test predictions |
| `reports/results/age_group_errors.csv` | Age-group errors |
| `reports/results/demo_predictions.csv` | Demo batch predictions |
| `reports/figures/*.png` | Three figures |

### 3.6 Notes

1. **Run `pipeline` or `train` first**: `artifacts/` does not currently exist, so running `evaluate` / `demo` directly will fail (purpose: see 1.5).
2. **Run `preprocess` first**: `data/processed/` is also not committed (purpose: see 1.5).
3. **Evaluate the test set only once**: model selection uses training CV only; do not tune against test results.
4. **`--quick` is not the final result**: use it only to validate the environment.
5. **Outputs are overwritten**: rerunning overwrites files with the same names; copy and back up a result if you need to preserve it.
6. **Git tracking reminder**: the raw CSV is already tracked; new files under `reports/` are ignored by default and require `git add -f`; `artifacts/` is never committed.

---

*Date: 2026-11-19 ｜ Based on commit `d6af9e7` ｜ If results change, update Sections 1.3 and 2.3.*
