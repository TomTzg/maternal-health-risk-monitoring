# Machine Learning for Maternal Risk Monitoring
Project due date: 6 December 2026
> Team members: 
- 1012453 Zhu Haoran
- 1012594 Chen Yizun
- 1012425 He Chenju
- 1012361 Liu YongJiang
- 1012405 Ren Zilong
- 1012037 Zhang Yiyang

## Project Overview
Maternal health risk classification using a single classical machine‑learning model.
We implement a Random Forest classifier on a maternal health risk dataset hosted by Mendeley Data.
Goal: multi‑class risk prediction: `low risk / mid risk / high risk`.

Dataset source: Hossain, Mohammad Mobarak (2025), “Maternal health risk data,” Mendeley Data, Version 1, DOI: [10.17632/ckhgtgdg4f.1](https://doi.org/10.17632/ckhgtgdg4f.1), licensed CC BY 4.0.

The repository data file contains 6,103 records and the six inputs retained from the original project scope: age, systolic and diastolic blood pressure, fasting blood glucose, body temperature, and heart rate. The source CSV's `Patient ID` and `Name` fields, along with its additional BMI and HbA1c features, are excluded from the repository copy. The glucose source header states `mg/dl`, while the accompanying article describes the 3.5–8.9 values as mmol/L; this unit discrepancy is recorded for verification.

The accompanying article reports removing 45 physiological outliers to obtain 6,058 records. The project audit retains the original 6,103 rows and treats range and cross-field plausibility concerns as audit findings; it does not silently remove records. In the retained six-feature view, identical input vectors can have different risk labels, so grouped splitting and explicit conflict reporting remain part of the experiment protocol.

## Models
- Logistic Regression
- RBF Support Vector Machine
- Gaussian Naive Bayes
- Decision Tree
- Random Forest
- Gradient Boosting Baseline: majority‑class predictor

## Model Choice: Random Forest
We use a single model, **Random Forest** (`sklearn.ensemble.RandomForestClassifier`). 

Reasons for this choice:

- **Well suited to small tabular data.** With 6,103 records and 6 features, tree ensembles are a strong default. Bagged trees capture non‑linear effects and interactions between blood pressure, glucose, temperature, age, and heart rate without manual feature engineering, and they usually outperform single linear or single‑tree models on this kind of benchmark.
- **Robust to heterogeneous units and outliers.** Inputs are measured in years, mmHg, mmol/L, °F, and bpm. Tree splits are invariant to monotone rescaling, so no normalization is needed, unlike distance‑based models such as RBF‑SVM. Out‑of‑range physiological readings have bounded influence on a split rather than dominating a distance or a linear coefficient.
- **Lower variance than a single decision tree.** Bagging plus random feature subsampling reduces overfitting to noisy or conflicting labels, which is relevant here because identical six‑feature input vectors can carry different risk labels. Cross‑validation estimates are therefore more stable.
- **Supports imbalance handling and risk triage.** `class_weight="balanced"` can lift mid‑ and high‑risk recall, and `predict_proba` provides per‑class probabilities that can be thresholded for triage instead of relying on hard labels.
- **Production friendly.** Training parallelizes (`n_jobs`), results are reproducible with a fixed `random_state`, batch inference is fast, and feature importance gives a documented, reviewable basis for clinical stakeholders.

**Trade‑offs acknowledged:** the model is less transparent than logistic regression, has a larger memory footprint than a single tree, and cannot extrapolate beyond the observed feature ranges. Feature importance (impurity‑ and permutation‑based) is used to document model behaviour, and out‑of‑range inputs are flagged at inference time.

**Tuned hyper‑parameters:** `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `class_weight`.

Baseline: majority‑class predictor, reported for reference only (not a competing model).


## Experiment protocol
- Grouped splitting by identical input vectors to prevent leakage between train and test
- 20% held‑out final test set
- 5‑fold stratified cross‑validation on the remaining 80% for hyper‑parameter tuning
- Tune Random Forest hyper‑parameters, selecting the configuration with the best validation macro‑F1
- Refit the single selected configuration on the full training set and evaluate once on the held‑out test set
- Metrics: accuracy, macro‑F1, per‑class precision/recall, confusion matrix, high‑risk recall, inference latency, age‑group error analysis
- Majority‑class baseline reported for reference

## Folder structure
```
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── reports/
│   ├── figures/
│   └── results/
├── demo/
└── docs/
│   ├── proposal/
│   ├── presentation/
│   └── final_report/
```


## Quick Start (Docker)

Install Docker Engine / Docker Desktop with Compose v2. No host Python setup is required.
Run all commands from this repository root. Place the original six-feature CSV at
`data/raw/Maternal Health Risk Data Set.csv` (raw data is intentionally ignored by Git).
The required columns are `Age,SystolicBP,DiastolicBP,BS,BodyTemp,HeartRate,RiskLevel`.

```bash
# Build the shared Python 3.11 image
docker compose build
# Audit, split, tune all six models + baseline, evaluate, then predict sample inputs
docker compose run --rm pipeline
# Faster smoke run: one parameter candidate per model, still five-fold CV
docker compose run --rm pipeline python -m src.pipeline --quick
# Automated data isolation, model fitting and batch inference checks
docker compose run --rm test
```

Compose uses the `docker.1ms.run` mirror for the Python base image. To use Docker
Hub directly, run `PYTHON_IMAGE=python:3.11-slim docker compose build`.

Individual stages (run in this order):

```bash
docker compose run --rm preprocess
docker compose run --rm train
docker compose run --rm evaluate
docker compose run --rm demo
```

Use `docker compose run --rm train python -m src.models.train --jobs 2` for
parallel parameter search. Custom demo CSV files can be placed in `data/raw`:

```bash
docker compose run --rm demo python -m demo.run_demo --input /app/data/raw/input.csv --output /app/reports/results/predictions.csv
```

### Data and experiment behavior

- A fixed seed (42) and five-way stratified **grouped** split allocate approximately
  20% to the final test set; exact size may vary to keep identical feature vectors together.
- Five-fold stratified grouped cross-validation uses only the remaining training data.
  Median imputation and scaling (LR/SVM) are fitted inside each training fold.
- All six models and the majority-class baseline are compared by validation macro-F1.
  The winning pipeline is refitted on all training rows and saved for evaluation/inference.
- Missing numeric values are imputed. Invalid columns, labels, nonnumeric values and
  infinities are rejected. Physiological flags and conflicting labels are reported without filtering.
- Holdout results do not participate in model selection. Run final evaluation after
  fixing the experiment configuration; repeated tuning against test results invalidates the holdout.
- Blood glucose units remain unresolved; values are preserved as provided. This is a
  research demonstration and is not validated for clinical decision-making.

### Persistent outputs

Docker mounts raw data read-only and writes other outputs to host bind mounts:

| Path | Content |
| --- | --- |
| `data/processed/train.csv`, `test.csv` | Group-disjoint split |
| `reports/results/data_audit.json` | Counts, missing values, conflicting labels, range flags, split summary |
| `reports/results/model_comparison.csv`, `cv_*.csv` | Validation scores and parameter searches |
| `artifacts/best_model.joblib`, `model_metadata.json` | Selected fitted pipeline and experiment settings |
| `reports/results/test_metrics.json` | Accuracy, macro-F1, per-class precision/recall, high-risk recall, warmed batch latency |
| `reports/results/confusion_matrix.csv`, `test_predictions.csv`, `age_group_errors.csv` | Test diagnostics |
| `reports/figures/confusion_matrix.png` | Confusion matrix plot |
| `reports/results/demo_predictions.csv` | Batch prediction example |

Outputs are overwritten on rerun. Models must be regenerated when dependencies change;
load only trusted joblib artifacts. Input/output directories can be overridden by
editing Compose bind mounts. Data and generated model artifacts are not baked into the image.
