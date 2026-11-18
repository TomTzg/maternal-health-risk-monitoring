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

##Models
-Logistic Regression
-RBF Support Vector Machine
-Gaussian Naive Bayes
-Decision Tree
-Random Forest
-Gradient Boosting Baseline: majority‑class predictor

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


## Quick Start
```bash
# create virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run preprocessing
python src/data/make_dataset.py

# train & cross‑validate
python src/models/train.py

# evaluate on held‑out test set
python src/evaluation/evaluate.py

# run batch prediction demo
python demo/run_demo.py
