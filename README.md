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
Maternal health risk classification using classical machine‑learning models.
We compare Logistic Regression, RBF‑SVM, Gaussian Naive Bayes, Decision Tree, Random Forest, and Gradient Boosting on a maternal health risk dataset hosted by Mendeley Data.
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
- Gradient Boosting
Baseline: majority‑class predictor

## Experiment protocol
- 20% held‑out final test set
- 5‑fold stratified cross‑validation for hyper‑parameter tuning
- Select best model by validation macro‑F1
- Metrics: accuracy, macro‑F1, per‑class precision/recall, confusion matrix, high‑risk recall, inference latency, age‑group error analysis

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
