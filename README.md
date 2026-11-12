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
We compare Logistic Regression, RBF‑SVM, Gaussian Naive Bayes, Decision Tree, Random Forest, Gradient Boosting on the UCI Maternal Health Risk dataset.
Goal: multi‑class risk prediction: `low risk / mid risk / high risk`.

Dataset source: UCI Machine Learning Repository (Maternal Health Risk) [CC‑BY‑4.0]

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