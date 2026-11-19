import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.data.preprocess import validate_frame
from src.utils.utilities import PROCESSED, ARTIFACTS, RESULTS, FIGURES, FEATURES, LABELS, save_json


def evaluate():
    frame = validate_frame(pd.read_csv(PROCESSED / "test.csv"))
    model = joblib.load(ARTIFACTS / "best_model.joblib")
    x, y = frame[FEATURES], frame.RiskLevel
    model.predict(x.iloc[:1])  # warm-up excluded from latency
    start = perf_counter()
    predicted = model.predict(x)
    elapsed = perf_counter() - start
    report = classification_report(y, predicted, labels=LABELS, output_dict=True, zero_division=0)
    metrics = {"test_rows": len(frame), "accuracy": accuracy_score(y, predicted),
               "macro_f1": f1_score(y, predicted, labels=LABELS, average="macro", zero_division=0),
               "high_risk_recall": report["high risk"]["recall"], "classification_report": report,
               "batch_latency_ms": elapsed * 1000, "amortized_latency_ms_per_row": elapsed * 1000 / len(frame),
               "latency_note": "One warmed batch on this machine; excludes model loading and input validation."}
    save_json(RESULTS / "test_metrics.json", metrics)
    matrix = confusion_matrix(y, predicted, labels=LABELS)
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(RESULTS / "confusion_matrix.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(cmap="Blues")
    plt.tight_layout()
    plt.savefig(FIGURES / "confusion_matrix.png", dpi=160)
    plt.close()
    output = frame.copy()
    output["PredictedRisk"] = predicted
    output.to_csv(RESULTS / "test_predictions.csv", index=False)
    age_groups = pd.cut(frame.Age, [-np.inf, 19, 29, 39, np.inf], labels=["<20", "20-29", "30-39", "40+"])
    errors = []
    for group in age_groups.cat.categories:
        mask = age_groups == group
        errors.append({"age_group": group, "count": int(mask.sum()),
                       "error_rate": float((y[mask] != predicted[mask]).mean()) if mask.any() else None,
                       "macro_f1": f1_score(y[mask], predicted[mask], labels=LABELS, average="macro", zero_division=0) if mask.any() else None})
    pd.DataFrame(errors).to_csv(RESULTS / "age_group_errors.csv", index=False)
    print(f"Held-out accuracy={metrics['accuracy']:.4f}; macro-F1={metrics['macro_f1']:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate()
