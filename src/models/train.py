import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from src.data.preprocess import validate_frame, feature_groups
from src.models.model_config import model_configs
from src.utils.utilities import PROCESSED, RESULTS, ARTIFACTS, FEATURES, LABELS, SEED, save_json


def train(quick=False, jobs=1):
    frame = validate_frame(pd.read_csv(PROCESSED / "train.csv"))
    groups = feature_groups(frame)
    cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
        frame[FEATURES], frame.RiskLevel, groups))
    for fit, val in cv:
        if set(frame.iloc[fit].RiskLevel) != set(LABELS) or set(frame.iloc[val].RiskLevel) != set(LABELS):
            raise ValueError("Insufficient class groups for stratified five-fold validation")
        assert not set(groups.iloc[fit]) & set(groups.iloc[val])
    rows, best_model, best_score, best_name = [], None, -1, None
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, pipeline, grid in model_configs(quick):
        search = GridSearchCV(pipeline, grid, cv=cv, scoring="f1_macro", n_jobs=jobs, error_score="raise")
        search.fit(frame[FEATURES], frame.RiskLevel)
        rows.append({"model": name, "validation_macro_f1": search.best_score_,
                     "validation_std": search.cv_results_["std_test_score"][search.best_index_],
                     "best_params": str(search.best_params_)})
        pd.DataFrame(search.cv_results_).to_csv(RESULTS / f"cv_{name}.csv", index=False)
        print(f"{name}: validation macro-F1={search.best_score_:.4f}", flush=True)
        if search.best_score_ > best_score:
            best_model, best_score, best_name = search.best_estimator_, search.best_score_, name
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, ARTIFACTS / "best_model.joblib")
    pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False).to_csv(RESULTS / "model_comparison.csv", index=False)
    save_json(ARTIFACTS / "model_metadata.json", {"selected_model": best_name,
              "validation_macro_f1": best_score, "features": FEATURES, "labels": LABELS,
              "seed": SEED, "folds": 5, "quick": quick, "selection": "training CV macro-F1 only"})
    return best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="One candidate per model; still five folds")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    train(args.quick, args.jobs)
