import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.utils.utilities import FEATURES, LABELS


def validate_frame(frame, require_target=True):
    columns = FEATURES + (["RiskLevel"] if require_target else [])
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    result = frame[columns].copy()
    if result.empty:
        raise ValueError("Input dataset is empty")
    for name in FEATURES:
        result[name] = pd.to_numeric(result[name], errors="raise")
        if result[name].isin([float("inf"), -float("inf")]).any():
            raise ValueError(f"Infinite values in {name}")
        if require_target and result[name].isna().all():
            raise ValueError(f"Feature {name} has no observed values")
    if require_target:
        result["RiskLevel"] = result["RiskLevel"].astype("string").str.strip().str.lower()
        if result["RiskLevel"].isna().any() or not result["RiskLevel"].isin(LABELS).all():
            raise ValueError(f"RiskLevel must contain only {LABELS}")
    return result


def feature_groups(frame):
    # Same six-feature vector always belongs to the same split/fold, even if labels conflict.
    return pd.util.hash_pandas_object(frame[FEATURES], index=False).astype(str)


def make_preprocessor(scale=False):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def audit(frame):
    groups = feature_groups(frame)
    counts = frame.groupby(groups)["RiskLevel"].nunique()
    conflicts = set(counts[counts > 1].index)
    # Broad screening flags are audit information, not exclusion criteria or clinical rules.
    ranges = {"Age": (10, 60), "SystolicBP": (70, 250), "DiastolicBP": (40, 150),
              "BS": (2, 30), "BodyTemp": (95, 106), "HeartRate": (40, 180)}
    flags = {name: int(((frame[name] < lo) | (frame[name] > hi)).sum())
             for name, (lo, hi) in ranges.items()}
    return {"rows": len(frame), "class_counts": frame.RiskLevel.value_counts().to_dict(),
            "missing_values": frame[FEATURES].isna().sum().to_dict(),
            "unique_feature_groups": groups.nunique(),
            "duplicate_feature_rows": int(groups.duplicated().sum()),
            "conflicting_label_groups": len(conflicts),
            "rows_in_conflicting_groups": int(groups.isin(conflicts).sum()),
            "range_flags": flags,
            "systolic_not_above_diastolic": int((frame.SystolicBP <= frame.DiastolicBP).sum()),
            "rows_removed": 0,
            "glucose_unit": "Unresolved: source header mg/dl versus article mmol/L; values retained unchanged.",
            "range_flag_note": "Broad audit thresholds only; no rows filtered."}
