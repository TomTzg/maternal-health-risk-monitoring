import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.data.preprocess import validate_frame, feature_groups, audit
from src.utils.utilities import ROOT, PROCESSED, RESULTS, SEED, save_json, LABELS


def prepare(input_path):
    frame = validate_frame(pd.read_csv(input_path))
    groups = feature_groups(frame)
    if set(frame.RiskLevel) != set(LABELS):
        raise ValueError("Training dataset must contain all three risk classes")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(splitter.split(frame, frame.RiskLevel, groups))
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    if set(train.RiskLevel) != set(LABELS) or set(test.RiskLevel) != set(LABELS):
        raise ValueError("Each split must contain all classes; provide more distinct feature groups")
    assert not set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED / "train.csv", index=False)
    test.to_csv(PROCESSED / "test.csv", index=False)
    report = audit(frame)
    report["split"] = {"train_rows": len(train), "test_rows": len(test),
                       "test_fraction": len(test) / len(frame), "group_overlap": 0, "seed": SEED}
    save_json(RESULTS / "data_audit.json", report)
    print(f"Prepared {len(train)} training / {len(test)} test rows; no shared feature groups")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/Maternal Health Risk Data Set.csv")
    prepare(parser.parse_args().input)
