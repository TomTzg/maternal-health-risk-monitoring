"""Shared paths and reproducibility settings."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
RESULTS = ROOT / "reports/results"
FIGURES = ROOT / "reports/figures"
ARTIFACTS = ROOT / "artifacts"
FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
LABELS = ["low risk", "mid risk", "high risk"]
SEED = 42

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")
