"""Run CSV batch inference using the selected model."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.models.predict import predict
from src.utils.utilities import ROOT, RESULTS


def run(input_path=ROOT / "demo/batch_input.csv", output_path=RESULTS / "demo_predictions.csv"):
    result = predict(pd.read_csv(input_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "demo/batch_input.csv")
    parser.add_argument("--output", type=Path, default=RESULTS / "demo_predictions.csv")
    args = parser.parse_args()
    run(args.input, args.output)
