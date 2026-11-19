import argparse
from src.data.make_dataset import prepare
from src.models.train import train
from src.evaluation.evaluate import evaluate
from demo.run_demo import run
from src.utils.utilities import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    prepare(ROOT / "data/raw/Maternal Health Risk Data Set.csv")
    train(args.quick, args.jobs)
    evaluate()
    run()


if __name__ == "__main__":
    main()
