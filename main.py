from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import run_training


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for model training.

    :returns: Configured argument parser with model and path options.
    """
    parser = argparse.ArgumentParser(description="Train marketing ROI model")
    parser.add_argument(
        "--data-path",
        default="customerGroups.csv",
        help="Path to the input dataset (CSV).",
    )
    parser.add_argument(
        "--model-path",
        default="models/trained_model.pkl",
        help="Destination path for the trained model artifact.",
    )
    parser.add_argument(
        "--prep-path",
        default="models/preprocessor.pkl",
        help="Destination path for the serialized preprocessor.",
    )
    parser.add_argument(
        "--model-type",
        default="ada",
        choices=["dummy", "rf", "gbm", "xgb", "cat", "ada", "tree_ensemble"],
        help="Choose which estimator to train.",
    )
    return parser


def main() -> None:
    """Parse command-line arguments and launch training."""
    parser = build_parser()
    args = parser.parse_args()

    model_path = Path(args.model_path)
    prep_path = Path(args.prep_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prep_path.parent.mkdir(parents=True, exist_ok=True)

    run_training(
        data_path=args.data_path,
        model_path=str(model_path),
        prep_path=str(prep_path),
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
