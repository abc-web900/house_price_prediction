"""Train from a raw CSV; works from any current directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from house_prices import config, pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train, evaluate, and export house-price models")
    parser.add_argument("--data", type=Path, default=config.TRAIN_PATH)
    parser.add_argument("--output", type=Path, default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile", choices=["default", "full", "notebook"], default="default")
    parser.add_argument("--models", nargs="+", choices=config.MODEL_NAMES)
    parser.add_argument("--feature-sets", nargs="+", choices=config.FEATURE_SETS)
    parser.add_argument("--target-transforms", nargs="+", choices=["raw", "log"])
    parser.add_argument("--encoding", choices=["target", "onehot"])
    parser.add_argument("--selection", choices=config.SELECTIONS)
    parser.add_argument("--missing-policy", choices=["impute", "drop"])
    parser.add_argument("--primary-metric", choices=["rmse", "mae"])
    for name in [
        "seed",
        "cv-folds",
        "n-estimators",
        "n-jobs",
        "search-iterations",
        "selection-iterations",
    ]:
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--no-smearing", action="store_true")
    args = parser.parse_args(argv)
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in {"data", "output", "profile", "no_smearing"}
    }
    if args.no_smearing:
        overrides["smearing"] = False
    try:
        result = pipeline.run(args.data, args.output, config.make_config(args.profile, **overrides))
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Model: {result['model_path']}")
    print(f"Evaluation: {args.output / 'metrics.json'}")


if __name__ == "__main__":
    main()
