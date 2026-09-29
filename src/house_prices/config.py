"""Paths and validated training profiles; no environment files are loaded."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TRAIN_PATH = DATA_DIR / "train.csv"
PREDICT_PATH = DATA_DIR / "test.csv"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "run"
MODEL_PATH = DEFAULT_OUTPUT_DIR / "model.joblib"
PREDICTIONS_PATH = OUTPUT_DIR / "submission.csv"
MODEL_NAMES = (
    "dummy",
    "linear",
    "ridge",
    "lasso",
    "elasticnet",
    "random_forest",
    "lightgbm",
    "xgboost",
    "catboost",
    "voting",
    "stacking",
    "weighted",
)
FEATURE_SETS = ("original", "engineered", "extended", "peer")
SELECTIONS = ("none", "null95", "null80", "stable_null", "boruta")


@dataclass(frozen=True)
class TrainConfig:
    models: tuple[str, ...] = ("ridge", "random_forest")
    feature_sets: tuple[str, ...] = ("original", "engineered")
    target_transforms: tuple[str, ...] = ("log",)
    encoding: str = "target"
    selection: str = "none"
    missing_policy: str = "impute"
    seed: int = 42
    cv_folds: int = 3
    n_estimators: int = 200
    n_jobs: int = 2
    search_iterations: int = 0
    selection_iterations: int = 20
    holdout_size: float = 0.20
    validation_size: float = 0.16
    primary_metric: str = "rmse"
    smearing: bool = True

    def __post_init__(self):
        for name, allowed in [
            ("models", MODEL_NAMES),
            ("feature_sets", FEATURE_SETS),
            ("target_transforms", ("raw", "log")),
        ]:
            values = getattr(self, name)
            if (
                not isinstance(values, (tuple, list))
                or not values
                or len(values) != len(set(values))
                or set(values) - set(allowed)
            ):
                raise ValueError(f"{name} must be a nonempty unique list from {allowed}")
        for name, allowed in [
            ("encoding", ("target", "onehot")),
            ("selection", SELECTIONS),
            ("missing_policy", ("impute", "drop")),
            ("primary_metric", ("rmse", "mae")),
        ]:
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        for name in [
            "seed",
            "cv_folds",
            "n_estimators",
            "n_jobs",
            "search_iterations",
            "selection_iterations",
        ]:
            value = getattr(self, name)
            minimum = 0 if name in {"seed", "search_iterations"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.cv_folds < 2 or self.selection_iterations < 2:
            raise ValueError("cv_folds and selection_iterations must be >= 2")
        for name in ["holdout_size", "validation_size"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite fraction")
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must lie strictly between 0 and 1")
        if self.holdout_size + self.validation_size >= 1:
            raise ValueError("Holdout and validation fractions must sum to less than 1")
        if not isinstance(self.smearing, bool):
            raise ValueError("smearing must be a boolean")

    def to_dict(self):
        return asdict(self)


def make_config(profile="default", **overrides):
    profiles = {
        "default": {},
        "full": {
            "models": (
                "linear",
                "ridge",
                "lasso",
                "elasticnet",
                "random_forest",
                "lightgbm",
                "xgboost",
                "catboost",
                "voting",
                "stacking",
                "weighted",
            ),
            "feature_sets": ("engineered", "extended"),
            "n_estimators": 500,
            "search_iterations": 4,
        },
        "notebook": {
            "models": ("xgboost", "catboost", "voting", "stacking"),
            "feature_sets": ("engineered",),
            "selection": "boruta",
            "n_estimators": 1000,
            "cv_folds": 5,
            "selection_iterations": 100,
        },
    }
    if profile not in profiles:
        raise ValueError("profile must be default, full, or notebook")
    values = {**profiles[profile], **{k: v for k, v in overrides.items() if v is not None}}
    for name in ["models", "feature_sets", "target_transforms"]:
        if name in values:
            values[name] = tuple(values[name])
    return TrainConfig(**values)
