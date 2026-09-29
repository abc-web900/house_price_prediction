"""Regressors with complete preprocessing inside every CV and stacking fit."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .features import FrameEncoder, HouseFeatures, PeerComparables
from .selection import FeatureSelector


class CatBoostCompat(RegressorMixin, BaseEstimator):
    """Adapter for CatBoost releases without current sklearn estimator tags."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)


class PriceRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        family="ridge",
        feature_set="engineered",
        target_transform="log",
        encoding="target",
        selection="none",
        missing_policy="impute",
        seed=42,
        cv_folds=3,
        n_estimators=200,
        n_jobs=2,
        selection_iterations=20,
        model_params=None,
    ):
        self.family = family
        self.feature_set = feature_set
        self.target_transform = target_transform
        self.encoding = encoding
        self.selection = selection
        self.missing_policy = missing_policy
        self.seed = seed
        self.cv_folds = cv_folds
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.selection_iterations = selection_iterations
        self.model_params = model_params

    def _pipeline(self, family):
        override = dict(self.model_params or {})
        if family == "dummy":
            estimator = DummyRegressor(strategy="median")
        elif family == "linear":
            estimator = LinearRegression(**override)
        elif family == "ridge":
            estimator = Ridge(**{"alpha": 10.0, **override})
        elif family == "lasso":
            estimator = Lasso(**{"alpha": 0.001, "max_iter": 10000, **override})
        elif family == "elasticnet":
            estimator = ElasticNet(
                **{
                    "alpha": 0.001,
                    "l1_ratio": 0.5,
                    "max_iter": 10000,
                    **override,
                }
            )
        elif family == "random_forest":
            estimator = RandomForestRegressor(
                **{
                    "n_estimators": self.n_estimators,
                    "min_samples_leaf": 2,
                    "random_state": self.seed,
                    "n_jobs": self.n_jobs,
                    **override,
                }
            )
        elif family == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:
                raise ImportError("Install .[boosting] to use LightGBM") from exc
            estimator = LGBMRegressor(
                **{
                    "n_estimators": self.n_estimators,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "verbosity": -1,
                    "random_state": self.seed,
                    "n_jobs": self.n_jobs,
                    **override,
                }
            )
        elif family == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError("Install .[boosting] to use XGBoost") from exc
            estimator = XGBRegressor(
                **{
                    "n_estimators": self.n_estimators,
                    "learning_rate": 0.05,
                    "max_depth": 3,
                    "subsample": 0.8,
                    "tree_method": "hist",
                    "objective": "reg:squarederror",
                    "random_state": self.seed,
                    "n_jobs": self.n_jobs,
                    **override,
                }
            )
        elif family == "catboost":
            try:
                from catboost import CatBoostRegressor
            except ImportError as exc:
                raise ImportError("Install .[boosting] to use CatBoost") from exc
            estimator = CatBoostCompat(
                CatBoostRegressor(
                    **{
                        "iterations": self.n_estimators,
                        "learning_rate": 0.05,
                        "depth": 6,
                        "loss_function": "RMSE",
                        "random_seed": self.seed,
                        "verbose": False,
                        "thread_count": self.n_jobs,
                        "allow_writing_files": False,
                        **override,
                    }
                )
            )
        else:
            raise ValueError(f"Unsupported model family: {family}")
        steps = []
        if self.feature_set == "peer":
            steps.append(
                (
                    "peer",
                    PeerComparables(
                        self.cv_folds,
                        self.seed,
                        self.target_transform == "log",
                    ),
                )
            )
        steps.extend(
            [
                ("features", HouseFeatures(self.feature_set, self.missing_policy)),
                ("encoding", FrameEncoder(self.encoding, self.cv_folds, seed=self.seed)),
                (
                    "selection",
                    FeatureSelector(
                        "none" if family == "dummy" else self.selection,
                        self.selection_iterations,
                        self.seed,
                        self.n_jobs,
                    ),
                ),
            ]
        )
        if family in {"linear", "ridge", "lasso", "elasticnet"}:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", estimator))
        return Pipeline(steps)

    def fit(self, X, y):
        target = np.asarray(y, dtype=float)
        if self.target_transform == "log":
            target = np.log1p(target)
        if self.family in {"voting", "weighted", "stacking"}:
            # Each base pipeline is independently refitted inside stacking folds.
            bases = [(name, self._pipeline(name)) for name in ("xgboost", "catboost")]
            if self.family == "stacking":
                self.estimator_ = StackingRegressor(
                    estimators=bases,
                    final_estimator=Ridge(alpha=1.0),
                    n_jobs=1,
                    cv=KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed),
                )
            else:
                self.estimator_ = VotingRegressor(
                    estimators=bases,
                    n_jobs=1,
                    weights=[0.3, 0.7] if self.family == "weighted" else None,
                )
        else:
            self.estimator_ = self._pipeline(self.family)
        self.estimator_.fit(X, target)
        self.smearing_factor_ = 1.0
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_transformed(self, X):
        check_is_fitted(self, "estimator_")
        return np.asarray(self.estimator_.predict(X), dtype=float)

    def predict(self, X):
        values = self.predict_transformed(X)
        if self.target_transform == "log":
            # For log1p targets, E[price|X] = exp(prediction) * factor - 1.
            values = np.exp(values) * self.smearing_factor_ - 1.0
        if not np.isfinite(values).all():
            raise ValueError("Model produced nonfinite prices")
        return np.maximum(values, 0.0)

    def feature_records(self):
        check_is_fitted(self, "estimator_")
        components = (
            self.estimator_.named_estimators_
            if self.family in {"voting", "weighted", "stacking"}
            else {self.family: self.estimator_}
        )
        records = []
        for name, pipeline in components.items():
            selector = pipeline.named_steps["selection"]
            for column, selected in zip(selector.columns_, selector.support_):
                records.append(
                    {
                        "component": name,
                        "feature": column,
                        "selected": bool(selected),
                        "selection_fallback": selector.fallback_,
                    }
                )
        return records


def candidates(config):
    common = {
        k: getattr(config, k)
        for k in [
            "encoding",
            "selection",
            "missing_policy",
            "seed",
            "cv_folds",
            "n_estimators",
            "n_jobs",
            "selection_iterations",
        ]
    }
    models = {
        "median_baseline": PriceRegressor(
            family="dummy",
            feature_set="original",
            target_transform="raw",
            **common,
        )
    }
    for family in config.models:
        for features in config.feature_sets:
            for target in config.target_transforms:
                name = f"{family}__{features}__{target}"
                models[name] = PriceRegressor(
                    family=family,
                    feature_set=features,
                    target_transform=target,
                    **common,
                )
    return models


def search_space(model):
    spaces = {
        "ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0)],
        "lasso": [{"alpha": a} for a in (0.0001, 0.001, 0.01, 0.1)],
        "elasticnet": [{"alpha": a, "l1_ratio": r} for a in (0.001, 0.01) for r in (0.3, 0.7)],
        "random_forest": [{"min_samples_leaf": n} for n in (1, 2, 4, 8)],
        "lightgbm": [
            {"num_leaves": n, "learning_rate": r} for n in (15, 31, 63) for r in (0.03, 0.05)
        ],
        "xgboost": [{"max_depth": n, "learning_rate": r} for n in (3, 4, 5) for r in (0.03, 0.05)],
        "catboost": [{"depth": n, "learning_rate": r} for n in (4, 6, 8) for r in (0.03, 0.05)],
    }
    return {"model_params": spaces[model.family]} if model.family in spaces else {}
