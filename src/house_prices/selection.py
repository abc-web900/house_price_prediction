"""Optional notebook selection experiments, learned inside each model fit."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted


class FeatureSelector(TransformerMixin, BaseEstimator):
    def __init__(self, method="none", iterations=20, seed=42, n_jobs=2):
        self.method = method
        self.iterations = iterations
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.columns_ = list(X.columns)
        self.fallback_ = False
        self.details_ = {}
        if self.method == "none":
            self.support_ = np.ones(X.shape[1], dtype=bool)
            return self
        if self.method == "boruta":
            try:
                from boruta import BorutaPy
            except ImportError as exc:
                raise ImportError("Install .[selection] to use Boruta") from exc
            model = RandomForestRegressor(
                max_depth=5,
                n_jobs=self.n_jobs,
                random_state=self.seed,
            )
            selector = BorutaPy(
                model,
                n_estimators="auto",
                perc=90,
                random_state=self.seed,
                max_iter=self.iterations,
                verbose=0,
            ).fit(X.to_numpy(), np.asarray(y))
            self.support_ = selector.support_.copy()
            ranking = selector.ranking_
            self.details_ = {
                "ranking": ranking.tolist(),
                "tentative": selector.support_weak_.tolist(),
            }
            if not self.support_.any():
                # Small datasets/short runs may confirm no columns; record this fallback.
                self.support_[int(np.argmin(ranking))] = True
                self.fallback_ = True
        else:
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:
                raise ImportError("Install .[boosting] to use null importance") from exc
            if self.method not in {"null95", "null80", "stable_null"}:
                raise ValueError("Unsupported selection method")
            runs = 5 if self.method == "stable_null" else 1
            percentile = 95 if self.method == "null95" else 80
            counts = np.zeros(X.shape[1], dtype=int)
            all_actual = []
            for run in range(runs):

                def estimator():
                    return LGBMRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        importance_type="gain",
                        n_jobs=self.n_jobs,
                        verbosity=-1,
                        random_state=self.seed + run,
                    )

                actual = estimator().fit(X, y).feature_importances_
                rng = np.random.default_rng(self.seed + run)
                null = np.array(
                    [
                        estimator().fit(X, rng.permutation(y)).feature_importances_
                        for _ in range(self.iterations)
                    ]
                )
                counts += actual > np.percentile(null, percentile, axis=0)
                all_actual.append(actual)
            self.support_ = counts >= (4 if runs == 5 else 1)
            self.details_ = {"selection_counts": counts.tolist(), "runs": runs}
            if not self.support_.any():
                self.support_[int(np.argmax(np.mean(all_actual, axis=0)))] = True
                self.fallback_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        return X.loc[:, np.asarray(self.columns_)[self.support_]]
