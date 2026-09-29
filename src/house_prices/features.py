"""Training-fold feature statistics, cross-fitted encoding, and peer comparables."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .data import require_columns
from .schema import CAT_COLS, NUM_COLS, RAW_FEATURES


def clean_raw(X):
    require_columns(X)
    z = X[RAW_FEATURES].copy()
    for col in NUM_COLS:
        z[col] = pd.to_numeric(z[col], errors="coerce").astype(float)
    z[NUM_COLS] = z[NUM_COLS].replace([np.inf, -np.inf], np.nan)
    for col in CAT_COLS:
        z[col] = z[col].astype("string").str.strip().replace("", pd.NA)
        z[col] = z[col].fillna("Missing").astype(object)
    return z


class HouseFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, feature_set="engineered", missing_policy="impute"):
        self.feature_set = feature_set
        self.missing_policy = missing_policy

    def fit(self, X, y=None):
        z = clean_raw(X)
        self.medians_ = z[NUM_COLS].median().fillna(0.0)
        if self.missing_policy == "drop":
            # Match complete-column experiments without inspecting validation rows.
            self.active_columns_ = [c for c in RAW_FEATURES if not X[c].isna().any()]
        else:
            self.active_columns_ = RAW_FEATURES.copy()
        filled = z[NUM_COLS].fillna(self.medians_)
        total = filled["TotalBsmtSF"] + filled["1stFlrSF"] + filled["2ndFlrSF"]
        self.neighborhood_sf_ = total.groupby(z["Neighborhood"]).median().to_dict()
        self.global_sf_ = float(total.median())
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X):
        check_is_fitted(self, "medians_")
        z = clean_raw(X)
        z[NUM_COLS] = z[NUM_COLS].fillna(self.medians_)
        result = z[self.active_columns_].copy()
        if self.feature_set != "original":
            result["TotalSF"] = z["TotalBsmtSF"] + z["1stFlrSF"] + z["2ndFlrSF"]
            result["TotalBath"] = (
                z["FullBath"] + 0.5 * z["HalfBath"] + z["BsmtFullBath"] + 0.5 * z["BsmtHalfBath"]
            )
            result["AgeAtSale"] = z["YrSold"] - z["YearBuilt"]
        if self.feature_set in {"extended", "peer"}:
            median_sf = z["Neighborhood"].map(self.neighborhood_sf_).fillna(self.global_sf_)
            result["SF_Rel_to_Nbhd"] = result["TotalSF"] / median_sf.where(median_sf > 0)
            result["Qual_SF_Interact"] = z["OverallQual"] * result["TotalSF"]
        if "Peer_Comp_Price" in X:
            result["Peer_Comp_Price"] = X["Peer_Comp_Price"]
        return result


class FrameEncoder(TransformerMixin, BaseEstimator):
    """Smoothed target means; fit_transform excludes each inner fold's labels."""

    def __init__(self, encoding="target", folds=5, smoothing=10.0, seed=42):
        self.encoding = encoding
        self.folds = folds
        self.smoothing = smoothing
        self.seed = seed

    def _mapping(self, X, y, col):
        values = pd.DataFrame({"category": X[col].to_numpy(), "target": np.asarray(y)})
        grouped = values.groupby("category")["target"].agg(["count", "mean"])
        prior = float(np.mean(y))
        means = grouped["count"] * grouped["mean"] + self.smoothing * prior
        return (means / (grouped["count"] + self.smoothing)).to_dict(), prior

    def fit(self, X, y=None):
        self.numeric_ = X.select_dtypes(include=np.number).columns.tolist()
        self.categories_ = [c for c in X if c not in self.numeric_]
        self.medians_ = X[self.numeric_].median().fillna(0.0)
        if self.encoding == "target":
            if y is None:
                raise ValueError("Target encoding requires training labels")
            self.prior_ = float(np.mean(y))
            self.mappings_ = {c: self._mapping(X, y, c)[0] for c in self.categories_}
            self.columns_ = self.numeric_ + [c + "_encoded" for c in self.categories_]
        elif self.encoding == "onehot":
            self.columns_ = pd.get_dummies(
                X, columns=self.categories_, dtype=float
            ).columns.tolist()
        else:
            raise ValueError("encoding must be target or onehot")
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        z = X.copy()
        z[self.numeric_] = z[self.numeric_].replace([np.inf, -np.inf], np.nan)
        z[self.numeric_] = z[self.numeric_].fillna(self.medians_)
        if self.encoding == "onehot":
            return pd.get_dummies(z, columns=self.categories_, dtype=float).reindex(
                columns=self.columns_,
                fill_value=0.0,
            )
        result = z[self.numeric_].copy()
        for col in self.categories_:
            result[col + "_encoded"] = z[col].map(self.mappings_[col]).fillna(self.prior_)
        return result

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        result = self.transform(X)
        if self.encoding == "target" and self.categories_:
            if len(X) < 2:
                raise ValueError("Cross-fitted target encoding needs at least two training rows")
            folds = KFold(n_splits=min(self.folds, len(X)), shuffle=True, random_state=self.seed)
            target = np.asarray(y)
            for train, valid in folds.split(X):
                for col in self.categories_:
                    mapping, prior = self._mapping(X.iloc[train], target[train], col)
                    result.iloc[valid, result.columns.get_loc(col + "_encoded")] = (
                        X.iloc[valid][col].map(mapping).fillna(prior).to_numpy()
                    )
        return result


class PeerComparables(TransformerMixin, BaseEstimator):
    """KNN peer prices cross-fitted inside each enclosing model-training fold."""

    def __init__(self, folds=5, seed=42, log_target=True):
        self.folds = folds
        self.seed = seed
        self.log_target = log_target

    def _inputs(self, raw):
        z = clean_raw(raw)
        return pd.DataFrame(
            {
                "TotalSF": z["TotalBsmtSF"] + z["1stFlrSF"] + z["2ndFlrSF"],
                "OverallQual": z["OverallQual"],
                "AgeAtSale": z["YrSold"] - z["YearBuilt"],
                "Neighborhood": z["Neighborhood"],
            },
            index=raw.index,
        )

    def _fit_model(self, X, y):
        prep = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    ["TotalSF", "OverallQual", "AgeAtSale"],
                ),
                (
                    "category",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ["Neighborhood"],
                ),
            ]
        )
        return Pipeline(
            [
                ("preprocess", prep),
                ("knn", KNeighborsRegressor(n_neighbors=min(5, len(X)), weights="distance")),
            ]
        ).fit(self._inputs(X), y)

    def fit(self, X, y):
        target = np.expm1(y) if self.log_target else np.asarray(y)
        self.model_ = self._fit_model(X, target)
        return self

    def transform(self, X):
        check_is_fitted(self, "model_")
        z = X.copy()
        z["Peer_Comp_Price"] = self.model_.predict(self._inputs(X))
        return z

    def fit_transform(self, X, y, **fit_params):
        target = np.expm1(y) if self.log_target else np.asarray(y)
        values = np.empty(len(X))
        folds = KFold(n_splits=min(self.folds, len(X)), shuffle=True, random_state=self.seed)
        for train, valid in folds.split(X):
            model = self._fit_model(X.iloc[train], target[train])
            values[valid] = model.predict(self._inputs(X.iloc[valid]))
        self.fit(X, y)
        z = X.copy()
        z["Peer_Comp_Price"] = values
        return z
