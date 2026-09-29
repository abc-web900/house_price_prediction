"""Optional boosting, selection, and ensemble paths use serializable pipelines."""

import joblib
import numpy as np
import pytest

from house_prices.demo import make_data
from house_prices.models import PriceRegressor
from house_prices.schema import RAW_FEATURES


@pytest.mark.parametrize(
    "family", ["lightgbm", "xgboost", "catboost", "voting", "weighted", "stacking"]
)
def test_optional_model_reload(tmp_path, family):
    needed = [family] if family in {"lightgbm", "xgboost", "catboost"} else ["xgboost", "catboost"]
    for package in needed:
        pytest.importorskip(package)
    frame = make_data(90)
    X, y = frame[RAW_FEATURES], frame.SalePrice
    model = PriceRegressor(family=family, n_estimators=6, cv_folds=2, n_jobs=1)
    model.fit(X.iloc[:70], y.iloc[:70])
    sample = X.iloc[70:].copy()
    sample["Neighborhood"] = "Unseen"
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(model.predict(sample), joblib.load(path).predict(sample))
    assert np.isfinite(model.predict(sample)).all()


@pytest.mark.parametrize("method", ["boruta", "null95", "null80", "stable_null"])
def test_selection_is_fitted_and_exported(tmp_path, method):
    pytest.importorskip("boruta" if method == "boruta" else "lightgbm")
    frame = make_data(90)
    model = PriceRegressor(
        family="ridge", selection=method, selection_iterations=3, cv_folds=2, n_jobs=1
    )
    model.fit(frame[RAW_FEATURES], frame.SalePrice)
    records = model.feature_records()
    assert any(row["selected"] for row in records)
    path = tmp_path / "selected.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(
        model.predict(frame[RAW_FEATURES].head()),
        joblib.load(path).predict(frame[RAW_FEATURES].head()),
    )


def test_peer_model_reload(tmp_path):
    frame = make_data(90)
    model = PriceRegressor(family="ridge", feature_set="peer", cv_folds=2, n_jobs=1)
    model.fit(frame[RAW_FEATURES], frame.SalePrice)
    path = tmp_path / "peer.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(
        model.predict(frame[RAW_FEATURES].tail()),
        joblib.load(path).predict(frame[RAW_FEATURES].tail()),
    )
