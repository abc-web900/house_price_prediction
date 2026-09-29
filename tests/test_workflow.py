"""Exercise data validation, honest holdout separation, and persisted prediction."""

import json

import numpy as np
import pandas as pd
import pytest

from house_prices.artifacts import load_artifact
from house_prices.config import TrainConfig
from house_prices.data import load_training_data, read_csv
from house_prices.demo import make_data
from house_prices.pipeline import run
from house_prices.predict import predict_frame
from house_prices.report import build
from house_prices.schema import RAW_FEATURES, TARGET
from house_prices.validation import split_data


def small_config():
    return TrainConfig(
        models=("ridge",), feature_sets=("engineered",), cv_folds=2, n_estimators=5, n_jobs=1
    )


def test_export_predict_and_report(tmp_path):
    data = tmp_path / "train.csv"
    frame = make_data(120)
    frame.to_csv(data, index=False)
    result = run(data, tmp_path / "run", small_config())
    bundle = load_artifact(result["model_path"])
    raw = read_csv(data).iloc[:5].copy()
    raw.loc[raw.index[0], "Neighborhood"] = "Unseen"
    raw.loc[raw.index[1], "LotArea"] = np.nan
    output = predict_frame(bundle, raw)
    assert output.columns.tolist() == ["Id", TARGET]
    assert output.Id.tolist() == raw.Id.tolist()
    assert output.Id.iloc[0] == "000001"
    assert np.isfinite(output[TARGET]).all() and (output[TARGET] >= 0).all()
    np.testing.assert_allclose(
        output[TARGET],
        predict_frame(bundle, raw[raw.columns[::-1]])[TARGET],
    )
    raw[TARGET] = 1e10
    np.testing.assert_allclose(output[TARGET], predict_frame(bundle, raw)[TARGET])
    with pytest.raises(ValueError, match="Missing required"):
        predict_frame(bundle, raw.drop(columns="Neighborhood"))
    assignments = pd.read_csv(tmp_path / "run/split_assignments.csv")
    assert assignments.csv_row_index.nunique() == 120
    assert set(assignments.partition) == {"train", "validation", "holdout"}
    paths = build(tmp_path / "run")
    assert all(p.stat().st_size > 100 for p in paths)
    with pytest.raises(FileExistsError):
        run(data, tmp_path / "run", small_config())


def test_holdout_labels_do_not_affect_model_or_calibration(tmp_path):
    path = tmp_path / "train.csv"
    frame = make_data(100)
    frame.to_csv(path, index=False)
    config = small_config()
    _, _, holdout = split_data(load_training_data(path), config)
    a = run(path, tmp_path / "a", config)
    frame.loc[holdout, TARGET] *= 5
    frame.to_csv(path, index=False)
    b = run(path, tmp_path / "b", config)
    assert a["metadata"]["selected_model"] == b["metadata"]["selected_model"]
    assert a["metadata"]["smearing_factor"] == b["metadata"]["smearing_factor"]
    sample = frame[RAW_FEATURES].head()
    np.testing.assert_allclose(
        load_artifact(a["model_path"])["model"].predict(sample),
        load_artifact(b["model_path"])["model"].predict(sample),
    )
    assert a["metadata"]["metrics"]["holdout"] != b["metadata"]["metrics"]["holdout"]


@pytest.mark.parametrize("problem", ["duplicate_header", "duplicate_id", "invalid_target"])
def test_invalid_input_rejected(tmp_path, problem):
    path = tmp_path / "train.csv"
    frame = make_data(100)
    if problem == "duplicate_id":
        frame.loc[1, "Id"] = frame.loc[0, "Id"]
    elif problem == "invalid_target":
        frame.loc[0, TARGET] = -1
    frame.to_csv(path, index=False)
    if problem == "duplicate_header":
        text = path.read_text().replace("LotFrontage", "LotArea", 1)
        path.write_text(text)
    with pytest.raises(ValueError):
        load_training_data(path)


def test_config_rejects_invalid_values():
    for overrides in [
        dict(cv_folds=1),
        dict(seed=True),
        dict(models=("unknown",)),
        dict(holdout_size=0.8, validation_size=0.3),
        dict(n_jobs=0),
    ]:
        with pytest.raises(ValueError):
            TrainConfig(**overrides)


def test_search_and_onehot_raw_target(tmp_path):
    path = tmp_path / "train.csv"
    make_data(100).to_csv(path, index=False)
    config = TrainConfig(
        models=("ridge",),
        feature_sets=("original",),
        target_transforms=("raw",),
        encoding="onehot",
        cv_folds=2,
        search_iterations=2,
        n_jobs=1,
    )
    result = run(path, tmp_path / "run", config)
    assert (tmp_path / "run/search_ridge__original__raw.csv").is_file()
    assert result["metadata"]["smearing_factor"] == 1.0
    assert json.loads((tmp_path / "run/metrics.json").read_text())["holdout"]["rows"] == 20
