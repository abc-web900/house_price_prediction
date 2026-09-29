"""Predict raw rows with the saved fitted workflow; export Id,SalePrice."""

from pathlib import Path

import pandas as pd

from .artifacts import load_artifact, write_json
from .config import MODEL_PATH, PREDICT_PATH, PREDICTIONS_PATH
from .data import identifiers, parse_target, read_csv, require_columns
from .schema import ID, TARGET
from .validation import metric_summary


def predict_frame(bundle, frame):
    columns = bundle["metadata"]["required_features"]
    require_columns(frame, columns)
    ids = identifiers(frame)
    values = bundle["model"].predict(frame[columns])
    return pd.DataFrame({ID: ids.to_numpy(), TARGET: values}, index=frame.index)


def run(model_path=None, data_path=None, output_path=None):
    model_path = MODEL_PATH if model_path is None else model_path
    data_path = PREDICT_PATH if data_path is None else data_path
    output = Path(PREDICTIONS_PATH if output_path is None else output_path)
    if output.resolve() in {Path(data_path).resolve(), Path(model_path).resolve()}:
        raise ValueError("Prediction output must not overwrite input data or the model")
    if output.exists():
        raise FileExistsError("Prediction output already exists; choose a new path")
    frame = read_csv(data_path)
    y = parse_target(frame) if TARGET in frame and frame[TARGET].notna().any() else None
    predictions = predict_frame(load_artifact(model_path), frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    if y is not None:
        write_json(output.with_suffix(".metrics.json"), metric_summary(y, predictions[TARGET]))
    return predictions
