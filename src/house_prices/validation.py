"""Separate model selection, smearing calibration, and final holdout evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def metric_summary(y, predictions):
    y, predictions = np.asarray(y, dtype=float), np.asarray(predictions, dtype=float)
    return {
        "rows": len(y),
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "r2": float(r2_score(y, predictions)) if len(y) > 1 else None,
        "rmsle": float(np.sqrt(np.mean((np.log1p(y) - np.log1p(predictions)) ** 2))),
        "median_absolute_error": float(np.median(np.abs(y - predictions))),
        "mean_residual_actual_minus_predicted": float(np.mean(y - predictions)),
    }


def split_data(data, config):
    positions = np.arange(len(data.y))
    development, holdout = train_test_split(
        positions,
        test_size=config.holdout_size,
        random_state=config.seed,
    )
    train, validation = train_test_split(
        development,
        test_size=config.validation_size / (1 - config.holdout_size),
        random_state=config.seed + 1,
    )
    if min(len(train), len(validation), len(holdout)) < 2 or len(train) < 2 * config.cv_folds:
        raise ValueError("Not enough rows for the requested split and CV folds")
    return train, validation, holdout


def calibrate_smearing(model, X, y, primary_metric, enabled=True):
    model.smearing_factor_ = 1.0
    rows = [{"setting": "uncorrected", "factor": 1.0, **metric_summary(y, model.predict(X))}]
    if enabled and model.target_transform == "log":
        residuals = np.log1p(np.asarray(y)) - model.predict_transformed(X)
        factor = float(np.mean(np.exp(residuals)))
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Invalid smearing factor")
        model.smearing_factor_ = factor
        corrected = metric_summary(y, model.predict(X))
        rows.append({"setting": "smearing", "factor": factor, **corrected})
        if corrected[primary_metric] >= rows[0][primary_metric]:
            model.smearing_factor_ = 1.0
    return pd.DataFrame(rows)
