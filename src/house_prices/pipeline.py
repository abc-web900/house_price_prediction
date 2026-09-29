"""Cross-validation selection, validation calibration, holdout scoring, and export."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from threadpoolctl import threadpool_limits

from .artifacts import file_hash, load_artifact, save_artifact, write_json
from .config import DEFAULT_OUTPUT_DIR, TRAIN_PATH, TrainConfig
from .data import load_training_data
from .models import candidates, search_space
from .predict import predict_frame
from .schema import RAW_FEATURES, TARGET
from .validation import calibrate_smearing, metric_summary, split_data


def run(data_path=None, output_dir=None, config=None):
    data_path = TRAIN_PATH if data_path is None else data_path
    output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else output_dir
    config = config or TrainConfig()
    with threadpool_limits(limits=config.n_jobs):
        return _run(data_path, output_dir, config)


def _run(data_path, output_dir, config):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    data = load_training_data(data_path)
    tr, va, te = split_data(data, config)
    X, y = data.X.iloc[tr], data.y.iloc[tr]
    folds = list(
        KFold(
            n_splits=config.cv_folds,
            shuffle=True,
            random_state=config.seed + 2,
        ).split(X)
    )
    templates = candidates(config)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [
            pd.DataFrame(
                {
                    "csv_row_index": data.X.iloc[ix].index,
                    "Id": data.ids.iloc[ix].to_numpy(),
                    "partition": label,
                }
            )
            for ix, label in [(tr, "train"), (va, "validation"), (te, "holdout")]
        ],
        ignore_index=True,
    ).to_csv(out / "split_assignments.csv", index=False)
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    records, fold_records = [], []
    for name, template in list(templates.items()):
        print(f"Cross-validating {name}", flush=True)
        started = time.perf_counter()
        space = search_space(template)
        if config.search_iterations and space:
            search = RandomizedSearchCV(
                template,
                space,
                n_iter=min(config.search_iterations, len(space["model_params"])),
                scoring=scoring,
                refit=config.primary_metric,
                cv=folds,
                random_state=config.seed,
                n_jobs=1,
                error_score="raise",
            ).fit(X, y)
            templates[name] = clone(search.best_estimator_)
            best = search.best_index_
            scores = {
                f"test_{metric}": np.array(
                    [
                        search.cv_results_[f"split{i}_test_{metric}"][best]
                        for i in range(config.cv_folds)
                    ]
                )
                for metric in scoring
            }
            pd.DataFrame(search.cv_results_).to_csv(out / f"search_{name}.csv", index=False)
        else:
            scores = cross_validate(
                template,
                X,
                y,
                cv=folds,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
            )
        row = {"candidate": name, "seconds": time.perf_counter() - started}
        for metric in scoring:
            values = scores[f"test_{metric}"] * (1 if metric == "r2" else -1)
            row[metric] = float(np.mean(values))
            row[metric + "_std"] = float(np.std(values, ddof=1))
            fold_records.extend(
                {
                    "candidate": name,
                    "fold": i + 1,
                    "metric": metric,
                    "value": float(value),
                }
                for i, value in enumerate(values)
            )
        records.append(row)
        pd.DataFrame(records).to_csv(out / "cv_results.csv", index=False)
    comparison = pd.DataFrame(records).sort_values(config.primary_metric, kind="stable")
    comparison.to_csv(out / "cv_results.csv", index=False)
    pd.DataFrame(fold_records).to_csv(out / "cv_folds.csv", index=False)
    winner = comparison.iloc[0]["candidate"]
    print(f"Fitting CV-selected model: {winner}", flush=True)
    model = clone(templates[winner]).fit(X, y)
    calibration = calibrate_smearing(
        model,
        data.X.iloc[va],
        data.y.iloc[va],
        config.primary_metric,
        config.smearing,
    )
    calibration.to_csv(out / "smearing_validation.csv", index=False)
    # Keep the same fitted model after calibration; holdout is never used for choices.
    predictions = model.predict(data.X.iloc[te])
    baseline = np.full(len(te), float(y.median()))
    metrics = {
        "holdout": metric_summary(data.y.iloc[te], predictions),
        "validation": metric_summary(data.y.iloc[va], model.predict(data.X.iloc[va])),
        "holdout_median_baseline": metric_summary(data.y.iloc[te], baseline),
    }
    metadata = {
        "selected_model": winner,
        "target": TARGET,
        "required_features": RAW_FEATURES,
        "selected_params": model.get_params(deep=False),
        "smearing_factor": model.smearing_factor_,
        "config": config.to_dict(),
        "data": data.summary,
        "data_sha256": file_hash(data_path),
        "partitions": {"train": len(tr), "validation": len(va), "holdout": len(te)},
        "fit_partition": "train only; smearing on validation",
        "metrics": metrics,
        "split_policy": "random listing rows; seed-fixed; no repeated IDs",
    }
    path = save_artifact(model, out, metadata)
    np.testing.assert_allclose(
        predict_frame(load_artifact(path), data.X.iloc[te])[TARGET],
        predictions,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.DataFrame(model.feature_records()).to_csv(out / "selected_features.csv", index=False)
    holdout = pd.DataFrame(
        {
            "csv_row_index": data.X.iloc[te].index,
            "Id": data.ids.iloc[te].to_numpy(),
            "actual_price": data.y.iloc[te].to_numpy(),
            "predicted_price": predictions,
        }
    )
    holdout["residual"] = holdout["actual_price"] - holdout["predicted_price"]
    holdout.to_csv(out / "holdout_predictions.csv", index=False)
    holdout["price_decile"] = pd.qcut(holdout["actual_price"], 10, duplicates="drop")
    total_sse = float(np.sum(holdout["residual"] ** 2))
    diagnostics = []
    for band, group in holdout.groupby("price_decile", observed=True):
        diagnostics.append(
            {
                "price_band": str(band),
                **metric_summary(group.actual_price, group.predicted_price),
                "sse_percent": 100 * float(np.sum(group.residual**2)) / total_sse
                if total_sse
                else 0.0,
            }
        )
    pd.DataFrame(diagnostics).to_csv(out / "holdout_price_deciles.csv", index=False)
    write_json(out / "metrics.json", metrics)
    write_json(out / "config.json", config.to_dict())
    _plots(holdout, out)
    print(f"Saved {path}; holdout RMSE={metrics['holdout']['rmse']:.2f}", flush=True)
    return {"model_path": path, "metadata": load_artifact(path)["metadata"]}


def _plots(frame, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(frame.actual_price, frame.predicted_price, alpha=0.5)
    ends = [
        min(frame.actual_price.min(), frame.predicted_price.min()),
        max(frame.actual_price.max(), frame.predicted_price.max()),
    ]
    axes[0].plot(ends, ends, "r--")
    axes[0].set(xlabel="Actual SalePrice", ylabel="Predicted SalePrice")
    axes[1].scatter(frame.actual_price, frame.residual, alpha=0.5)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set(xlabel="Actual SalePrice", ylabel="Actual minus predicted")
    fig.tight_layout()
    fig.savefig(out / "holdout_diagnostics.png", dpi=150)
    plt.close(fig)
