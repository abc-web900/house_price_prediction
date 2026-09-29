## Run the project

Put the original unprocessed training CSV in `data/train.csv` and the external
prediction CSV in `data/test.csv`. Run these commands from this project folder:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/run_eda.py
python scripts/run_train.py
python scripts/run_predict.py
python scripts/build_report.py
```

The scripts import their local `src/` directory, so an editable installation is
optional. Use a new output directory for another training or EDA run; existing
results are preserved. CLI paths are relative to your current directory, while
default paths resolve relative to the project. No environment files are loaded.

Prediction creates **`outputs/submission.csv` with `Id,SalePrice`**. Missing
values and category strings should remain raw: the saved model handles them.
`SalePrice` is optional in the prediction CSV. When true labels are supplied,
prediction also writes a metrics JSON; unlabeled input yields predictions only.

## Layout

```text
house_prices/
├── scripts/
│   ├── run_train.py
│   ├── run_predict.py
│   ├── run_eda.py
│   ├── build_report.py
│   └── make_demo_data.py
├── src/house_prices/
│   ├── config.py          # profiles and paths
│   ├── schema.py          # 79 raw predictors; Id and SalePrice excluded
│   ├── data.py            # validation, target parsing, ID preservation
│   ├── features.py        # house features, target encoding, KNN comparables
│   ├── selection.py       # null importance, stability selection, Boruta
│   ├── models.py          # regressors and complete voting/stacking pipelines
│   ├── validation.py      # splits, smearing calibration, metrics
│   ├── pipeline.py        # training, evaluation, model export
│   ├── predict.py         # saved-model prediction
│   ├── artifacts.py       # model persistence, hashes, dependency versions
│   ├── eda.py
│   ├── report.py
│   ├── demo.py
│   └── __init__.py
├── data/README.md         # actual notebook findings
├── examples/predict.csv  # synthetic input example
├── outputs/
├── tests/
├── .github/workflows/ci.yml
├── Untitled235.ipynb      # unchanged source notebook
├── requirements.txt
├── pyproject.toml
└── VALIDATION.md
```

## Models and experiment options

The default profile compares a median baseline, ridge, and random forest with
original/engineered features and a log target. It requires only the core
dependencies. The full profile adds the notebook's linear models, boosting
families, ensembles, extended features, and a small hyperparameter search.

```bash
python -m pip install -e ".[boosting,selection]"
python scripts/run_train.py --profile full --output outputs/full
```

The `notebook` profile runs XGBoost, CatBoost, equal voting, and Ridge stacking
with Boruta, engineered features, log targets, 1,000 boosting rounds, and five
folds. It can be expensive because feature selection is refitted inside the
evaluation and stacking folds.

```bash
python scripts/run_train.py --profile notebook --output outputs/notebook_recipe
```

| Option | Choices / purpose |
| --- | --- |
| `--models` | `dummy`, `linear`, `ridge`, `lasso`, `elasticnet`, `random_forest`, `lightgbm`, `xgboost`, `catboost`, `voting`, `stacking`, `weighted` |
| `--feature-sets` | `original`, `engineered`, `extended`, `peer` |
| `--target-transforms` | `raw`, `log`; accepts multiple values for comparison |
| `--encoding` | `target` (smoothed cross-fitted means) or `onehot` |
| `--selection` | `none`, `null95`, `null80`, `stable_null`, `boruta` |
| `--missing-policy` | `impute` (default) or training-fold complete-column `drop` |
| `--primary-metric` | Original-price `rmse` (default) or `mae` |
| `--search-iterations` | Per-candidate randomized parameter trials; zero disables search |
| `--selection-iterations` | Null permutations per run or Boruta maximum iterations |
| `--no-smearing` | Keep the inverse log transform without calibration |

`--seed`, `--cv-folds`, `--n-estimators`, and `--n-jobs` control reproducibility
and computation. `python scripts/run_train.py --help` lists all options.

For example, compare the notebook's extra neighborhood/quality features and
KNN-comparable feature using one family:

```bash
python scripts/run_train.py --models catboost --feature-sets engineered extended peer --output outputs/features
```

Equal voting and stacking combine XGBoost and CatBoost. Weighted voting uses
30% XGBoost and 70% CatBoost, reproducing the notebook's fixed blend recipe
without claiming it is optimal. With a log target, those ensembles combine
log predictions before converting back to price units.

## Preprocessing and features

All 79 original predictor columns are required; their order does not matter.
The complete list is in [schema.py](src/house_prices/schema.py). Extra columns
are ignored. `Id` is optional, preserved as text when present, and never used
as a predictor. Training requires at least 80 rows, unique nonempty IDs when
present, and finite positive prices. Exact duplicate records are removed first.

The notebook dropped every column with any missing values. The default module
keeps those columns, fills numeric values using training-fold medians (zero for
entirely missing columns), and labels missing categories `Missing`. The optional
`drop` policy chooses complete columns using each training fold only.

`engineered` adds `TotalSF`, `TotalBath`, and `AgeAtSale`. `extended` also adds
size relative to the neighborhood's training median and quality × size.
Unknown neighborhoods use the overall training median. `peer` adds KNN peer
prices through inner cross-fitting, using size, quality, age, and one-hot
neighborhood for similarity. Its scaler and nearest-neighbor reference rows
are also fitted within those inner folds.

Target encoding learns smoothed categorical target means. Training rows receive
inner-fold encodings computed without that fold's labels, including the smoothing
prior. Inference uses mappings fitted on training labels only; unseen categories
use the training prior. Raw-price models encode raw prices, while log-target
models encode log prices.

Null-importance selection uses LightGBM and 95th/80th percentile shuffled-target
thresholds. Stability selection keeps features passing an 80th-percentile test
in at least four of five runs. Boruta uses a depth-five random forest and the
90th-percentile shadow threshold. If a run confirms no feature, the selector
keeps its strongest-ranked feature and records that fallback in
`selected_features.csv`; it does not claim that feature passed the test.

## Evaluation and smearing

1. Reserve 20% of rows for holdout and 16% for calibration, leaving about 64%
   for training CV. Splits are fixed by the configured seed.
2. Select model/feature/target recipes using the same training folds. Learned
   preprocessing, supervised encoding, selection, and peer features stay within
   each outer training fold. Stacking includes complete pipelines in each base
   estimator, so inner stack folds also refit those transformations.
3. Fit the selected recipe on training rows. Estimate a smearing factor from
   the separate calibration set and retain it only if it improves the configured
   calibration metric. The holdout does not influence this decision.
4. Freeze that model and factor, evaluate the holdout, and export the exact
   evaluated workflow. It is not refitted on the calibration/holdout rows.

For a `log1p` target, corrected prices use **`exp(predicted_log) * factor - 1`**.
This corrects the notebook's slightly different `expm1(predicted_log) * factor`
expression. All predicted prices are clipped at zero. RMSE, MAE, and residuals
are reported in original price units; RMSLE is also saved.

The notebook performed target encoding and feature selection on the whole
dataset before later CV, reused folds during tuning, and estimated/reported
smearing on the same pooled predictions. Its printed results are exploratory.
This module deliberately changes those evaluation boundaries, so reproducing
23,816.76 is not an expected test assertion. The original notebook remains
available for historical plots and the later passthrough meta-stack experiment.

## Saved artifacts and reports

Training writes these files to `outputs/run/` by default:

- `model.joblib`: preprocessing, feature selection, estimator(s), and smearing factor.
- `metadata.json`, `config.json`, `requirements.lock.txt`: schema, parameters,
  split sizes, source/data/model hashes, and dependency versions.
- `cv_results.csv`, `cv_folds.csv`, optional `search_*.csv`: development comparisons.
- `split_assignments.csv`: original indices, IDs, and assigned partitions.
- `smearing_validation.csv`: corrected and uncorrected calibration scores.
- `metrics.json`, `holdout_predictions.csv`: final evaluation and individual errors.
- `selected_features.csv`: actual fitted features, selections, and fallback flags.
- `holdout_price_deciles.csv`, `holdout_diagnostics.png`: error patterns by price.

`build_report.py` creates Markdown and HTML reports from these saved results.
It does not retrain the model. EDA separately writes missing-value tables,
numeric summaries/correlations, category counts, target associations, and plots.

Only load trusted model files with compatible project code and dependencies.
To predict without repeating any training:

```bash
python scripts/run_predict.py --model outputs/run/model.joblib --data data/test.csv --output outputs/submission.csv
```

## Verify with synthetic data

```bash
python scripts/make_demo_data.py --output data/demo.csv --rows 240
python scripts/run_train.py --data data/demo.csv --output outputs/demo --models ridge --feature-sets engineered
python scripts/run_predict.py --model outputs/demo/model.joblib --data examples/predict.csv --output outputs/demo_submission.csv
python scripts/build_report.py --run outputs/demo
```

Synthetic rows verify software behavior only. Real training and test CSVs are
not included in this project. Generated data, model files, archives, and
environment files are excluded from version control.

Tests and packaging checks:

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check src scripts tests
python -m build
```

See [VALIDATION.md](VALIDATION.md) for checks actually completed for this conversion.
