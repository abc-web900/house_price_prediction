# Conversion verification

Checked on 2026-09-29 using synthetic Ames-shaped data. The original training
and external-test CSVs were not supplied, so these checks establish software
behavior rather than a new real-data accuracy or RMSE claim.

| Check | Result |
| --- | --- |
| Automated tests, including optional boosting and selectors | 23 passed |
| Direct demo, EDA, training, prediction, and report scripts | Passed from a working directory outside the project |
| Fresh-process model loading and raw-CSV inference | Passed |
| Source lint | Passed |
| Wheel and source-distribution build | Passed |
| Core linear, ridge, lasso, elastic-net, and random-forest fitting | Passed |
| Optional LightGBM, XGBoost, CatBoost, voting, weighted voting, and stacking | Passed |
| Boruta, null-importance thresholds, and stability selection | Passed |
| Cross-fitted target encoding and KNN peer features exclude their own labels | Passed |
| Modifying only holdout labels leaves model selection, fitted predictions, and calibration unchanged | Passed |
| Missing values, unseen categories, column order, and textual ID preservation | Passed |
| Duplicate headers/IDs and invalid targets rejected | Passed |
| Existing training outputs preserved | Passed |

The tests also check price predictions remain finite and nonnegative, optional
selection survives model serialization, and reports are generated from saved
artifacts. Synthetic test scores are intentionally not presented as notebook
reproduction results.

Local test versions: NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1,
LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10, Boruta 0.4.3, and pytest 9.1.1.
The included CI targets Python 3.11/3.12; those remote jobs have not been run.

The source notebook is copied unchanged from `Untitled235.ipynb`.
SHA-256: `81ca07a286c36baeb454af48be3a791da8621a57b3738af9dc791acdc4cf3f45`.
