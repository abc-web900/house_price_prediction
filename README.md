# House sale prices: notebook results and findings


## Dataset and preparation

The original training table contained **1,460 rows and 81 columns**, including
`Id` and the target `SalePrice`. Dropping every column with at least one missing
value left **62 columns**: the ID, target, and **60 predictors**. No row removal
was shown at that step.

Some of the missing-value counts recorded before dropping columns were:

| Column | Missing rows |
| --- | ---: |
| `PoolQC` | 1,453 |
| `MiscFeature` | 1,406 |
| `Alley` | 1,369 |
| `Fence` | 1,179 |
| `MasVnrType` | 872 |
| `FireplaceQu` | 690 |
| `LotFrontage` | 259 |
| `GarageYrBlt` | 81 |
| `MasVnrArea` | 8 |
| `Electrical` | 1 |

This complete-column rule removed 19 columns, including columns with only one
or a few missing entries. The module defaults to training-fold imputation so
those predictors can be retained; their benefit needs evaluation in a new run.

## Encoding and feature engineering

The notebook encoded **27 categorical features** using smoothed target means
and five inner folds. Smoothing used a strength of 10. Neighborhood had the
highest reported cardinality at **25 values**, followed by `Exterior2nd` at 16
and `Exterior1st` at 15. IDs and the target were excluded from model inputs.

The main engineered features were:

| Feature | Calculation |
| --- | --- |
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` |
| `TotalBath` | `FullBath + 0.5 × HalfBath + BsmtFullBath + 0.5 × BsmtHalfBath` |
| `AgeAtSale` | `YrSold - YearBuilt` |
| `SF_Rel_to_Nbhd` | `TotalSF` / neighborhood median `TotalSF` |
| `Qual_SF_Interact` | `OverallQual × TotalSF` |
| `Peer_Comp_Price` | Distance-weighted price from five similar houses using size, quality, age, and encoded neighborhood |

The final three feature groups were later experiments. They were not part of
the configuration that produced the notebook's lowest printed RMSE.

## Null importance versus Boruta

Null importance compared actual LightGBM feature gain against gain under shuffled
targets. The first test used **50 shuffled-target iterations** and the **95th
percentile**, selecting only **4 of 60 features**:

- `OverallQual`
- `GrLivArea`
- `Neighborhood_encoded`
- `GarageCars`

Reducing the percentile to 80 still selected four features. A stability check
using five runs, 20 permutations per run, and a four-of-five pass requirement
returned the same four features.

After adding `TotalSF`, `TotalBath`, and `AgeAtSale`, the stable set changed to
`OverallQual`, `GarageCars`, `Neighborhood_encoded`, `TotalSF`, and `TotalBath`.
CatBoost's printed original-price RMSE improved from **31,628.22** on the earlier
four-feature/log-target set to **26,957.85** on the engineered stable set.

Boruta then used a depth-five random forest and a 90th-percentile shadow-feature
threshold. It confirmed **31 features**, with `RoofStyle_encoded` remaining
tentative. The confirmed set included quality, area/capacity, construction years,
garage and basement attributes, the three engineered features, and encoded
location/property/quality categories. CatBoost on that set achieved **24,420.70
RMSE**. The less restrictive feature set worked better in the recorded comparison
than the four- or five-feature null-importance sets.

## Recorded model results

The initial table reports mean five-fold RMSE from `cross_val_score` or the
best search score. Later milestones report RMSE over pooled out-of-fold
predictions, usually after reversing a log target. Those aggregation methods
and feature sets differ, so this is a record of experiments rather than one
controlled leaderboard.

| Initial model on the four selected features | Mean CV RMSE |
| --- | ---: |
| Linear regression | 37,477.75 |
| Ridge | 37,477.64 |
| Lasso | 37,477.74 |
| ElasticNet | 37,494.06 |
| Tuned LightGBM | 33,437.21 |
| Tuned XGBoost | 31,638.23 |
| Tuned CatBoost | 31,429.06 |
| XGBoost/CatBoost equal voting | 31,283.54 |
| XGBoost/CatBoost Ridge stacking | 31,450.06 |

| Later experiment | Printed original-price RMSE |
| --- | ---: |
| Log-target CatBoost, original stable four features | 31,628.22 |
| Log-target CatBoost, engineered stable five features | 26,957.85 |
| Voting, engineered stable features | 27,085.51 |
| Stacking, engineered stable features | 27,052.25 |
| CatBoost, Boruta features | 24,420.70 |
| Voting, Boruta features | 23,906.68 |
| Stacking, Boruta features | 23,898.85 |
| **Boruta stacking + smearing correction** | **23,816.76** |
| Neighborhood relativity + quality/size interactions | 24,000.93 |
| Refined parameters + stacking + smearing | 24,362.61 |
| Refined 70% CatBoost / 30% XGBoost blend + smearing | 24,547.08 |
| Added KNN comparable-price feature | 24,425.66 |
| Refined passthrough meta-stacking with KNN + smearing | 24,978.44 |

The best recorded stack used XGBoost with **1,000 trees, learning rate 0.05,
maximum depth 3, and subsample 0.8**, plus CatBoost with **1,000 iterations,
learning rate 0.05, and depth 6**. The meta-regressor was Ridge with `alpha=1.0`.
Both base estimators learned `log1p(SalePrice)`.

Smearing lowered the printed RMSE from 23,898.85 to 23,816.76, a small change of
**82.09**. The correction factor was estimated from the same pooled residuals
used to report the corrected score, so this gain needs separate validation.

## Error concentration and variability

A later shuffled five-fold diagnostic on the extended feature set reported
**mean RMSE 26,690.60 with standard deviation 5,170.22**. This used a different
fold setup and feature recipe from the best printed result. It shows meaningful
variation across folds; it is not a confidence interval for 23,816.76 or a
statistical significance test between models.

The notebook also divided errors from the earlier smeared stack into actual-price
deciles:

| Price decile, low to high | Mean actual price | RMSE | Share of total squared error |
| --- | ---: | ---: | ---: |
| 0 | 85,936.19 | 19,376.13 | 6.62% |
| 1 | 115,286.73 | 13,964.83 | 3.51% |
| 2 | 130,121.58 | 13,007.09 | 2.94% |
| 3 | 141,416.55 | 18,567.20 | 6.24% |
| 4 | 155,136.42 | 16,820.76 | 4.89% |
| 5 | 171,691.56 | 14,395.36 | 3.60% |
| 6 | 187,710.34 | 18,257.34 | 5.88% |
| 7 | 214,336.09 | 18,575.62 | 6.21% |
| 8 | 252,606.19 | 28,325.39 | 13.95% |
| 9 | 357,554.33 | **51,347.87** | **46.16%** |

The most expensive decile contributed nearly half the total squared error.
Together, the top two deciles contributed about **60.11%**. The highest decile
did not exceed the notebook's suggested 50% heuristic by itself, but the table
still shows a substantial concentration of error among expensive homes.

## What worked and what did not

- **House-size and bathroom engineering improved the conservative feature set.**
  `TotalSF` and `TotalBath` survived the stability selection and accompanied a
  substantial reduction in the recorded CatBoost RMSE.
- **Boruta retained useful supporting variables.** Its 31-feature set outperformed
  the much smaller null-importance sets in the recorded experiments.
- **Voting and stacking helped on the Boruta set**, although the gap between
  their RMSEs was only 7.83 and does not establish a reliable ranking.
- **Smearing provided a small recorded improvement**, but its evaluation reused
  the residuals that estimated the correction.
- **Log transformation alone did not improve the original four-feature setup.**
  Its CatBoost result was slightly worse than the earlier raw-target tuned model.
- **Later complexity failed to beat the best recorded result:** extended
  interactions, refined searches, the 70/30 blend, KNN comparables, and the
  passthrough meta-stack all produced higher printed RMSEs.
- **Expensive homes remained the main source of squared error.** More complex
  modeling did not remove that weakness.

## Limits of the recorded evaluation

The notebook produced exploratory CV/OOF scores, not an independent final test
score. Its target encoding was calculated across the full dataset before later
model CV, and its smoothing prior used the full target mean. Feature selection
also used the full target, while neighborhood medians and the KNN scaler were
prepared globally. Consequently, later held-out folds could influence learned
inputs or feature choices. Reusing CV during parameter search and comparing many
variants on the same rows adds selection optimism.

The converted module fits learned steps inside training folds and estimates
smearing on a separate validation partition before scoring an untouched holdout.
Its future scores must be read from that run's generated metrics, not copied from
the historical notebook. The source notebook did not record a separate external
`test.csv` prediction run, a submission score, or an external-test RMSE.
