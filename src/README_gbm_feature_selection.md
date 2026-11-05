# GBM feature selection and modelling

Two scripts, run one after the other:

1. `src/gbm_feature_selection.py` picks a small set of features for the AML risk model and saves them to `results/gbm_feature_selection/`.
2. `src/modelling.py` trains a gradient boosting model (GBM) on those features three ways (plain, with class weights, with SMOTE), keeps the best one, checks it on clients it has never seen, and saves everything to `results/modelling/`.

## Run it

From the root of the repo, after `src/Data_transformation.py` (both scripts read `data/AML_data_transformed.pkl`):

```
uv run python src/gbm_feature_selection.py
uv run python src/modelling.py
```

`modelling.py` reads the features from `results/gbm_feature_selection/selected_features.json`, so rerun it every time you rerun the selection. Each script prints its results and saves them in its own `summary.txt`.

If you just cloned the repo, run `uv sync` first: SMOTE needs the `imbalanced-learn` package.

## What you can change

Both scripts read top to bottom like `Data_transformation.py`, and the values are written where they're used:

| What | Where |
|---|---|
| The small model used to pick features | `small_gbm`, `gbm_feature_selection.py` |
| How many features go into selection (50) | `shap_rank.index[:50]`, step 2 of `gbm_feature_selection.py` |
| The final model's tuning grid | `param_grid`, step 2 of `modelling.py` (keep the `gbm__` in front of each setting) |
| The rebalancing rule (weights or SMOTE only win if more than one standard error better) | `approach = ...`, step 2 of `modelling.py` |
| The threshold rule (lowest cost, a missed high-risk client costs as much as 4 false alarms) | `COST_FN`, step 3 of `modelling.py` |

Every column except `rating` is a feature. If you add or remove columns in `Data_transformation.py`, both scripts and all their outputs adapt on their own.

If you change `SEED`, change it in both scripts. `modelling.py` checks that it uses the same test set as the selection and stops with an error if not.

## How it works

On the current data:

```
gbm_feature_selection.py:  145 features  →  drop near-duplicates  →  62  →  pre-screen  →  50 candidates  →  forward / stepwise  →  kept features
modelling.py:              kept features  →  tune plain / weights / SMOTE  →  threshold  →  test  →  drop-one check  →  SHAP
```

**Both scripts.** 20% of clients are set aside as a test set. Both scripts make the same split (same `SEED`), so the selection never sees the test clients. Everything else uses 5-fold cross-validation (CV) on the other 80%, with the same folds in both scripts.

`gbm_feature_selection.py`:

1. **Split and drop near-duplicates.** Many features measure the same thing (e.g. the 3, 6 and 12 month versions, or `txn_wires_in_*` vs `txn_cross_border_in_*`). On the training set, any feature with a rank correlation of 0.95 or more with a stronger one is dropped, where "stronger" means a better AUC on its own. This takes the data from 145 to 62 features. Twins otherwise make the selection pick one of them more or less at random, and split the SHAP importance between them. The dropped features and the one kept instead are listed in `duplicates.csv`.
2. **Pre-screen.** The small model is fitted once on all features. SHAP measures how much each feature moves its predictions, and the 50 most important features become the candidates.
3. **Forward and stepwise selection.** Both start with no features and add the one that improves the CV score most, one at a time, until all 50 are in.
   - *Forward* never removes a feature once it's in.
   - *Stepwise* checks after every addition whether dropping an earlier feature now gives a better score.

   Each one then keeps the smallest set whose score is within one standard error of the best it found. Extra features add little and make the model harder to explain.
4. **Pick the final set.** Forward and stepwise are compared on CV, and the one with the higher PR AUC is kept. The kept features are saved to `selected_features.json`.

`modelling.py`:

1. **Load.** The data, the kept features, and the same split. It checks that the test set is the same one the selection never saw.
2. **Tune three ways: plain, class weights, SMOTE.** The model is tuned on the kept features by trying every combination in `param_grid` (216 combinations × 5 folds). This is done three times, with the same features, folds and grid, so the only difference is how the imbalance is handled (see below). Weights or SMOTE only win if they beat the plain model by more than one standard error, otherwise the simpler model is kept.
3. **Choose the threshold.** The model gives a probability, and the threshold turns it into high risk yes or no. It's chosen on out-of-fold predictions of the training data, never on the test set: the threshold with the lowest cost, where a missed high-risk client costs `COST_FN` = 4 and a false alarm costs 1. The best-F1 threshold is also computed, just to compare. Picking the threshold on the same model also handles rebalancing: a model trained on 50/50 data gives higher probabilities.
4. **Test.** The final model is scored once on the test set: ROC AUC, PR AUC, precision, recall, F1 and a confusion matrix. A logistic regression on the same features and folds is the benchmark.
5. **Drop-one check.** The tuned model is refitted on the training folds without each kept feature in turn, with the same settings. The drop in CV PR AUC shows how much the model needs that feature. It's useful for a feature with an odd direction, or one the model leans on a lot.
6. **Explain.** SHAP shows how much each kept feature pushes each client towards high or low risk. The results are summarized in `shap_summary.csv`.

## Why a smaller model is used for selection

The selection uses `small_gbm`: a deliberately small gradient boosting model with fixed settings (100 trees, each only 2 levels deep). The tuned model only comes in `modelling.py`. This helps in four ways.

- **Speed.** Selection fits a lot of models. Forward selection on 50 candidates tries 50 + 49 + … + 1 = 1,275 feature sets, each on 5 folds, so 6,375 fits, and stepwise does even more. A small model fits quickly. Tuning the model inside every one of those steps would multiply the work by 216.
- **A fair comparison.** The small model's settings never change, so when the score goes up or down, it's because of the features and not because of the settings.
- **Less overfitting.** With only 470 training clients, a big model can learn patterns that are just noise. Shallow trees can't do that as easily, so a feature that helps the small model is more likely to really matter.
- **Good enough to rank features.** At this stage we only need to know *which* features matter, not get the best possible score. Getting the best score is the job of the tuning step, which is cheap because it only sees the kept features.

The pre-screen uses the same idea. One fit of the small model ranks all 62 remaining features, and the top 50 go into selection.

The trade-off: the features chosen are the best ones *for the small model*. The tuned model might have preferred slightly different ones.

## Class weights and SMOTE

About a third of clients are high risk. Two ways to rebalance are tested against the plain model. There's no downsampling, because with 588 clients we can't afford to throw any away.

**Class weights.** Each training client gets a weight, `"balanced"`: a high-risk client counts about twice as much as a non-high one in the fit, so both classes weigh the same. `GridSearchCV` cuts the weights along the folds like the data, and the validation folds are scored without weights.

**SMOTE.** SMOTE creates synthetic high-risk clients by picking a real one and placing a new point between it and one of its nearest high-risk neighbours, until both classes are the same size. Only high-risk clients are added; the others are left as they are.

Three things make the SMOTE comparison fair:

- **SMOTE only touches the training folds.** It sits inside an `imblearn` `Pipeline`, so in every CV split it upsamples the 4 training folds only. The validation fold, and later the test set, contain real clients only. Upsampling before the split would put copies of the same clients on both sides and make the scores look better than they are.
- **Binary features stay binary.** Plain SMOTE would give values like `prod_ted = 0.37`. `SMOTENC` takes the most common value among the neighbours for the binary features (any feature with exactly 2 values).
- **Selection is done once, without SMOTE.** All approaches are tuned on the same kept features, so the comparison is only about the rebalancing.

All three approaches are scored with the same metric (PR AUC) on the same real validation folds, so their scores can be compared directly.

## Outputs

```
results/gbm_feature_selection/
├── summary.txt                     Short report of the selection
├── selected_features.json          Kept features and test rows, read by modelling.py
├── selection_path.png              CV score for each number of features, forward and stepwise
└── csv/
    ├── feature_summary.csv         One row per feature: pre-screen SHAP, candidate / forward / stepwise / kept
    ├── selection_comparison.csv    CV scores for all features, the candidates, forward and stepwise
    ├── duplicates.csv              Dropped near-duplicates and the feature kept instead
    ├── selection_path_forward.csv  CV score for each number of features
    └── selection_path_stepwise.csv

results/modelling/
├── summary.txt                     Short report: rebalancing comparison, best settings, CV and test scores,
│                                   benchmark, thresholds, drop-one check, SHAP
├── shap_beeswarm.png               SHAP plot of the kept features, for the report
├── roc_curve.png                   Test ROC curve
├── pr_curve.png                    Test precision-recall curve
├── confusion_matrix.png            Test confusion matrix
└── csv/
    ├── shap_summary.csv            SHAP results for the kept features
    └── drop_one.csv                CV scores without each kept feature
```

### shap_beeswarm.png

One row per kept feature, most important at the top. Each dot is a training client: its position is how much that feature pushes the client towards high risk (right) or low risk (left), in log-odds, and its color is the client's value of the feature (red = high, blue = low). For example, red dots on the right mean that high values increase the risk.

### shap_summary.csv

One row per kept feature. SHAP values are in log-odds: a positive value pushes a client towards high risk.

| Column | What it means |
|---|---|
| `mean_abs_shap` | How much the feature moves predictions on average: its importance |
| `share_of_total` | Its share of the total importance of all kept features |
| `direction` | `+` if higher values push towards high risk, `-` if they push towards low risk |
| `mean_shap_above_avg` | The average push for clients with an above-average value of the feature |
| `mean_shap_below_avg` | The average push for clients with a below-average value |

For example, if `mean_shap_above_avg` is 0.8, clients with above-average values of that feature get, on average, 0.8 added to their log-odds of being high risk.

## Use the results in a notebook

Don't import the scripts: importing one reruns it. Read the saved files instead. From a notebook in `data_analysis/`:

```python
import pandas as pd

shap_summary = pd.read_csv("../results/modelling/csv/shap_summary.csv", index_col = 0)
```

Every CSV in the `csv/` folders can be read the same way.

## Things to know

- **The test scores are the honest ones.** The CV scores are slightly too optimistic, because the features were picked and the model tuned on the same folds.
- **The fitted model isn't saved.** Running the scripts again gives exactly the same model, because `SEED` fixes all the randomness.
- **To change the model**, edit `small_gbm` in the selection, and the two GBMs in `approaches` and `param_grid` in `modelling.py`. It has to stay a tree-based model, because SHAP uses `TreeExplainer`. Keep the `"gbm"` step name, SHAP uses it to get the model out of the pipeline.
- **SHAP explains real clients only.** It's computed on the GBM inside the pipeline, on the real training clients, not on the synthetic ones.
- **A rare random crash on Windows.** About 1 run in 8, a script can stop with `BrokenProcessPool` / `WinError 6`. It comes from the parallel workers (`n_jobs = -1`), not from the code. Just rerun it.
