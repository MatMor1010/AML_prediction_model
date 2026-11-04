# GBM feature selection

`src/gbm_feature_selection.py` picks a small set of features for the AML risk model, trains a gradient boosting model on them, checks it on clients it has never seen, and saves everything to `results/gbm_feature_selection/`.

## Run it

From the root of the repo:

```
uv run python src/gbm_feature_selection.py
```

It takes about 4 minutes. The results are printed and also saved in `summary.txt`.

## Settings

All settings are at the top of the file. You shouldn't need to edit anything else.

| Setting | What it does |
|---|---|
| `DATA_PATH` | The data file (`.pkl`, `.csv`, `.parquet` or `.xlsx`) |
| `TARGET` | The column to predict (`rating`, 1 = high risk) |
| `DROP_COLS` | Columns that are not features, e.g. IDs |
| `RUN_NAME` | Name of the output folder in `results/` |
| `MAX_CANDIDATES` | How many features go into selection (40) |
| `SELECTION_METRIC` | The score selection tries to maximize: PR AUC (`average_precision`) or ROC AUC (`roc_auc`) |
| `TARGET_RECALL` | `None` picks the threshold with the best F1. Set it to e.g. `0.8` to catch at least 80% of high-risk clients |
| `SELECTION_MODEL` | The small model used to pick features |
| `MODEL`, `PARAM_GRID` | The final model and the settings it is tuned over |

Every column except `TARGET` and `DROP_COLS` is a feature. If you add or remove columns in the data, the script and all its outputs adapt on their own.

## How it works

On the current data:

```
145 features  →  pre-screen  →  40 candidates  →  forward / stepwise  →  8 features  →  tune  →  test
```

1. **Split.** 20% of clients are set aside as a test set and not touched until step 7. Everything else uses 5-fold cross-validation (CV) on the other 80%.
2. **Pre-screen.** The small model is fitted once on all features. SHAP measures how much each feature moves its predictions, and the 40 most important features become the candidates.
3. **Forward and stepwise selection.** Both start with no features and add the one that improves the CV score most, one at a time, until all 40 are in.
   - *Forward* never removes a feature once it's in.
   - *Stepwise* checks after every addition whether dropping an earlier feature now gives a better score.

   Each one then keeps the smallest set whose score is close to the best it found. Extra features add little and make the model harder to explain.
4. **Pick the final set.** Forward and stepwise are compared on CV. The higher score wins, but if the two are within one standard error, the smaller set wins.
5. **Tune.** The final model is tuned on just those features, by trying every combination in `PARAM_GRID` (32 combinations × 5 folds).
6. **Choose the threshold.** The model gives a probability, and the threshold turns it into high risk yes or no. It's chosen on the training data only, never on the test set.
7. **Test.** The final model is scored once on the test set: ROC AUC, PR AUC, precision, recall, F1 and a confusion matrix.
8. **Explain.** SHAP shows how much each kept feature pushes each client towards high or low risk. The results are summarized in `shap_summary.csv`.

## Why a smaller model is used for selection

Steps 2 to 4 use `SELECTION_MODEL`: a deliberately small gradient boosting model with fixed settings (100 trees, each only 2 levels deep). The tuned model is only fitted in step 5. This helps in four ways.

- **Speed.** Selection fits a lot of models. Forward selection on 40 candidates tries 40 + 39 + … + 1 = 820 feature sets, each on 5 folds, so 4,100 fits, and stepwise does even more. A small model fits quickly, so the whole script runs in minutes. Tuning the model inside every one of those steps would multiply the work by 32.
- **A fair comparison.** The small model's settings never change, so when the score goes up or down, it's because of the features and not because of the settings.
- **Less overfitting.** With only 470 training clients, a big model can learn patterns that are just noise. Shallow trees can't do that as easily, so a feature that helps the small model is more likely to really matter.
- **Good enough to rank features.** At this stage we only need to know *which* features matter, not get the best possible score. Getting the best score is the job of the tuning step, which is cheap because it only sees the final 8 features.

The pre-screen uses the same idea. One fit of the small model ranks all 145 features. Without it, forward selection on all 145 would need 10,585 feature sets × 5 folds, so about 53,000 fits.

The trade-off: the features chosen are the best ones *for the small model*. The tuned model might have preferred slightly different ones.

## Outputs

```
results/gbm_feature_selection/
├── summary.txt                     Short report of the whole run
├── metrics.json                    Chosen set, best settings, CV and test scores, threshold
├── selection_path.png              CV score for each number of features, forward and stepwise
├── roc_curve.png                   Test ROC curve
├── pr_curve.png                    Test precision-recall curve
├── confusion_matrix.png            Test confusion matrix
└── csv/
    ├── feature_summary.csv         One row per feature: candidate / forward / stepwise / kept, plus SHAP
    ├── selection_comparison.csv    CV scores for all features, the candidates, forward and stepwise
    ├── selection_path_forward.csv  CV score for each number of features
    ├── selection_path_stepwise.csv
    ├── predictions.csv             Test clients: true label, probability and prediction
    └── shap_summary.csv            SHAP results for the kept features
```

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

From a notebook in `data_analysis/`:

```python
import sys
sys.path.append("..")
from src.gbm_feature_selection import load_results

metrics, tables = load_results("gbm_feature_selection")

metrics["test_pr_auc"]
tables["shap_summary"]   # every CSV in csv/ is in tables, by file name
```

## Things to know

- **The test scores are the honest ones.** The CV scores of the selected features are slightly too optimistic, because selection used the same folds.
- **The fitted model isn't saved.** Running the script again with the same settings gives exactly the same model, because `SEED` fixes all the randomness.
- **To change the model**, edit `SELECTION_MODEL`, `MODEL` and `PARAM_GRID`. It has to stay a tree-based model, because SHAP uses `TreeExplainer`.
