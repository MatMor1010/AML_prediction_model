import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             average_precision_score, confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

# Run from the repo folder!!
# Run gbm_feature_selection.py first, it gives the features used here

# Outputs go here, tables in csv/
out = "results/modelling/"
os.makedirs(out + "csv", exist_ok = True)


# 1. Import the data and the kept features
df = pd.read_pickle("data/AML_data_transformed.pkl")

y = df["rating"].astype(int)
X = df.drop(columns = "rating").astype(float)

with open("results/gbm_feature_selection/selected_features.json") as f:
    selection = json.load(f)

final = selection["features"]

# Same split as the feature selection, so the test set was never seen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = SEED)

# Check it's the same test set as in the feature selection
assert sorted(X_test.index) == sorted(selection["test_rows"]), "Not the same test set as gbm_feature_selection.py"

# Same CV folds and metrics as the feature selection
cv = StratifiedKFold(5, shuffle = True, random_state = SEED)
scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}


# 2. Tune the GBM without and with SMOTE, same features, folds and grid
# No downsampling, we don't have a lot of data already
print(f"Tuning on {len(final)} features, without and with SMOTE...")

# gbm__ = setting of the gbm step in the pipeline
param_grid = {"gbm__n_estimators": [100, 200, 400],
              "gbm__learning_rate": [0.01, 0.03, 0.1],
              "gbm__max_depth": [2, 3, 4],
              "gbm__subsample": [0.8, 1.0],
              "gbm__min_samples_leaf": [1, 2, 5, 10]}

# SMOTE creates new 1 between real ones until both classes are the same size
# I use SMOTENC to keep the binary features at 0 or 1, plain SMOTE if there are none
binary = [col for col in final if X[col].nunique() == 2]
smote = SMOTENC(categorical_features = binary, random_state = SEED) if binary else SMOTE(random_state = SEED)

# Scale for SMOTE since it uses distances, the GBM doesn't care about scale
# set_output keeps the column names for SMOTENC
scaler = StandardScaler().set_output(transform = "pandas")

# The pipeline is only fitted on the training folds, never on validation or test
approaches = {"no_smote": Pipeline([("scaler", scaler), ("gbm", GradientBoostingClassifier(random_state = SEED))]),
              "smote": Pipeline([("scaler", scaler), ("smote", smote),
                                 ("gbm", GradientBoostingClassifier(random_state = SEED))])}

grids = {}
rows = []
for name, pipeline in approaches.items():
    grid = GridSearchCV(pipeline, param_grid, cv = cv, scoring = scoring, refit = "pr_auc", n_jobs = -1)
    grid.fit(X_train[final], y_train)
    grids[name] = grid

    i = grid.best_index_
    rows.append({"approach": name,
                 "cv_roc_auc_mean": grid.cv_results_["mean_test_roc_auc"][i],
                 "cv_roc_auc_std": grid.cv_results_["std_test_roc_auc"][i],
                 "cv_pr_auc_mean": grid.cv_results_["mean_test_pr_auc"][i],
                 "cv_pr_auc_std": grid.cv_results_["std_test_pr_auc"][i]})

approach_comparison = pd.DataFrame(rows).set_index("approach")

# SMOTE only wins if it's better by more than one standard error
approach_score = approach_comparison["cv_pr_auc_mean"]
smote_gain = approach_score["smote"] - approach_score["no_smote"]
smote_se = approach_comparison.loc["no_smote", "cv_pr_auc_std"] / np.sqrt(5)

approach = "smote" if smote_gain > smote_se else "no_smote"

grid = grids[approach]
model = grid.best_estimator_
cv_final = approach_comparison.loc[approach].to_dict()

# Remove gbm__ to make the outputs easier to read
best_params = {key.replace("gbm__", ""): value for key, value in grid.best_params_.items()}


# 3. Threshold to turn the probability into high risk or not
# Chosen on out-of-fold train predictions, the test set stays untouched
oof_prob = cross_val_predict(model, X_train[final], y_train, cv = cv, method = "predict_proba", n_jobs = -1)[:, 1]
oof_precision, oof_recall, thresholds = precision_recall_curve(y_train, oof_prob)

# The last precision / recall point has no threshold
oof_precision = oof_precision[:-1]
oof_recall = oof_recall[:-1]

# Best F1 threshold, only kept to compare with the cost threshold on test
oof_f1 = 2 * oof_precision * oof_recall / (oof_precision + oof_recall + 1e-12)
threshold_f1 = thresholds[np.argmax(oof_f1)]

# Lowest cost on the train confusion matrix: a missed 1 costs COST_FN, a false alarm costs 1
COST_FN = 4
costs = []
for t in thresholds:
    tn, fp, fn, tp = confusion_matrix(y_train, (oof_prob >= t).astype(int), labels = [0, 1]).ravel()
    costs.append(COST_FN * fn + fp)

threshold = thresholds[np.argmin(costs)]


# 4. Test set, used only once: these are the honest scores
test_prob = model.predict_proba(X_test[final])[:, 1]
test_pred = (test_prob >= threshold).astype(int)

# Same cost on test for both thresholds, to see if the cost one beats F1
test_cost = {}
for name, t in [("cost", threshold), ("f1", threshold_f1)]:
    tn, fp, fn, tp = confusion_matrix(y_test, (test_prob >= t).astype(int), labels = [0, 1]).ravel()
    test_cost[name] = {"threshold": t, "caught": tp, "missed": fn, "false_alarms": fp,
                       "flagged": (test_prob >= t).mean(), "cost": COST_FN * fn + fp}

test_cost = pd.DataFrame(test_cost).T.astype({"caught": int, "missed": int, "false_alarms": int, "cost": int})

precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average = "binary",
                                                           zero_division = 0)
test = {"test_roc_auc": roc_auc_score(y_test, test_prob),
        "test_pr_auc": average_precision_score(y_test, test_prob),
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1}

confusion = pd.DataFrame(confusion_matrix(y_test, test_pred, labels = [0, 1]),
                         index = ["true_0", "true_1"], columns = ["pred_0", "pred_1"])

# Benchmark: logistic regression on the same features and folds, to see if the GBM is worth it
logit = Pipeline([("scaler", StandardScaler()), ("logit", LogisticRegression(max_iter = 1000))])
logit_cv = cross_validate(logit, X_train[final], y_train, cv = cv, scoring = scoring)
logit_prob = logit.fit(X_train[final], y_train).predict_proba(X_test[final])[:, 1]

benchmark = pd.DataFrame({"cv_roc_auc": [cv_final["cv_roc_auc_mean"], logit_cv["test_roc_auc"].mean()],
                          "cv_pr_auc": [cv_final["cv_pr_auc_mean"], logit_cv["test_pr_auc"].mean()],
                          "test_roc_auc": [test["test_roc_auc"], roc_auc_score(y_test, logit_prob)],
                          "test_pr_auc": [test["test_pr_auc"], average_precision_score(y_test, logit_prob)]},
                         index = ["gbm", "logistic_regression"])


# 5. SHAP on the GBM (in log-odds), it needs the scaled data like in training
X_train_scaled = model["scaler"].transform(X_train[final])
explanation = shap.TreeExplainer(model["gbm"])(X_train_scaled)
shap_values = pd.DataFrame(explanation.values, index = X_train.index, columns = final)
feature_values = X_train[final]

# One row per kept feature: importance and direction
shap_summary = pd.DataFrame(index = pd.Index(final, name = "feature"))
shap_summary["mean_abs_shap"] = shap_values.abs().mean()
shap_summary["share_of_total"] = shap_summary["mean_abs_shap"] / shap_summary["mean_abs_shap"].sum()

# + means higher values push towards high risk
shap_summary["direction"] = np.sign(feature_values.corrwith(shap_values)).map({1.0: "+", -1.0: "-"})

# Average push for clients above and below the feature's average
above_avg = feature_values > feature_values.mean()
shap_summary["mean_shap_above_avg"] = shap_values.where(above_avg).mean()
shap_summary["mean_shap_below_avg"] = shap_values.where(~above_avg).mean()

shap_summary = shap_summary.sort_values("mean_abs_shap", ascending = False)


# Export the SHAP table
shap_summary.to_csv(out + "csv/shap_summary.csv")


# SHAP beeswarm for the report, colored with the real feature values
explanation.data = feature_values.values
shap.plots.beeswarm(explanation, max_display = len(final), show = False)
plt.savefig(out + "shap_beeswarm.png", dpi = 150, bbox_inches = "tight")
plt.close()

# Test set plots
RocCurveDisplay.from_predictions(y_test, test_prob, name = "Test", plot_chance_level = True)
plt.savefig(out + "roc_curve.png", dpi = 150, bbox_inches = "tight")
plt.close()

PrecisionRecallDisplay.from_predictions(y_test, test_prob, name = "Test", plot_chance_level = True)
plt.savefig(out + "pr_curve.png", dpi = 150, bbox_inches = "tight")
plt.close()

ConfusionMatrixDisplay(confusion.values, display_labels = [0, 1]).plot(cmap = "Blues")
plt.savefig(out + "confusion_matrix.png", dpi = 150, bbox_inches = "tight")
plt.close()


# Summary, printed and saved
scores = pd.DataFrame({"cv_mean": [cv_final["cv_roc_auc_mean"], cv_final["cv_pr_auc_mean"]],
                       "cv_std": [cv_final["cv_roc_auc_std"], cv_final["cv_pr_auc_std"]],
                       "test": [test["test_roc_auc"], test["test_pr_auc"]]},
                      index = ["roc_auc", "pr_auc"])

summary = f"""Data: AML_data_transformed.pkl, {df.shape[0]} rows
Train/test rows: {len(X_train)}/{len(X_test)}, positive rate {y.mean():.1%}
Features: {len(final)} kept by gbm_feature_selection.py ({selection['chosen_set']})

Approaches compared with the tuned GBM (CV on train):
{approach_comparison.round(4).to_string()}

Chosen: {approach}, SMOTE gain in pr_auc {smote_gain:+.4f}, one standard error {smote_se:.4f}
(SMOTE only wins if its gain is more than one standard error)

Best params: {best_params}

Final model ({approach}):
{scores.round(3).to_string()}

Benchmark, same features and folds:
{benchmark.round(3).to_string()}

At threshold {threshold:.3f} (lowest train cost, a missed 1 = {COST_FN} false alarms), on test: precision {precision:.3f}, recall {recall:.3f}, F1 {f1:.3f}

Cost threshold vs best F1 threshold, both chosen on train, compared on test (cost = {COST_FN} x missed + false alarms):
{test_cost.round(3).to_string()}

Test confusion matrix:
{confusion.to_string()}

Kept features (SHAP in log-odds; + means higher values push towards high risk):
{shap_summary.round(3).to_string()}

Note: CV scores are slightly optimistic because the features were picked on the same folds;
the test scores are the honest numbers."""

print(summary)

with open(out + "summary.txt", "w", encoding = "utf-8") as f:
    f.write(summary)
