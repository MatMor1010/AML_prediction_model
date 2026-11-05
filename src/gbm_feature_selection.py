import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

SEED = 42

# Run from the repo folder!!
# Picks the features used in modelling.py

# Outputs go here, tables in csv/
out = "results/gbm_feature_selection/"
os.makedirs(out + "csv", exist_ok = True)


# 1. Import the data and split it
df = pd.read_pickle("data/AML_data_transformed.pkl")

# Every column except the rating is a feature
y = df["rating"].astype(int)
X = df.drop(columns = "rating").astype(float)

# Let's check if the target has enough 1
print(f"We have {(y == 1).sum() / y.count():.2%} of the dataset with target 1.")

# Test set (20%), not used here, only at the end of modelling.py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = SEED)

# Drop the near-duplicates (|spearman| >= 0.95) on the train set, keep the one with the best AUC alone
# |AUC - 0.5| so a feature that lowers the risk counts too
power = (X_train.apply(lambda col: roc_auc_score(y_train, col)) - 0.5).abs().sort_values(ascending = False)
corr = X_train.rank().corr().abs()

no_duplicates = []
duplicates = {}
for col in power.index:
    twins = corr.loc[col, no_duplicates]
    if (twins >= 0.95).any():
        duplicates[col] = twins.idxmax()
    else:
        no_duplicates.append(col)

X_train = X_train.drop(columns = list(duplicates))
print(f"{len(duplicates)} near-duplicates dropped, {X_train.shape[1]} features left.")

# 5-fold CV on the train set, stratified so each fold has the same share of 1
cv = StratifiedKFold(5, shuffle = True, random_state = SEED)

# We maximize PR AUC since it focuses on the 1, ROC AUC is just tracked
scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}

# Small GBM with fixed settings: fast, and only the features change, not the parameters
small_gbm = GradientBoostingClassifier(n_estimators = 100, max_depth = 2, learning_rate = 0.1,
                                       subsample = 0.8, random_state = SEED)


# 2. Pre-screen with SHAP, the top 50 become the candidates (selection on all would be too slow)
small_gbm.fit(X_train, y_train)
shap_prescreen = shap.TreeExplainer(small_gbm)(X_train).values

shap_rank = pd.Series(np.abs(shap_prescreen).mean(axis = 0), index = X_train.columns)
shap_rank = shap_rank.sort_values(ascending = False)
candidates = list(shap_rank.index[:50])


# 3. Forward and stepwise: add the feature that helps PR AUC the most, one at a time
# Forward never drops a feature, stepwise can drop one after each addition
chosen = {}
paths = {}

for name, floating in [("forward", False), ("stepwise", True)]:
    print(f"Running {name} selection on {len(candidates)} candidates...")

    selector = SequentialFeatureSelector(small_gbm, k_features = "best", forward = True,
                                         floating = floating, scoring = "average_precision",
                                         cv = cv, n_jobs = -1)
    selector.fit(X_train[candidates], y_train)

    # Best CV score for each number of features
    path = pd.DataFrame.from_dict(selector.get_metric_dict(), orient = "index")

    # One standard error rule: smallest set within one std error of the best score
    best = path["avg_score"].idxmax()
    k = path.index[path["avg_score"] >= path.loc[best, "avg_score"] - path.loc[best, "std_err"]].min()
    chosen[name] = list(path.loc[k, "feature_names"])

    path["feature_names"] = path["feature_names"].map("; ".join)
    paths[name] = path[["avg_score", "std_dev", "std_err", "feature_names"]].rename_axis("n_features")


# 4. Compare the feature sets with the small GBM and pick the final one
feature_sets = {"all": list(X_train.columns),
                "candidates": candidates,
                "forward": chosen["forward"],
                "stepwise": chosen["stepwise"]}

rows = []
for name, cols in feature_sets.items():
    scores = cross_validate(small_gbm, X_train[cols], y_train, cv = cv, scoring = scoring, n_jobs = -1)

    rows.append({"set": name,
                 "n_features": len(cols),
                 "roc_auc_mean": scores["test_roc_auc"].mean(),
                 "roc_auc_std": scores["test_roc_auc"].std(),
                 "pr_auc_mean": scores["test_pr_auc"].mean(),
                 "pr_auc_std": scores["test_pr_auc"].std()})

comparison = pd.DataFrame(rows).set_index("set")

# Keep the best PR AUC between forward and stepwise
final_name = comparison.loc[["forward", "stepwise"], "pr_auc_mean"].idxmax()
final = chosen[final_name]


# Export the tables
# One row per feature: pre-screen SHAP and where it got in the selection
features = pd.DataFrame(index = shap_rank.index.rename("feature"))
features["prescreen_shap"] = shap_rank
features["candidate"] = features.index.isin(candidates)
features["forward"] = features.index.isin(chosen["forward"])
features["stepwise"] = features.index.isin(chosen["stepwise"])
features["kept"] = features.index.isin(final)
features = features.sort_values(["kept", "prescreen_shap"], ascending = False)

paths["forward"].to_csv(out + "csv/selection_path_forward.csv")
paths["stepwise"].to_csv(out + "csv/selection_path_stepwise.csv")
comparison.to_csv(out + "csv/selection_comparison.csv")
features.to_csv(out + "csv/feature_summary.csv")

# Dropped near-duplicates and the feature kept instead
pd.Series(duplicates, name = "duplicate_of").rename_axis("feature").to_csv(out + "csv/duplicates.csv")

# Kept features for modelling.py, with the test rows to check the split
selection = {"chosen_set": final_name,
             "features": final,
             "test_rows": X_test.index.tolist()}

with open(out + "selected_features.json", "w") as f:
    json.dump(selection, f, indent = 2)


# Selection path, the dashed lines are the number of features kept
colors = {"forward": "#2a78d6", "stepwise": "#eb6834"}

for name, path in paths.items():
    mean = path["avg_score"]
    std = path["std_dev"]

    plt.plot(path.index, mean, color = colors[name], linewidth = 2, label = f"{name} ({len(chosen[name])} chosen)")
    plt.fill_between(path.index, mean - std, mean + std, color = colors[name], alpha = 0.15)
    plt.axvline(len(chosen[name]), color = colors[name], linestyle = "--", linewidth = 1)

plt.title("Selection path")
plt.xlabel("Number of features")
plt.ylabel("CV PR AUC (mean ± std)")
plt.legend()
plt.savefig(out + "selection_path.png", dpi = 150, bbox_inches = "tight")
plt.close()


# Summary, printed and saved
summary = f"""
Amount of variables:
Without duplicates:    {X_train.shape[1]} (of {X.shape[1]})
Chosen by forward:     {len(chosen['forward'])}
Chosen by stepwise:    {len(chosen['stepwise'])}
Kept:                  {len(final)} ({final_name}, best CV pr_auc)

Feature sets compared with the small GBM (CV on train):
{comparison.round(4).to_string()}

Kept features (pre-screen mean |SHAP|):
{features.loc[final, 'prescreen_shap'].round(4).to_string()}
"""

print(summary)

with open(out + "summary.txt", "w", encoding = "utf-8") as f:
    f.write(summary)


# From here I'll look at the feature selected in data_analysis/feature_selection.ipynb
