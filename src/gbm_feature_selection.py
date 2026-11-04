import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)


# Settings: the only part of the file that should need editing
DATA_PATH = "data/AML_data_transformed.pkl"  # .pkl, .csv, .parquet or .xlsx
TARGET = "rating"
DROP_COLS = []  # e.g. ID columns
RUN_NAME = "gbm_feature_selection"  # outputs go to results/RUN_NAME/

TEST_SIZE = 0.2
N_SPLITS = 5
SEED = 42

SELECTION_METRIC = "average_precision"  # PR AUC; "roc_auc" also works
MAX_CANDIDATES = 40  # features passed into forward/stepwise selection
TARGET_RECALL = None  # None = threshold that maximizes F1

# Small, fixed model used during selection to keep it fast
SELECTION_MODEL = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=2,
    learning_rate=0.1,
    subsample=0.8,
    random_state=SEED,
)

# Final model and the grid it is tuned over
MODEL = GradientBoostingClassifier(random_state=SEED)
PARAM_GRID = {
    "n_estimators": [100, 300],
    "learning_rate": [0.03, 0.1],
    "max_depth": [2, 3],
    "subsample": [0.8, 1.0],
    "min_samples_leaf": [5, 20],
}


ROOT = Path(__file__).resolve().parents[1]
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
METRIC = "pr_auc" if SELECTION_METRIC == "average_precision" else "roc_auc"


def load_results(run_name=RUN_NAME):
    folder = ROOT / "results" / run_name

    metrics = json.loads((folder / "metrics.json").read_text())
    tables = {csv.stem: pd.read_csv(csv, index_col=0) for csv in sorted((folder / "csv").glob("*.csv"))}

    return metrics, tables


def cv_scores(model, X, y, cv):
    scores = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=-1)

    summary = {}
    for name in SCORING:
        summary[f"{name}_mean"] = scores[f"test_{name}"].mean()
        summary[f"{name}_std"] = scores[f"test_{name}"].std()

    return summary


def save_plot(path):
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    start = time.perf_counter()
    plt.switch_backend("agg")

    # PNGs, summary and metrics go in results/RUN_NAME/, tables in results/RUN_NAME/csv/
    out = ROOT / "results" / RUN_NAME
    csv_dir = out / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)


    # 1. Load, encode and split the data
    data_path = ROOT / DATA_PATH
    readers = {".pkl": pd.read_pickle, ".csv": pd.read_csv, ".parquet": pd.read_parquet, ".xlsx": pd.read_excel}
    df = readers[data_path.suffix.lower()](data_path)

    # Every column except TARGET and DROP_COLS is a feature
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET, *DROP_COLS])

    # Category columns become their values (0/1 here); text columns are one-hot encoded
    X = X.apply(lambda col: col.astype(object).infer_objects() if col.dtype == "category" else col)
    X = pd.get_dummies(X, dtype=float).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)


    # 2. Pre-screen: keep the MAX_CANDIDATES features with the highest mean |SHAP|
    prescreen_model = clone(SELECTION_MODEL).fit(X_train, y_train)
    prescreen_shap = shap.TreeExplainer(prescreen_model)(X_train).values

    shap_rank = pd.Series(np.abs(prescreen_shap).mean(axis=0), index=X.columns)
    shap_rank = shap_rank.sort_values(ascending=False)
    candidates = list(shap_rank.index[:MAX_CANDIDATES])


    # 3. Forward and stepwise selection (stepwise can also drop an earlier feature after each addition)
    chosen = {}
    paths = {}

    for name, floating in [("forward", False), ("stepwise", True)]:
        print(f"Running {name} selection on {len(candidates)} candidates...")

        selector = SequentialFeatureSelector(
            SELECTION_MODEL,
            k_features="parsimonious",
            forward=True,
            floating=floating,
            scoring=SELECTION_METRIC,
            cv=cv,
            n_jobs=-1,
        )
        selector.fit(X_train[candidates], y_train)
        chosen[name] = list(selector.k_feature_names_)

        # Best CV score for each number of features
        path = pd.DataFrame.from_dict(selector.get_metric_dict(), orient="index")
        path["feature_names"] = path["feature_names"].map("; ".join)
        paths[name] = path[["avg_score", "std_dev", "std_err", "feature_names"]].rename_axis("n_features")


    # 4. Compare feature sets with the selection model
    feature_sets = {"all": list(X.columns), "candidates": candidates, **chosen}

    rows = []
    for name, cols in feature_sets.items():
        scores = cv_scores(SELECTION_MODEL, X_train[cols], y_train, cv)
        rows.append({"set": name, "n_features": len(cols), **scores})

    comparison = pd.DataFrame(rows).set_index("set")

    # Final set: the better of forward/stepwise, or the smaller one if they are within one standard error
    score = comparison[f"{METRIC}_mean"]
    best, other = ("forward", "stepwise") if score["forward"] >= score["stepwise"] else ("stepwise", "forward")
    one_se = comparison.loc[best, f"{METRIC}_std"] / np.sqrt(N_SPLITS)

    if score[best] - score[other] <= one_se:
        final_name = best if len(chosen[best]) <= len(chosen[other]) else other
        why = "within one standard error, so the smaller set wins"
    else:
        final_name = best
        why = "more than one standard error apart, so the higher score wins"

    final = chosen[final_name]
    reason = f"CV {METRIC} {best} {score[best]:.4f} vs {other} {score[other]:.4f}, one standard error {one_se:.4f}"


    # 5. Tune the final model
    print(f"Tuning on {len(final)} features...")

    grid = GridSearchCV(MODEL, PARAM_GRID, cv=cv, scoring=SCORING, refit=METRIC, n_jobs=-1)
    grid.fit(X_train[final], y_train)
    model = grid.best_estimator_

    cv_final = {}
    for name in SCORING:
        cv_final[f"cv_{name}_mean"] = grid.cv_results_[f"mean_test_{name}"][grid.best_index_]
        cv_final[f"cv_{name}_std"] = grid.cv_results_[f"std_test_{name}"][grid.best_index_]


    # 6. Threshold from out-of-fold probabilities on train (the test set is not used here)
    oof_prob = cross_val_predict(model, X_train[final], y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    oof_precision, oof_recall, thresholds = precision_recall_curve(y_train, oof_prob)

    # The last precision/recall point has no threshold
    oof_precision = oof_precision[:-1]
    oof_recall = oof_recall[:-1]

    if TARGET_RECALL is None:
        oof_f1 = 2 * oof_precision * oof_recall / np.maximum(oof_precision + oof_recall, 1e-12)
        threshold = thresholds[np.argmax(oof_f1)]
        rule = "max out-of-fold F1"
    else:
        threshold = thresholds[oof_recall >= TARGET_RECALL].max()
        rule = f"highest with out-of-fold recall >= {TARGET_RECALL}"


    # 7. Test set, used once
    test_prob = model.predict_proba(X_test[final])[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average="binary", zero_division=0
    )
    test = {
        "test_roc_auc": roc_auc_score(y_test, test_prob),
        "test_pr_auc": average_precision_score(y_test, test_prob),
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
    }

    confusion = pd.DataFrame(
        confusion_matrix(y_test, test_pred, labels=[0, 1]),
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )


    # 8. SHAP on the final model, explaining the training rows (values are in log-odds)
    explanation = shap.TreeExplainer(model)(X_train[final])
    shap_values = pd.DataFrame(explanation.values, index=X_train.index, columns=final)
    feature_values = X_train[final]

    # SHAP summary, one row per kept feature: how much it moves predictions and in which direction
    shap_summary = pd.DataFrame(index=pd.Index(final, name="feature"))
    shap_summary["mean_abs_shap"] = shap_values.abs().mean()
    shap_summary["share_of_total"] = shap_summary["mean_abs_shap"] / shap_summary["mean_abs_shap"].sum()

    # + means higher values of the feature push towards high risk
    shap_summary["direction"] = np.sign(feature_values.corrwith(shap_values)).map({1.0: "+", -1.0: "-"})

    # Average push for clients with above-average and below-average values of the feature
    above_avg = feature_values > feature_values.mean()
    shap_summary["mean_shap_above_avg"] = shap_values.where(above_avg).mean()
    shap_summary["mean_shap_below_avg"] = shap_values.where(~above_avg).mean()

    shap_summary = shap_summary.sort_values("mean_abs_shap", ascending=False)

    # Feature table, one row per feature, kept features first
    features = pd.DataFrame(index=shap_rank.index.rename("feature"))
    features["candidate"] = features.index.isin(candidates)
    features["forward"] = features.index.isin(chosen["forward"])
    features["stepwise"] = features.index.isin(chosen["stepwise"])
    features["kept"] = features.index.isin(final)
    features["mean_abs_shap"] = shap_summary["mean_abs_shap"]
    features["direction"] = shap_summary["direction"]
    features = features.sort_values(["kept", "mean_abs_shap"], ascending=False)


    # Save tables
    predictions = pd.DataFrame({"y_true": y_test, "y_prob": test_prob, "y_pred": test_pred})

    tables = {
        "selection_path_forward": paths["forward"],
        "selection_path_stepwise": paths["stepwise"],
        "selection_comparison": comparison,
        "predictions": predictions.rename_axis("row"),
        "feature_summary": features,
        "shap_summary": shap_summary,
    }
    for name, table in tables.items():
        table.to_csv(csv_dir / f"{name}.csv")


    # Save metrics
    metrics = {
        "chosen_set": final_name,
        "n_features": len(final),
        "best_params": grid.best_params_,
        **cv_final,
        **test,
        "threshold": threshold,
        "n_rows": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "positive_rate": y.mean(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))


    # Selection path: CV score by number of features, with a band of +/- one std
    colors = {"forward": "#2a78d6", "stepwise": "#eb6834"}

    for name, path in paths.items():
        mean = path["avg_score"]
        std = path["std_dev"]

        plt.plot(path.index, mean, color=colors[name], linewidth=2, label=f"{name} ({len(chosen[name])} chosen)")
        plt.fill_between(path.index, mean - std, mean + std, color=colors[name], alpha=0.15)
        plt.axvline(len(chosen[name]), color=colors[name], linestyle="--", linewidth=1)

    plt.title("Selection path")
    plt.xlabel("Number of features")
    plt.ylabel(f"CV {SELECTION_METRIC} (mean ± std)")
    plt.legend()
    save_plot(out / "selection_path.png")


    # Test set plots
    RocCurveDisplay.from_predictions(y_test, test_prob, name="Test", plot_chance_level=True)
    save_plot(out / "roc_curve.png")

    PrecisionRecallDisplay.from_predictions(y_test, test_prob, name="Test", plot_chance_level=True)
    save_plot(out / "pr_curve.png")

    ConfusionMatrixDisplay(confusion.values, display_labels=[0, 1]).plot(cmap="Blues")
    save_plot(out / "confusion_matrix.png")


    # Summary
    scores = pd.DataFrame(
        {
            "cv_mean": [cv_final[f"cv_{name}_mean"] for name in SCORING],
            "cv_std": [cv_final[f"cv_{name}_std"] for name in SCORING],
            "test": [test[f"test_{name}"] for name in SCORING],
        },
        index=list(SCORING),
    )
    runtime = (time.perf_counter() - start) / 60

    lines = [
        f"Data: {data_path.name}, {df.shape[0]} rows x {df.shape[1]} columns",
        f"Train/test rows: {len(X_train)}/{len(X_test)}, positive rate {y.mean():.1%}",
        "",
        f"Features given:        {X.shape[1]}",
        f"Pre-screened (SHAP):   {len(candidates)}",
        f"Chosen by forward:     {len(chosen['forward'])}",
        f"Chosen by stepwise:    {len(chosen['stepwise'])}",
        f"Kept:                  {len(final)} ({final_name})",
        "",
        f"Why {final_name}: {reason}, {why}.",
        "",
        "Feature sets compared with the selection model (CV on train):",
        comparison.round(4).to_string(),
        "",
        "Kept features (SHAP in log-odds; + means higher values push towards high risk):",
        shap_summary.round(3).to_string(),
        "",
        f"Best params: {grid.best_params_}",
        "",
        "Final model:",
        scores.round(3).to_string(),
        "",
        f"At threshold {threshold:.3f} ({rule}): "
        f"precision {precision:.3f}, recall {recall:.3f}, F1 {f1:.3f}",
        "",
        "Test confusion matrix:",
        confusion.to_string(),
        "",
        "Note: CV scores of the selected set are slightly optimistic because selection used those folds;",
        "the test scores are the honest numbers.",
        "",
        f"Runtime: {runtime:.1f} min",
    ]

    summary = "\n".join(lines)
    print(summary)
    (out / "summary.txt").write_text(summary, encoding="utf-8")
