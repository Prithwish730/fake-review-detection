"""
train_classical.py

Week 3: train and compare classical ML models for fake review detection.

Models compared (the "baseline -> classical ML" half of your comparison table):
  1. Logistic Regression  + TF-IDF                 (text only)
  2. Naive Bayes          + TF-IDF                 (text only)
  3. Linear SVM           + TF-IDF                 (text only)
  4. Logistic Regression  + linguistic features    (style only)
  5. Random Forest        + linguistic features    (style only)
  6. XGBoost              + linguistic features    (style only; falls back to
                                                    sklearn HistGradientBoosting
                                                    if xgboost isn't installed)
  7. Logistic Regression  + TF-IDF + linguistic    (combined)

METHODOLOGY (explain these in your viva)
  * Split is grouped by reviewer: all reviews of one reviewer stay on the same
    side of the split, so the model can't "memorize" a reviewer's template.
  * Split is stratified: train and test keep the same fake/genuine ratio.
  * TF-IDF is fitted INSIDE each pipeline, i.e. on training data only (no leakage).
  * The best model is chosen by cross-validated F1 on the TRAIN set, never by
    peeking at the test set.
  * Fake is the positive class (label 1). With imbalanced data, precision,
    recall, F1 and PR-AUC are more honest than accuracy.

Run:
    python src/train_classical.py                 # synthetic sample
    python src/train_classical.py --source ott    # real labeled data
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from build_features import LINGUISTIC_COLS

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
RANDOM_STATE = 42

FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Optional MLflow tracking (never crashes the script if unavailable)
# ----------------------------------------------------------------------
_mlflow_warned = False


def log_to_mlflow(run_name: str, params: dict, metrics: dict):
    global _mlflow_warned
    try:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
        mlflow.set_experiment("fake-review-classifier")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
    except ImportError:
        if not _mlflow_warned:
            print("  (mlflow not installed - skipping experiment tracking)")
            _mlflow_warned = True
    except Exception as e:  # tracking must never break training
        if not _mlflow_warned:
            print(f"  (mlflow logging skipped: {e})")
            _mlflow_warned = True


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_data(source="sample"):
    processed = PROCESSED_DIR / f"{source}_review_features.csv"
    if not processed.exists():
        raise FileNotFoundError(
            f"{processed} not found. Run "
            f"`python src/build_features.py --source {source}` first."
        )
    df = pd.read_csv(processed)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["clean_text"] = df["clean_text"].fillna("")
    y = (df["label"] == "fake").astype(int).to_numpy()  # fake = positive class
    ling_cols = [c for c in LINGUISTIC_COLS if c in df.columns and df[c].notna().all()]
    return df, y, ling_cols


def grouped_stratified_split(df, y):
    """80/20 split; reviewers never appear in both sets."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(df, y, groups=df["reviewer_id"]))
    return train_idx, test_idx


# ----------------------------------------------------------------------
# Model definitions
# ----------------------------------------------------------------------
def tfidf_block():
    return ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000,
                                     sublinear_tf=True), "clean_text")


def ling_block(ling_cols):
    return ("ling", StandardScaler(), ling_cols)


def build_models(ling_cols, pos_weight):
    """Return {name: (feature_set, pipeline)}."""
    try:
        from xgboost import XGBClassifier
        boosted_name = "XGBoost"
        boosted = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                                subsample=0.9, colsample_bytree=0.9,
                                scale_pos_weight=pos_weight, eval_metric="logloss",
                                random_state=RANDOM_STATE, n_jobs=1)
    except ImportError:
        boosted_name = "HistGradientBoosting (XGBoost fallback)"
        boosted = HistGradientBoostingClassifier(class_weight="balanced",
                                                 random_state=RANDOM_STATE)

    text_only = lambda: ColumnTransformer([tfidf_block()])
    ling_only = lambda: ColumnTransformer([ling_block(ling_cols)])
    combined = lambda: ColumnTransformer([tfidf_block(), ling_block(ling_cols)])

    return {
        "Logistic Regression": ("TF-IDF", Pipeline([
            ("features", text_only()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])),
        "Naive Bayes": ("TF-IDF", Pipeline([
            ("features", text_only()),
            ("clf", MultinomialNB()),
        ])),
        "Linear SVM": ("TF-IDF", Pipeline([
            ("features", text_only()),
            ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
        ])),
        "Logistic Regression ": ("Linguistic", Pipeline([
            ("features", ling_only()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])),
        "Random Forest": ("Linguistic", Pipeline([
            ("features", ling_only()),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                           random_state=RANDOM_STATE, n_jobs=1)),
        ])),
        boosted_name: ("Linguistic", Pipeline([
            ("features", ling_only()),
            ("clf", boosted),
        ])),
        "Logistic Regression  ": ("TF-IDF + Linguistic", Pipeline([
            ("features", combined()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])),
    }


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def get_scores(model, X):
    """Continuous score for the positive class (needed for ROC / PR curves)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def plot_curves(curves, y_test, tag):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, score in curves.items():
        fpr, tpr, _ = roc_curve(y_test, score)
        ax.plot(fpr, tpr, label=name.strip())
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"07_roc_curves_{tag}.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, score in curves.items():
        p, r, _ = precision_recall_curve(y_test, score)
        ax.plot(r, p, label=name.strip())
    ax.axhline(y_test.mean(), color="k", ls="--", alpha=0.4, label="no-skill baseline")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"08_pr_curves_{tag}.png", dpi=150)
    plt.close()


def plot_confusion(model, X_test, y_test, title, tag):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["genuine", "fake"],
        cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"09_confusion_matrix_best_{tag}.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sample", "ott", "amazon"], default="sample")
    source = ap.parse_args().source

    df, y, ling_cols = load_data(source)
    train_idx, test_idx = grouped_stratified_split(df, y)
    X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = df["reviewer_id"].iloc[train_idx]

    print(f"Train: {len(X_train)} reviews | Test: {len(X_test)} reviews")
    print(f"Fake rate  train: {y_train.mean():.1%} | test: {y_test.mean():.1%}")
    print(f"Reviewer overlap train/test: "
          f"{len(set(X_train['reviewer_id']) & set(X_test['reviewer_id']))} (should be 0)")
    print(f"Linguistic features used ({len(ling_cols)}): {ling_cols}\n")

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models = build_models(ling_cols, pos_weight)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows, fitted, curves = [], {}, {}
    for name, (feature_set, pipe) in models.items():
        t0 = time.time()
        cv_res = cross_validate(pipe, X_train, y_train, groups=groups_train,
                                cv=cv, scoring="f1", n_jobs=1)
        pipe.fit(X_train, y_train)
        scores = get_scores(pipe, X_test)
        preds = pipe.predict(X_test)
        m = compute_metrics(y_test, preds, scores)
        elapsed = time.time() - t0

        row = {"model": name.strip(), "features": feature_set,
               "cv_f1_mean": cv_res["test_score"].mean(),
               "cv_f1_std": cv_res["test_score"].std(), **m,
               "seconds": round(elapsed, 1)}
        rows.append(row)
        fitted[name], curves[name] = pipe, scores
        log_to_mlflow(f"{name.strip()} [{feature_set}]",
                      {"model": name.strip(), "features": feature_set,
                       "n_train": len(X_train)},
                      {**m, "cv_f1_mean": row["cv_f1_mean"]})
        print(f"  {name.strip():<42} [{feature_set:<20}] "
              f"CV-F1 {row['cv_f1_mean']:.3f}  test-F1 {m['f1']:.3f}  "
              f"PR-AUC {m['pr_auc']:.3f}")

    results = pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)
    results.to_csv(REPORT_DIR / f"model_comparison_{source}.csv", index=False)

    # best model chosen by CV score on TRAIN only
    best_row = results.iloc[0]
    best_key = next(k for k in fitted if k.strip() == best_row["model"]
                    and models[k][0] == best_row["features"])
    best_model = fitted[best_key]
    joblib.dump(best_model, MODEL_DIR / f"best_classifier_{source}.joblib")
    # Also keep the plain TF-IDF Logistic Regression: it is linear (every word's contribution
    # can be explained exactly) and gives usable probabilities -> used by explain.py + dashboard
    joblib.dump(fitted["Logistic Regression"], MODEL_DIR / f"explainable_lr_{source}.joblib")
    (MODEL_DIR / f"best_classifier_{source}_meta.json").write_text(json.dumps({
        "model": best_row["model"], "features": best_row["features"],
        "linguistic_columns": ling_cols,
        "test_metrics": {k: float(best_row[k]) for k in
                         ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]},
    }, indent=2))

    plot_curves(curves, y_test, source)
    plot_confusion(best_model, X_test, y_test,
                   f"{best_row['model']} ({best_row['features']})", source)

    show = ["model", "features", "cv_f1_mean", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    print("\n" + "=" * 78)
    print(f"MODEL COMPARISON - {source} data (sorted by cross-validated F1 on train set)")
    print("=" * 78)
    print(results[show].round(3).to_string(index=False))
    print(f"\nBest model: {best_row['model']} [{best_row['features']}]  -> models/best_classifier_{source}.joblib")
    print(f"Saved: reports/model_comparison_{source}.csv + figures 07/08/09 (_{source})")


if __name__ == "__main__":
    main()
