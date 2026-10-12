"""
explain.py

Explainability for the fake-review classifier.

For a LINEAR model (Logistic Regression, linear SVM, Naive Bayes) the prediction is

    score = intercept + sum over words of (weight_of_word * tfidf_value_of_word)

so every word's contribution is known EXACTLY. This is the same additive
decomposition that SHAP's linear explainer approximates, but exact and fast,
with no extra dependency. Positive contribution = pushes toward "fake".

Two uses:
  * global : which words does the model associate with fake vs genuine reviews?
  * local  : why was THIS review scored the way it was? (used by the dashboard)

Run:
    python src/explain.py --source ott
    python src/explain.py --source ott --text "Best hotel ever, absolutely amazing!"
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import _single_review_linguistic
from preprocess import clean_text

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"

STYLE_KEYS = ["word_count", "exclamation_count", "caps_ratio", "first_person_ratio",
              "intensifier_ratio", "sentiment_score", "flesch_reading_ease"]


def _parts(pipeline):
    return pipeline.named_steps["features"], pipeline.named_steps["clf"]


def _short(name: str) -> str:
    return name.split("__", 1)[-1]


def linear_weights(clf):
    """Return (weights, intercept) so that score = intercept + weights . x"""
    if hasattr(clf, "coef_"):
        return np.asarray(clf.coef_).ravel(), float(np.ravel(clf.intercept_)[0])
    if hasattr(clf, "feature_log_prob_"):          # MultinomialNB: log-odds is linear
        w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        b = clf.class_log_prior_[1] - clf.class_log_prior_[0]
        return np.asarray(w), float(b)
    raise ValueError("Per-feature explanations need a linear model "
                     "(Logistic Regression, linear SVM or Naive Bayes).")


def explain_row(pipeline, row: pd.DataFrame):
    """
    Exact contribution of every non-zero feature for ONE row.
    Returns (DataFrame[feature, value, weight, contribution], intercept).
    """
    ct, clf = _parts(pipeline)
    w, b = linear_weights(clf)
    X = ct.transform(row)
    x = X.toarray().ravel() if hasattr(X, "toarray") else np.asarray(X).ravel()
    names = ct.get_feature_names_out()
    idx = np.nonzero(x)[0]
    df = pd.DataFrame({
        "feature": [_short(names[i]) for i in idx],
        "value": x[idx],
        "weight": w[idx],
        "contribution": w[idx] * x[idx],
    })
    return df.sort_values("contribution", ascending=False).reset_index(drop=True), b


def top_words(pipeline, n=20):
    """Global view: words with the largest weights toward fake / genuine."""
    ct, clf = _parts(pipeline)
    w, _ = linear_weights(clf)
    raw = ct.get_feature_names_out()
    table = pd.DataFrame({"word": [_short(r) for r in raw], "weight": w,
                          "is_text": [r.startswith("tfidf__") for r in raw]})
    table = table[table["is_text"]].drop(columns="is_text")
    fake = table.nlargest(n, "weight").reset_index(drop=True)
    genuine = table.nsmallest(n, "weight").reset_index(drop=True)
    return fake, genuine


def analyze_review(text: str, pipeline, top_n=8) -> dict:
    """Score one raw review text and explain it. Used by the dashboard."""
    clean = clean_text(text)
    style = _single_review_linguistic(text)
    row = pd.DataFrame([{**style, "clean_text": clean}])

    contribs, intercept = explain_row(pipeline, row)
    decision = float(contribs["contribution"].sum() + intercept)
    if hasattr(pipeline, "predict_proba"):
        p_fake = float(pipeline.predict_proba(row)[0, 1])
    else:                                           # e.g. LinearSVC: uncalibrated
        p_fake = float(1.0 / (1.0 + np.exp(-decision)))

    ct, _ = _parts(pipeline)
    vocab = getattr(ct.named_transformers_.get("tfidf"), "vocabulary_", {}) or {}
    tokens = clean.split()
    coverage = (sum(t in vocab for t in tokens) / len(tokens)) if tokens else 0.0

    return {
        "p_fake": p_fake,
        "decision": decision,
        "vocab_coverage": coverage,       # share of words the model has ever seen
        "n_words": len(tokens),
        "top_fake": contribs[contribs["contribution"] > 0].head(top_n),
        "top_genuine": contribs[contribs["contribution"] < 0].sort_values("contribution").head(top_n),
        "style": {k: round(float(style[k]), 3) for k in STYLE_KEYS},
        "clean_text": clean,
    }


def plot_top_words(fake: pd.DataFrame, genuine: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    axes[0].barh(fake["word"][::-1], fake["weight"][::-1], color="#C44E52")
    axes[0].set_title("Words pushing toward FAKE")
    axes[1].barh(genuine["word"][::-1], genuine["weight"][::-1], color="#4C72B0")
    axes[1].set_title("Words pushing toward GENUINE")
    for ax in axes:
        ax.set_xlabel("model weight")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ott")
    ap.add_argument("--n", type=int, default=20, help="how many top words per side")
    ap.add_argument("--text", default=None, help="also explain this one review")
    args = ap.parse_args()

    path = MODEL_DIR / f"explainable_lr_{args.source}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/train_classical.py --source {args.source}` first.")
    pipe = joblib.load(path)

    fake, genuine = top_words(pipe, args.n)
    out_csv = REPORT_DIR / f"top_words_{args.source}.csv"
    pd.concat([fake.assign(side="fake"), genuine.assign(side="genuine")]).to_csv(out_csv, index=False)
    out_fig = FIG_DIR / f"12_top_words_{args.source}.png"
    plot_top_words(fake, genuine, out_fig)

    print(f"Top {args.n} words toward FAKE   : {', '.join(fake['word'].head(args.n))}")
    print(f"Top {args.n} words toward GENUINE: {', '.join(genuine['word'].head(args.n))}")
    print(f"\nSaved {out_csv.name} and figures/{out_fig.name}")

    if args.text:
        res = analyze_review(args.text, pipe)
        print(f"\nReview: {args.text}")
        print(f"Estimated chance fake: {res['p_fake']:.0%}  "
              f"(model has seen {res['vocab_coverage']:.0%} of its words)")
        print("Pushing toward fake   :", ", ".join(res["top_fake"]["feature"]) or "-")
        print("Pushing toward genuine:", ", ".join(res["top_genuine"]["feature"]) or "-")


if __name__ == "__main__":
    main()
