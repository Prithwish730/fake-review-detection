"""
Fake Review Detection & Reviewer Trust Score - dashboard

Run from the project root:
    streamlit run app/app.py

Needs (create them first, see README):
    models/explainable_lr_ott.joblib          <- python src/train_classical.py --source ott
    data/processed/amazon_trust_scores.csv    <- python src/trust_score.py
    data/processed/amazon_review_features.csv <- python src/build_features.py --source amazon ...
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st


def _ensure_nltk_data():
    """A fresh server has no NLTK data. Fetch it once (skipped if already present),
    BEFORE importing the text-processing modules, which look for it at import time."""
    try:
        import nltk
        needed = {"vader_lexicon": ["sentiment/vader_lexicon.zip", "sentiment/vader_lexicon"],
                  "wordnet": ["corpora/wordnet.zip", "corpora/wordnet"],
                  "omw-1.4": ["corpora/omw-1.4.zip", "corpora/omw-1.4"]}
        for resource, locations in needed.items():
            found = False
            for loc in locations:
                try:
                    nltk.data.find(loc)
                    found = True
                    break
                except LookupError:
                    pass
            if not found:
                nltk.download(resource, quiet=True)
    except Exception:
        pass  # the app still works without them (simpler fallbacks), just less accurately


_ensure_nltk_data()

from explain import analyze_review  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"

ASSETS = ROOT / "deploy_assets"   # small copies of the files above, committed for deployment


def first_existing(*paths: Path) -> Path:
    return next((p for p in paths if p.exists()), paths[0])


CLASSIFIER_PATH = first_existing(MODELS / "explainable_lr_ott.joblib",
                                 ASSETS / "explainable_lr_ott.joblib")
TRUST_PATH = first_existing(PROCESSED / "amazon_trust_scores.csv",
                            ASSETS / "amazon_trust_scores.csv")
REVIEWS_PATH = first_existing(PROCESSED / "amazon_review_features.csv",
                              ASSETS / "amazon_reviews_slim.csv.gz")
COMPARISON_PATH = first_existing(REPORTS / "model_comparison_ott.csv",
                                 ASSETS / "model_comparison_ott.csv")


def figure_path(name: str) -> Path:
    return first_existing(FIGS / name, ASSETS / "figures" / name)

COMPONENT_LABELS = {
    "s_template_reuse": "Same text across products",
    "s_burstiness": "Many reviews in one day",
    "s_rating_deviation": "Ratings far from product average",
    "s_five_star_share": "Almost always 5 stars",
    "s_velocity": "Reviews per month of activity",
    "s_unverified": "Unverified purchases",
}

EXAMPLES = {
    "(write your own)": "",
    "Example: hotel review, gushing style":
        "My husband and I stayed here for our anniversary and it was absolutely AMAZING! "
        "The staff were wonderful and the room was perfect. Best hotel experience ever, "
        "we will definitely be back!",
    "Example: hotel review, specific style":
        "Stayed two nights for a conference. The room was small but clean, the elevator was "
        "slow, and parking cost $45 a night. Breakfast was fine. A short walk to the train.",
}

st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️", layout="wide")


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource
def load_classifier():
    return joblib.load(CLASSIFIER_PATH)


@st.cache_data
def load_trust():
    return pd.read_csv(TRUST_PATH)


@st.cache_data
def load_reviews():
    wanted = {"reviewer_id", "product_id", "rating", "date", "verified_purchase", "review_text"}
    return pd.read_csv(REVIEWS_PATH, usecols=lambda c: c in wanted)


def need(path: Path, command: str) -> bool:
    """Show a friendly message and return False if a required file is missing."""
    if path.exists():
        return True
    st.error(f"Missing `{path.relative_to(ROOT)}`. Create it with:")
    st.code(command, language="bash")
    return False


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🕵️ Fake Review Detection & Reviewer Trust Score")
st.caption("Text classifier trained on the Ott et al. hotel-review corpus, plus a behavioral "
           "trust score computed on Amazon (All_Beauty) reviewers.")

tab_check, tab_reviewer, tab_flagged, tab_results = st.tabs(
    ["Check a review", "Reviewer trust lookup", "Flagged reviewers", "Model results"])


# ----------------------------------------------------------------------
# Tab 1: check a single review
# ----------------------------------------------------------------------
with tab_check:
    st.subheader("Does this review look fake?")
    if need(CLASSIFIER_PATH, "python src/train_classical.py --source ott"):
        pipe = load_classifier()
        choice = st.selectbox("Try an example", list(EXAMPLES))
        text = st.text_area("Review text", value=EXAMPLES[choice], height=150,
                            key=f"review_text_{choice}")

        if st.button("Analyze review") and text.strip():
            res = analyze_review(text, pipe)
            p = res["p_fake"]

            left, right = st.columns(2)
            with left:
                st.metric("Model's estimated chance this is fake", f"{p:.0%}")
                st.progress(float(min(max(p, 0.0), 1.0)))
                if 0.35 <= p <= 0.65:
                    st.info("Uncertain: the model is close to 50/50 on this one.")
                elif p > 0.65:
                    st.error("Looks fake-like.")
                else:
                    st.success("Looks genuine-like.")
                if res["vocab_coverage"] < 0.5:
                    st.warning(
                        f"Only {res['vocab_coverage']:.0%} of this review's words appear in the "
                        "model's vocabulary. The model learned from hotel reviews, so treat "
                        "this prediction as unreliable.")
                with st.expander("Writing-style signals"):
                    st.table(pd.DataFrame({"value": res["style"]}))
            with right:
                st.markdown("**Words pushing toward FAKE**")
                if len(res["top_fake"]):
                    st.bar_chart(res["top_fake"].set_index("feature")["contribution"])
                else:
                    st.caption("None.")
                st.markdown("**Words pushing toward GENUINE**")
                if len(res["top_genuine"]):
                    st.bar_chart(res["top_genuine"].set_index("feature")["contribution"])
                else:
                    st.caption("None.")
            st.caption("Bars show each word's exact contribution to the model's score "
                       "(weight x TF-IDF value). The probability is a model estimate, not a "
                       "verdict.")


# ----------------------------------------------------------------------
# Tab 2: reviewer trust lookup
# ----------------------------------------------------------------------
with tab_reviewer:
    st.subheader("How trustworthy does this reviewer's behavior look?")
    cmd = ("python src/build_features.py --source amazon --path data/raw/All_Beauty.jsonl "
           "--max-reviewers 10000\npython src/trust_score.py")
    if need(TRUST_PATH, cmd) and need(REVIEWS_PATH, cmd):
        trust = load_trust()
        reviews = load_reviews()

        lowest = trust.sort_values("trust_score")["reviewer_id"].head(50).tolist()
        picked = st.selectbox("Pick one of the 50 least-trusted reviewers", ["(none)"] + lowest)
        typed = st.text_input("...or paste a reviewer ID")
        rid = typed.strip() or (picked if picked != "(none)" else "")

        if rid:
            row = trust[trust["reviewer_id"] == rid]
            if row.empty:
                st.warning("That reviewer ID is not in the scored set. Only reviewers with at "
                           "least 3 reviews are scored.")
            else:
                r = row.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Trust score (0-100)", f"{r['trust_score']:.1f}")
                c2.metric("Heuristic score", f"{r['heuristic_trust']:.1f}")
                c3.metric("Isolation Forest score", f"{r['learned_trust']:.1f}")
                st.progress(float(min(max(r["trust_score"] / 100.0, 0.0), 1.0)))
                if bool(r.get("flagged", False)):
                    st.error("Flagged: in the least-trusted group of reviewers.")
                st.caption("Scores are relative to the other reviewers in this dataset. A low "
                           "score means unusual, suspicious-looking behavior, not proven fraud.")

                comps = {COMPONENT_LABELS[c]: float(r[c])
                         for c in COMPONENT_LABELS if c in trust.columns and pd.notna(r[c])}
                st.markdown("**What drove the heuristic score** (0 = typical, 1 = most suspicious "
                            "among all reviewers)")
                st.bar_chart(pd.Series(comps))

                st.markdown("**Their reviews**")
                if "review_text" not in reviews.columns:
                    st.caption("Review text is not included in this deployment.")
                mine = reviews[reviews["reviewer_id"] == rid].sort_values("date")
                st.dataframe(mine.drop(columns="reviewer_id").reset_index(drop=True))


# ----------------------------------------------------------------------
# Tab 3: flagged reviewers
# ----------------------------------------------------------------------
with tab_flagged:
    st.subheader("Least-trusted reviewers")
    if need(TRUST_PATH, "python src/trust_score.py"):
        trust = load_trust()
        a, b = st.columns(2)
        a.metric("Reviewers scored", f"{len(trust):,}")
        b.metric("Flagged (least trusted group)", f"{int(trust['flagged'].sum()):,}"
                 if "flagged" in trust.columns else "n/a")

        counts, edges = np.histogram(trust["trust_score"], bins=20, range=(0, 100))
        st.markdown("**Distribution of trust scores**")
        st.bar_chart(pd.Series(counts, index=[f"{int(e)}" for e in edges[:-1]]))

        threshold = st.slider("Show reviewers with a trust score below", 0, 100,
                              int(trust["trust_score"].quantile(0.10)))
        cols = [c for c in ["reviewer_id", "trust_score", "heuristic_trust", "learned_trust",
                            "review_count", "pct_5star", "max_reviews_in_one_day",
                            "avg_cross_product_similarity", "verified_ratio"]
                if c in trust.columns]
        subset = trust[trust["trust_score"] < threshold].sort_values("trust_score")[cols]
        st.write(f"{len(subset):,} reviewers below {threshold}")
        st.dataframe(subset.head(500).reset_index(drop=True))
        st.download_button("Download this list as CSV", subset.to_csv(index=False).encode("utf-8"),
                           file_name="flagged_reviewers.csv", mime="text/csv")


# ----------------------------------------------------------------------
# Tab 4: model results
# ----------------------------------------------------------------------
with tab_results:
    st.subheader("How well does the classifier work?")
    if need(COMPARISON_PATH, "python src/train_classical.py --source ott"):
        comp = pd.read_csv(COMPARISON_PATH)
        keep = [c for c in ["model", "features", "cv_f1_mean", "precision", "recall", "f1",
                            "roc_auc", "pr_auc"] if c in comp.columns]
        st.dataframe(comp[keep].round(3))
        st.caption("Fake is the positive class. Test set = 320 reviews, so differences of about "
                   "0.01 between models are within noise.")

    pairs = [("ROC curves", "07_roc_curves_ott.png"),
             ("Precision-recall curves", "08_pr_curves_ott.png"),
             ("Confusion matrix (best model)", "09_confusion_matrix_best_ott.png"),
             ("Words the model associates with fake vs genuine", "12_top_words_ott.png"),
             ("Trust score distribution", "10_trust_score_distribution.png"),
             ("Heuristic vs Isolation Forest agreement", "11_trust_score_agreement.png")]
    for i in range(0, len(pairs), 2):
        cols = st.columns(2)
        for col, (title, fname) in zip(cols, pairs[i:i + 2]):
            with col:
                st.markdown(f"**{title}**")
                if figure_path(fname).exists():
                    st.image(str(figure_path(fname)))
                else:
                    st.caption(f"(figure `{fname}` not generated yet)")

    with st.expander("Limitations"):
        st.markdown(
            "- The classifier was trained on **1,600 hotel reviews** written by crowd workers "
            "(fake) and review-site users (genuine). It may not transfer to other products.\n"
            "- Trust scores are **relative** rankings, not probabilities of fraud.\n"
            "- High-volume reviewers with unverified, mostly 5-star reviews are flagged even though "
            "some may be legitimate (for example reviewers who receive free products).\n"
            "- A different product ID does not always mean a different product: sellers list "
            "the same item under several IDs.")
