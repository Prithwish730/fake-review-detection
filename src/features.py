"""
features.py

Feature engineering in three groups:

  A. LINGUISTIC features   -> computed from the review text alone.
                              Used by the fake-review CLASSIFIER.
  B. REVIEW-LEVEL behavior -> rating vs. product average, etc.
  C. REVIEWER-LEVEL behavior -> one row per reviewer (burstiness, template
                              reuse, verified ratio...). Used by the TRUST SCORE.

Why keep A separate from C?
The Ott et al. corpus has text + label but no reviewer info, while Amazon/Yelp
data has reviewer info but no labels. Keeping the groups separate lets each
dataset feed the right module, and avoids "leakage" (a classifier that wins by
reading reviewer history instead of learning from the text).
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# Small word lists used for style features
# ----------------------------------------------------------------------
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "our", "ours", "us"}
INTENSIFIERS = {
    "amazing", "best", "perfect", "love", "loved", "awesome", "excellent",
    "fantastic", "incredible", "greatest", "wonderful", "outstanding",
    "highly", "definitely", "absolutely", "ever", "everyone", "life",
}
POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "love", "loved", "perfect", "best",
    "solid", "decent", "happy", "recommend", "recommended", "awesome", "nice",
    "fantastic", "wonderful", "works", "comfortable", "fast",
}
NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "awful", "worst", "disappointing", "disappointed",
    "broke", "broken", "cheap", "buggy", "slow", "heavy", "annoyance", "unclear",
    "wearing", "drains", "waste", "return", "refund", "hate", "problem",
}

WORD_RE = re.compile(r"[A-Za-z']+")
SENTENCE_RE = re.compile(r"[.!?]+")


# ----------------------------------------------------------------------
# Sentiment: use VADER if available, otherwise a tiny lexicon fallback
# ----------------------------------------------------------------------
def _get_vader():
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()  # raises LookupError if lexicon missing
        return sia
    except Exception:
        return None


_VADER = _get_vader()


def sentiment_score(text: str) -> float:
    """Return sentiment in [-1, 1]. Uses VADER compound score when installed."""
    if _VADER is not None:
        return _VADER.polarity_scores(text)["compound"]
    words = [w.lower() for w in WORD_RE.findall(text)]
    if not words:
        return 0.0
    pos = sum(w in POSITIVE_WORDS for w in words)
    neg = sum(w in NEGATIVE_WORDS for w in words)
    return float(np.clip((pos - neg) / np.sqrt(len(words)), -1, 1))


def sentiment_backend() -> str:
    return "VADER" if _VADER is not None else "fallback-lexicon"


# ----------------------------------------------------------------------
# A. Linguistic features
# ----------------------------------------------------------------------
def _count_syllables(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    return max(1, len(groups))


def flesch_reading_ease(text: str) -> float:
    words = WORD_RE.findall(text)
    sentences = [s for s in SENTENCE_RE.split(text) if s.strip()]
    if not words or not sentences:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))


def _single_review_linguistic(text: str) -> dict:
    text = text if isinstance(text, str) else ""
    words = WORD_RE.findall(text)
    n_words = len(words)
    n_chars = len(text)
    letters = [c for c in text if c.isalpha()]
    lower_words = [w.lower() for w in words]

    return {
        "char_count": n_chars,
        "word_count": n_words,
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0.0,
        "exclamation_count": text.count("!"),
        "exclamation_ratio": text.count("!") / n_chars if n_chars else 0.0,
        "question_count": text.count("?"),
        "caps_ratio": sum(c.isupper() for c in letters) / len(letters) if letters else 0.0,
        "all_caps_word_ratio": (
            sum(w.isupper() and len(w) > 1 for w in words) / n_words if n_words else 0.0
        ),
        "punctuation_ratio": (
            sum(c in "!?.,;:-" for c in text) / n_chars if n_chars else 0.0
        ),
        "digit_ratio": sum(c.isdigit() for c in text) / n_chars if n_chars else 0.0,
        "unique_word_ratio": len(set(lower_words)) / n_words if n_words else 0.0,
        "first_person_ratio": (
            sum(w in FIRST_PERSON for w in lower_words) / n_words if n_words else 0.0
        ),
        "intensifier_ratio": (
            sum(w in INTENSIFIERS for w in lower_words) / n_words if n_words else 0.0
        ),
        "flesch_reading_ease": flesch_reading_ease(text),
        "sentiment_score": sentiment_score(text),
    }


def add_linguistic_features(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """Append linguistic feature columns (computed on RAW text) to df."""
    feats = pd.DataFrame([_single_review_linguistic(t) for t in df[text_col]], index=df.index)
    out = pd.concat([df, feats], axis=1)

    # Sentiment-vs-rating mismatch: a 5-star review with negative text (or vice versa)
    if "rating" in out.columns and out["rating"].notna().any():
        expected = (out["rating"].astype(float) - 3.0) / 2.0  # 1->-1, 3->0, 5->+1
        out["rating_sentiment_mismatch"] = (out["sentiment_score"] - expected).abs()
    return out


# ----------------------------------------------------------------------
# B. Review-level behavioral features
# ----------------------------------------------------------------------
def add_review_level_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """Rating deviation from the product's average (excluding the review itself)."""
    out = df.copy()
    if {"product_rating_sum", "product_rating_count"} <= set(out.columns):
        # stats computed over ALL reviews of the product (see data_loader)
        sums, counts = out["product_rating_sum"], out["product_rating_count"]
    else:
        grp = out.groupby("product_id")["rating"]
        sums, counts = grp.transform("sum"), grp.transform("count")
    # leave-one-out mean: so a review never influences its own benchmark
    loo_mean = (sums - out["rating"]) / (counts - 1).replace(0, np.nan)
    out["product_avg_rating_loo"] = loo_mean
    out["rating_deviation"] = (out["rating"] - loo_mean).fillna(0.0)
    out["abs_rating_deviation"] = out["rating_deviation"].abs()
    return out


def add_self_similarity(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    For every review, the max cosine similarity to ANOTHER review by the SAME
    reviewer (TF-IDF). Near 1.0 means the reviewer is re-posting templated text.
    No labels are used here, so there is no leakage.
    """
    out = df.reset_index(drop=True).copy()
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(out[text_col].fillna(""))
    has_product = "product_id" in out.columns
    products = out["product_id"].to_numpy() if has_product else None
    sims = np.zeros(len(out))
    cross = np.zeros(len(out))
    for _, idx in out.groupby("reviewer_id").indices.items():
        if len(idx) < 2:
            continue
        S = cosine_similarity(tfidf[idx])
        np.fill_diagonal(S, 0.0)
        sims[idx] = S.max(axis=1)
        if has_product:
            # similarity to this reviewer's OTHER reviews on DIFFERENT products only.
            # Same text on the same product is usually a variant/double-post artifact;
            # the same text pasted across different products is the real spam signal.
            p = products[idx]
            cross[idx] = np.where(p[:, None] == p[None, :], 0.0, S).max(axis=1)
    out["max_self_similarity"] = sims
    if has_product:
        out["max_cross_product_similarity"] = cross
    return out


# ----------------------------------------------------------------------
# C. Reviewer-level features (one row per reviewer) -> trust score input
# ----------------------------------------------------------------------
def build_reviewer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df to already have: abs_rating_deviation, max_self_similarity.
    Returns one row per reviewer.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["is_5star"] = (d["rating"] == 5).astype(int)
    # may be NA in real data -> NaN, and means below simply skip it
    d["verified_int"] = d["verified_purchase"].astype("boolean").astype("float64")

    def per_reviewer(g: pd.DataFrame) -> pd.Series:
        dates = g["date"].sort_values()
        gaps = dates.diff().dt.days.dropna()
        n = len(g)
        age = g["account_age_days"].iloc[0] if "account_age_days" in g else np.nan
        span = (dates.max() - dates.min()).days if n > 1 else 0
        # real datasets rarely give account age; fall back to the span we can observe
        denom = age if pd.notna(age) else span
        reviews_per_day = g.groupby(g["date"].dt.date).size()
        stats = {
            "review_count": n,
            "avg_rating": g["rating"].mean(),
            "rating_std": g["rating"].std(ddof=0),
            "pct_5star": g["is_5star"].mean(),
            "verified_ratio": g["verified_int"].mean(),
            "avg_abs_rating_deviation": g["abs_rating_deviation"].mean(),
            "avg_self_similarity": g["max_self_similarity"].mean(),
            "max_reviews_in_one_day": reviews_per_day.max(),
            "avg_gap_days": gaps.mean() if len(gaps) else np.nan,
            "account_age_days": age,
            "activity_span_days": span,
            # reviews per 30 days of account life (or observed activity span if
            # account age is unknown): high values = suspicious velocity
            "review_velocity": (n / max(denom, 1)) * 30,
        }
        if "max_cross_product_similarity" in g.columns:
            stats["avg_cross_product_similarity"] = g["max_cross_product_similarity"].mean()
        return pd.Series(stats)

    feats = d.groupby("reviewer_id").apply(per_reviewer, include_groups=False).reset_index()
    return feats
