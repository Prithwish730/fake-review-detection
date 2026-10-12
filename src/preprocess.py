"""
preprocess.py

Text cleaning for review text. Produces a `clean_text` column used for
TF-IDF, similarity detection, and (later) model training.

IMPORTANT DESIGN RULE
---------------------
Style signals such as "!!!", ALL CAPS and punctuation are strong fake-review
clues. So we compute those style features from the RAW text (see features.py)
and only strip them here to create the normalized `clean_text`.

Lemmatization uses NLTK WordNet when available. If the WordNet data isn't
downloaded, we silently skip lemmatization so the pipeline never crashes.
To enable it once, run in Python:
    import nltk; nltk.download("wordnet"); nltk.download("omw-1.4")
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Keep negations: "not good" and "good" mean opposite things.
KEEP_WORDS = {"not", "no", "nor", "never", "n't"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - KEEP_WORDS

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
NON_ALPHA_RE = re.compile(r"[^a-z\s']")
MULTISPACE_RE = re.compile(r"\s+")


def _get_lemmatizer():
    """Return a lemmatize(word) function, or None if WordNet is unavailable."""
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("tests")  # raises LookupError if data is missing
        return lemmatizer.lemmatize
    except Exception:
        return None


_LEMMATIZE = _get_lemmatizer()


def clean_text(text, remove_stopwords=True, lemmatize=True) -> str:
    """Lowercase, strip URLs/HTML/punctuation/digits, drop stopwords, lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    tokens = MULTISPACE_RE.sub(" ", text).strip().split(" ")
    tokens = [t.strip("'") for t in tokens if t.strip("'")]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if lemmatize and _LEMMATIZE is not None:
        tokens = [_LEMMATIZE(t) for t in tokens]
    return " ".join(tokens)


def add_clean_text(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """Return a copy of df with a `clean_text` column added."""
    out = df.copy()
    out["clean_text"] = out[text_col].apply(clean_text)
    return out


def lemmatization_available() -> bool:
    return _LEMMATIZE is not None


if __name__ == "__main__":
    sample = "This product is AMAZING!!! Visit http://spam.com <b>now</b>, 5 stars!!!"
    print("RAW  :", sample)
    print("CLEAN:", clean_text(sample))
    print("Lemmatization active:", lemmatization_available())
