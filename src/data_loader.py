"""
data_loader.py

One loader per dataset. Every loader returns the SAME standardized schema, so
the rest of the pipeline never needs to know where the data came from:

    review_id          str     unique review identifier
    reviewer_id        str     unique reviewer identifier
    product_id         str     product / hotel identifier
    rating             float   star rating (NaN if the dataset has none)
    review_text        str     raw review text
    date               datetime (NaT if the dataset has none)
    verified_purchase  boolean (<NA> if the dataset has none)
    label              str     'genuine' / 'fake'  (None if unlabeled)
    account_age_days   float   (NaN if unavailable)

WHICH DATASET FEEDS WHICH MODULE
    source="ott"     labeled text, NO reviewer info  -> trains the CLASSIFIER
    source="amazon"  reviewer info, NO labels        -> feeds the TRUST SCORE
    source="sample"  synthetic data with everything  -> for testing the pipeline

Usage:
    from data_loader import load_reviews
    df = load_reviews("ott",    path="data/raw/deceptive-opinion.csv")
    df = load_reviews("amazon", path="data/raw/All_Beauty.jsonl", min_reviews=3)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

REQUIRED_COLUMNS = [
    "review_id", "reviewer_id", "product_id", "rating",
    "review_text", "date", "verified_purchase",
]


# ======================================================================
# 1. Synthetic sample data (for testing the pipeline)
# ======================================================================
def load_sample_data(path=None) -> pd.DataFrame:
    path = Path(path) if path else RAW_DIR / "reviews_sample.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/generate_sample_data.py` first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return _validate(df)


# ======================================================================
# 2. Ott et al. Deceptive Opinion Spam Corpus  (labeled, text only)
# ======================================================================
def load_ott_corpus(path=None) -> pd.DataFrame:
    """
    Accepts EITHER of the two formats the corpus is distributed in:
      (a) the Kaggle CSV  'deceptive-opinion.csv'
          columns: deceptive, hotel, polarity, source, text
      (b) the original folder 'op_spam_v1.4/' full of .txt files
    `path` can be the CSV file, the folder, or None (auto-detect in data/raw/).
    """
    path = _resolve_ott_path(path)
    if path.is_file():
        return _load_ott_csv(path)
    csvs = sorted(path.rglob("*.csv"))
    if csvs:
        return _load_ott_csv(csvs[0])
    return _load_ott_txt_folder(path)


def _resolve_ott_path(path) -> Path:
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Ott corpus path not found: {p}")
        return p
    candidates = (list(RAW_DIR.glob("*deceptive*opinion*.csv"))
                  + list(RAW_DIR.glob("*op_spam*"))
                  + list(RAW_DIR.glob("*ott*")))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find the Ott corpus in {RAW_DIR}.\n"
            "Put 'deceptive-opinion.csv' (Kaggle) or the 'op_spam_v1.4' folder there,\n"
            "or pass --path explicitly."
        )
    return candidates[0]


def _load_ott_csv(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    raw.columns = [c.strip().lower() for c in raw.columns]

    label_col = next((c for c in ("deceptive", "label", "class") if c in raw.columns), None)
    text_col = next((c for c in ("text", "review", "review_text") if c in raw.columns), None)
    if label_col is None or text_col is None:
        raise ValueError(
            f"Unexpected columns in {csv_path.name}: {list(raw.columns)}. "
            "Need a label column (deceptive/label) and a text column (text/review)."
        )
    label = raw[label_col].astype(str).str.strip().str.lower().map(
        {"deceptive": "fake", "truthful": "genuine", "fake": "fake", "genuine": "genuine",
         "1": "fake", "0": "genuine"})
    if label.isna().any():
        bad = raw.loc[label.isna(), label_col].unique()[:5]
        raise ValueError(f"Unrecognized label values: {bad}")

    n = len(raw)
    df = pd.DataFrame({
        "review_id": [f"OTT{i:05d}" for i in range(n)],
        "reviewer_id": [f"ott_reviewer_{i}" for i in range(n)],  # not in dataset
        "product_id": raw["hotel"] if "hotel" in raw.columns else "unknown",
        "rating": np.nan,
        "review_text": raw[text_col].astype(str),
        "date": pd.NaT,
        "verified_purchase": pd.array([pd.NA] * n, dtype="boolean"),
        "label": label.to_numpy(),
        "account_age_days": np.nan,
        "polarity": raw["polarity"] if "polarity" in raw.columns else None,
        "source": raw["source"] if "source" in raw.columns else None,
    })
    return _validate(df)


def _load_ott_txt_folder(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(root.rglob("*.txt")):
        rel = [p.lower() for p in f.relative_to(root).parts[:-1]]  # folders only, not root
        rel_str = "/".join(rel)
        if "__macosx" in rel_str:
            continue
        if "deceptive" in rel_str:
            label = "fake"
        elif "truthful" in rel_str:
            label = "genuine"
        else:
            continue  # README etc.
        polarity = ("positive" if "positive" in rel_str
                    else "negative" if "negative" in rel_str else None)
        source = next((p.split("from_")[-1] for p in rel if "from_" in p), None)
        m = re.match(r"^[dt]_(.+?)_\d+$", f.stem)
        rows.append({
            "review_text": f.read_text(encoding="utf-8", errors="ignore").strip(),
            "label": label, "polarity": polarity, "source": source,
            "product_id": m.group(1) if m else "unknown",
        })
    if not rows:
        raise FileNotFoundError(f"No Ott .txt review files found under {root}")
    df = pd.DataFrame(rows)
    n = len(df)
    df["review_id"] = [f"OTT{i:05d}" for i in range(n)]
    df["reviewer_id"] = [f"ott_reviewer_{i}" for i in range(n)]
    df["rating"] = np.nan
    df["date"] = pd.NaT
    df["verified_purchase"] = pd.array([pd.NA] * n, dtype="boolean")
    df["account_age_days"] = np.nan
    return _validate(df)


# ======================================================================
# 3. Amazon / Yelp style reviews  (reviewer metadata, no labels)
# ======================================================================
# Different Amazon releases name columns differently. We try these in order.
COLUMN_ALIASES = {
    "reviewer_id": ["user_id", "reviewerID", "UserId", "userId", "reviewer_id", "customer_id"],
    "product_id": ["parent_asin", "asin", "ProductId", "product_id", "productId"],
    "rating": ["rating", "overall", "Score", "stars", "star_rating"],
    "review_text": ["text", "reviewText", "Text", "review_text", "review_body", "content"],
    "date": ["timestamp", "unixReviewTime", "Time", "reviewTime", "date", "review_date"],
    "verified_purchase": ["verified_purchase", "verified", "Verified"],
}


def load_amazon_reviews(path=None, nrows=1_000_000, min_reviews=3,
                        max_reviewers=5000, random_state=42,
                        drop_duplicates=True) -> pd.DataFrame:
    """
    Load an Amazon-style review file (.csv, .jsonl, .json, optionally .gz).

    nrows          read only the first N rows (these files can be many GB)
    min_reviews    keep reviewers with at least this many reviews. IMPORTANT:
                   most reviewers post once, and behavioral features (burstiness,
                   template reuse...) are meaningless for them.
    max_reviewers  randomly keep at most this many reviewers (keeps all their reviews)
    drop_duplicates  drop exact repeats (same reviewer + product + text). On Amazon these
                   are usually variant/double-submission artifacts, not manipulation, and
                   they would otherwise max out several suspicion signals at once.
    """
    path = _resolve_amazon_path(path)
    name = path.name.lower()
    if ".jsonl" in name or ".json" in name:
        raw = pd.read_json(path, lines=True, nrows=nrows)
    else:
        raw = pd.read_csv(path, nrows=nrows)
    print(f"  read {len(raw):,} rows from {path.name}")

    picked = {}
    for std, options in COLUMN_ALIASES.items():
        col = next((c for c in options if c in raw.columns), None)
        if col is not None:
            picked[std] = col
    missing = [c for c in ("reviewer_id", "product_id", "rating", "review_text")
               if c not in picked]
    if missing:
        raise ValueError(
            f"Could not find columns for {missing} in {path.name}.\n"
            f"File columns: {list(raw.columns)}\n"
            "Add the right names to COLUMN_ALIASES in data_loader.py."
        )
    print(f"  column mapping: {picked}")

    df = pd.DataFrame({std: raw[col] for std, col in picked.items()})
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["review_text"] = df["review_text"].astype(str)
    df["date"] = _parse_dates(df["date"]) if "date" in df else pd.NaT
    df["verified_purchase"] = (_to_nullable_bool(df["verified_purchase"])
                               if "verified_purchase" in df
                               else pd.array([pd.NA] * len(df), dtype="boolean"))
    df = df.dropna(subset=["reviewer_id", "product_id", "rating"])
    df = df[df["review_text"].str.strip().str.len() > 0]

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["reviewer_id", "product_id", "review_text"], keep="first")
        print(f"  dropped {before - len(df):,} exact duplicate reviews "
              "(same reviewer, product and text)")

    # Product averages must come from ALL reviews of a product, not only the
    # reviewers we keep below -- otherwise most products would have 1-2 reviews
    # and "rating deviation from the product average" would be mostly noise.
    by_product = df.groupby("product_id")["rating"]
    df["product_rating_sum"] = by_product.transform("sum")
    df["product_rating_count"] = by_product.transform("count")

    counts = df["reviewer_id"].value_counts()
    keep = counts[counts >= min_reviews].index
    print(f"  reviewers: {len(counts):,} total -> {len(keep):,} with >= {min_reviews} reviews")
    if len(keep) == 0:
        raise ValueError(
            f"No reviewer has >= {min_reviews} reviews in the rows read. "
            "Lower --min-reviews or raise --nrows."
        )
    if len(keep) > max_reviewers:
        keep = pd.Series(keep).sample(max_reviewers, random_state=random_state)
        print(f"  sampled down to {max_reviewers:,} reviewers")
    df = df[df["reviewer_id"].isin(set(keep))].reset_index(drop=True)

    df["review_id"] = [f"AMZ{i:08d}" for i in range(len(df))]
    df["label"] = None            # unlabeled -> used for the trust score module
    df["account_age_days"] = np.nan
    return _validate(df)


def _resolve_amazon_path(path) -> Path:
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Amazon reviews file not found: {p}")
        return p
    candidates = []
    for pat in ("*.jsonl", "*.jsonl.gz", "*.json", "*amazon*.csv", "*review*.csv"):
        candidates += [p for p in RAW_DIR.glob(pat) if "sample" not in p.name.lower()
                       and "deceptive" not in p.name.lower()]
    if not candidates:
        raise FileNotFoundError(
            f"No Amazon review file found in {RAW_DIR}. Put the .jsonl/.csv there "
            "or pass --path explicitly."
        )
    return candidates[0]


def _parse_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        unit = "ms" if s.dropna().median() > 1e11 else "s"   # 2023 data is in ms
        return pd.to_datetime(s, unit=unit, errors="coerce")
    return pd.to_datetime(s, errors="coerce", format="mixed")


def _to_nullable_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")
    mapping = {"true": True, "false": False, "y": True, "n": False,
               "yes": True, "no": False, "1": True, "0": False}
    return s.astype(str).str.strip().str.lower().map(mapping).astype("boolean")


# ======================================================================
# Shared
# ======================================================================
def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    for col in ("label", "account_age_days"):
        if col not in df.columns:
            df[col] = None
    return df


def load_reviews(source: str = "sample", path=None, **kwargs) -> pd.DataFrame:
    """Single entry point used by every script."""
    if source == "sample":
        return load_sample_data(path)
    if source == "ott":
        return load_ott_corpus(path)
    if source == "amazon":
        return load_amazon_reviews(path, **kwargs)
    raise ValueError(f"Unknown source '{source}'. Use: sample, ott, amazon")


if __name__ == "__main__":
    df = load_reviews("sample")
    print(f"Loaded {len(df)} reviews, {df['reviewer_id'].nunique()} reviewers, "
          f"{df['product_id'].nunique()} products")
    print(df.dtypes)
