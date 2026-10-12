"""
build_features.py

Runs the feature pipeline for ONE data source.

    python src/build_features.py                                  # synthetic sample
    python src/build_features.py --source ott                     # real labeled data
    python src/build_features.py --source ott --path data/raw/deceptive-opinion.csv
    python src/build_features.py --source amazon --path data/raw/All_Beauty.jsonl

Outputs (prefixed by source so datasets never overwrite each other):
    data/processed/<source>_review_features.csv     one row per review
    data/processed/<source>_reviewer_features.csv   one row per reviewer
                                                    (only if the source has reviewer info)

Ott has NO reviewer/date info, so only text + linguistic features are built.
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_reviews
from preprocess import add_clean_text, lemmatization_available
from features import (
    add_linguistic_features,
    add_review_level_behavior,
    add_self_similarity,
    build_reviewer_features,
    sentiment_backend,
)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LINGUISTIC_COLS = [
    "char_count", "word_count", "avg_word_length", "exclamation_count",
    "exclamation_ratio", "question_count", "caps_ratio", "all_caps_word_ratio",
    "punctuation_ratio", "digit_ratio", "unique_word_ratio", "first_person_ratio",
    "intensifier_ratio", "flesch_reading_ease", "sentiment_score",
    "rating_sentiment_mismatch",
]

# sources that include reviewer + date info (behavioral features make sense)
HAS_REVIEWER_INFO = {"sample", "amazon"}


def parse_args():
    ap = argparse.ArgumentParser(description="Build features for one data source")
    ap.add_argument("--source", choices=["sample", "ott", "amazon"], default="sample")
    ap.add_argument("--path", default=None, help="file or folder with the dataset")
    ap.add_argument("--nrows", type=int, default=None, help="amazon: max rows to read")
    ap.add_argument("--min-reviews", type=int, default=3,
                    help="amazon: keep reviewers with at least this many reviews")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="amazon: keep exact duplicate reviews (default: drop them)")
    ap.add_argument("--max-reviewers", type=int, default=5000,
                    help="amazon: randomly keep at most this many reviewers")
    return ap.parse_args()


def main():
    args = parse_args()
    src = args.source

    print(f"1/5 Loading data (source = {src})...")
    kwargs = {}
    if src == "amazon":
        kwargs = {"min_reviews": args.min_reviews, "max_reviewers": args.max_reviewers,
                  "drop_duplicates": not args.keep_duplicates}
        if args.nrows:
            kwargs["nrows"] = args.nrows
    df = load_reviews(src, path=args.path, **kwargs)
    print(f"    {len(df)} reviews, {df['reviewer_id'].nunique()} reviewers")

    print("2/5 Cleaning text "
          f"(lemmatization: {'on' if lemmatization_available() else 'off'})...")
    df = add_clean_text(df)

    print(f"3/5 Linguistic features (sentiment backend: {sentiment_backend()})...")
    df = add_linguistic_features(df)

    reviewer_df = None
    if src in HAS_REVIEWER_INFO:
        print("4/5 Behavioral features...")
        df = add_review_level_behavior(df)
        df = add_self_similarity(df)
        print("5/5 Reviewer-level features...")
        reviewer_df = build_reviewer_features(df)
    else:
        print("4/5 Behavioral features: skipped (this dataset has no reviewer/date info)")
        print("5/5 Reviewer-level features: skipped")

    review_path = PROCESSED_DIR / f"{src}_review_features.csv"
    df.to_csv(review_path, index=False)
    print(f"\nSaved {review_path.name}: {df.shape[0]} rows x {df.shape[1]} cols")
    if reviewer_df is not None:
        reviewer_path = PROCESSED_DIR / f"{src}_reviewer_features.csv"
        reviewer_df.to_csv(reviewer_path, index=False)
        print(f"Saved {reviewer_path.name}: {reviewer_df.shape[0]} rows x {reviewer_df.shape[1]} cols")

    labeled = df.dropna(subset=["label"])
    if len(labeled):
        check_cols = [c for c in LINGUISTIC_COLS if c in df.columns
                      and df[c].notna().any()]
        if "abs_rating_deviation" in df.columns:
            check_cols += ["abs_rating_deviation", "max_self_similarity"]
            if "max_cross_product_similarity" in df.columns:
                check_cols.append("max_cross_product_similarity")
        print(f"\nLabel balance: {labeled['label'].value_counts(normalize=True).round(3).to_dict()}")
        print("Feature means by label (big gaps = useful features):")
        print(labeled.groupby("label")[check_cols].mean().T.round(3).to_string())


if __name__ == "__main__":
    main()
