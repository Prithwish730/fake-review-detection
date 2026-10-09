"""
generate_sample_data.py

Generates a synthetic review dataset that mirrors the schema of the real
datasets this project is designed for:
  - Ott et al. Deceptive Opinion Spam Corpus  -> labeled genuine/fake reviews
  - Amazon Reviews (Kaggle)                   -> reviewer behavioral metadata

WHY THIS EXISTS:
Real labeled fake-review data isn't available in this dev environment (no
internet access). This script lets you build and test the ENTIRE pipeline
(EDA -> preprocessing -> features -> models -> trust score -> dashboard)
right now, using data with the same columns as the real thing.

WHEN YOU GET REAL DATA:
Download the real CSVs, put them in data/raw/, and point data_loader.py at
them instead (see the DATA SWAP section in data_loader.py). No other code
needs to change, because everything downstream reads the standardized
schema this script also produces.

Run:
    python src/generate_sample_data.py
Output:
    data/raw/reviews_sample.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

RANDOM_SEED = 42
N_REVIEWS = 3000
N_REVIEWERS = 450
N_PRODUCTS = 120

rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Vocabulary banks used to assemble synthetic review text.
# Genuine reviews: more specific, varied, mixed sentiment/rating.
# Fake reviews: more generic, superlative-heavy, sentiment/rating mismatch,
# templated phrasing (so similarity-based features will later be able to
# catch them, same as in real fake-review data).
# ---------------------------------------------------------------------------

GENUINE_TEMPLATES = [
    "Used this for {duration} and it {verb}. The {feature} is {opinion}, "
    "though the {flaw} could be better. Would {recommend}.",
    "Bought it for {reason}. {feature} works as expected, {flaw} is a minor "
    "annoyance. Overall {opinion} for the price.",
    "{duration} in and still holding up. Not perfect -- {flaw} -- but the "
    "{feature} makes up for it.",
    "Mixed feelings. {feature} is {opinion} but {flaw} really let it down "
    "for my use case ({reason}).",
]

FAKE_TEMPLATES = [
    "This product is AMAZING!!! Best purchase ever, changed my life "
    "completely. 5 stars, highly recommend to everyone!!!",
    "Absolutely perfect product, exceeded all my expectations, will "
    "definitely buy again, amazing quality, fast shipping, five stars!!!",
    "I love love LOVE this item so much, it is the best thing I have ever "
    "bought, everyone needs this, amazing amazing amazing!!!",
    "Great product great price great seller, highly recommended, five "
    "stars, will buy again, amazing experience overall!!!",
]

DURATIONS = ["a week", "two months", "six months", "a year", "three weeks"]
VERBS = ["held up well", "started wearing out", "performed great", "broke down"]
FEATURES = ["battery life", "build quality", "screen", "sound", "comfort"]
OPINIONS = ["solid", "decent", "disappointing", "excellent", "average"]
FLAWS = ["the battery drains fast", "it feels a bit cheap", "the app is buggy",
         "it's a bit heavy", "the instructions were unclear"]
REASONS = ["work", "travel", "gaming", "my kids", "daily commute"]
RECOMMENDS = ["recommend it to others", "not recommend it", "recommend it with caveats"]


def make_genuine_text():
    t = random.choice(GENUINE_TEMPLATES)
    return t.format(
        duration=random.choice(DURATIONS),
        verb=random.choice(VERBS),
        feature=random.choice(FEATURES),
        opinion=random.choice(OPINIONS),
        flaw=random.choice(FLAWS),
        reason=random.choice(REASONS),
        recommend=random.choice(RECOMMENDS),
    )


def make_fake_text():
    # Fake reviews draw from a SMALL template pool on purpose, so that
    # near-duplicate / cosine-similarity features have something real to
    # detect later, exactly like templated review-farm text in the wild.
    return random.choice(FAKE_TEMPLATES)


def random_date(start_days_ago=730):
    d = datetime.now() - timedelta(days=random.randint(0, start_days_ago))
    return d.strftime("%Y-%m-%d")


def generate_reviewers(n):
    reviewers = []
    for i in range(n):
        is_spammer = rng.random() < 0.18  # ~18% of reviewers are "fake-heavy"
        account_age_days = int(rng.integers(5, 2000))
        reviewers.append({
            "reviewer_id": f"R{i:05d}",
            "account_age_days": account_age_days,
            "is_spammer_profile": is_spammer,  # hidden ground truth, not a feature
        })
    return pd.DataFrame(reviewers)


def generate_reviews(reviewers_df, n_reviews, n_products):
    rows = []
    product_ids = [f"P{p:04d}" for p in range(n_products)]
    # products have a "true" average rating fake reviews will deviate from
    product_true_rating = {p: rng.uniform(2.5, 4.5) for p in product_ids}

    for i in range(n_reviews):
        reviewer = reviewers_df.sample(1, random_state=None).iloc[0]
        is_fake = bool(reviewer["is_spammer_profile"]) and rng.random() < 0.75
        product_id = random.choice(product_ids)

        if is_fake:
            text = make_fake_text()
            rating = 5 if rng.random() < 0.85 else int(rng.integers(1, 3))
            verified = rng.random() < 0.3
        else:
            text = make_genuine_text()
            base = product_true_rating[product_id]
            rating = int(np.clip(round(rng.normal(base, 0.8)), 1, 5))
            verified = rng.random() < 0.8

        rows.append({
            "review_id": f"REV{i:06d}",
            "reviewer_id": reviewer["reviewer_id"],
            "product_id": product_id,
            "rating": rating,
            "review_text": text,
            "date": random_date(),
            "verified_purchase": verified,
            "label": "fake" if is_fake else "genuine",  # ground truth for training
        })

    return pd.DataFrame(rows)


def main():
    reviewers_df = generate_reviewers(N_REVIEWERS)
    reviews_df = generate_reviews(reviewers_df, N_REVIEWS, N_PRODUCTS)

    # merge in account_age_days as reviewer metadata available at review time
    merged = reviews_df.merge(
        reviewers_df[["reviewer_id", "account_age_days"]],
        on="reviewer_id", how="left"
    )
    merged = merged.sort_values("date").reset_index(drop=True)

    out_path = "data/raw/reviews_sample.csv"
    merged.to_csv(out_path, index=False)
    print(f"Wrote {len(merged)} rows to {out_path}")
    print(merged["label"].value_counts(normalize=True).rename("proportion"))


if __name__ == "__main__":
    main()
