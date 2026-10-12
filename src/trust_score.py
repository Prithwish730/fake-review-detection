"""
trust_score.py

Reviewer Trust Score (0-100, higher = more trustworthy), built from behavior.

Amazon data has NO fake/genuine labels, so we cannot train a supervised model.
Instead we build TWO independent scores and combine them:

  1. HEURISTIC score   - weighted sum of interpretable "suspiciousness" signals.
                         Easy to explain in a dashboard and in a viva.
  2. LEARNED score     - Isolation Forest (unsupervised anomaly detection) over
                         the same behavioral features. Needs no hand-picked weights.

  trust_score = average of the two.

IMPORTANT: scores are RELATIVE to the population of reviewers in the dataset.
"Trust 20" means "more suspicious than most reviewers here", not "proven fake".

VALIDATION without labels (two checks, both printed by this script):
  a) Agreement: do two very different methods rank reviewers similarly?
  b) Planted-spammer test: we inject synthetic spammers (burst posting, templated
     5-star text, unverified) into the real data, rerun the whole pipeline, and
     measure how many land at the bottom. This proves the pipeline CAN find such
     patterns; it does not prove real fraud exists in the data.

Run:
    python src/trust_score.py
    python src/trust_score.py --flag-pct 5 --n-planted 100
    python src/trust_score.py --no-planted-test
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Heuristic weights (sum to 1). Justify these with your EDA in the report;
# if a signal is unavailable (e.g. no verified flag) the rest are renormalized.
DEFAULT_WEIGHTS = {
    "template_reuse": 0.25,   # re-posting near-identical text
    "burstiness": 0.20,       # many reviews in a single day
    "rating_deviation": 0.15, # ratings far from each product's average
    "five_star_share": 0.15,  # almost always 5 stars
    "unverified": 0.15,       # reviews without a verified purchase
    "velocity": 0.10,         # reviews per 30 days of activity
}

# component -> reviewer-level column it is built from
COMPONENT_SOURCE = {   # first available column in each list is used
    "template_reuse": ["avg_cross_product_similarity", "avg_self_similarity"],
    "burstiness": ["max_reviews_in_one_day"],
    "rating_deviation": ["avg_abs_rating_deviation"],
    "five_star_share": ["pct_5star"],
    "velocity": ["review_velocity"],
    "unverified": ["verified_ratio"],
}

# (column, apply log1p?) used by the Isolation Forest
IF_FEATURES = [
    ("review_count", True), ("pct_5star", False), ("rating_std", False),
    ("verified_ratio", False), ("avg_abs_rating_deviation", False),
    ("avg_self_similarity", False), ("avg_cross_product_similarity", False),
    ("max_reviews_in_one_day", True),
    ("review_velocity", True), ("avg_gap_days", True),
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def rank01(series: pd.Series) -> pd.Series:
    """Map values to [0, 1] by rank. Ties at the lowest value get exactly 0."""
    s = series.astype(float)
    s = s.fillna(s.median())
    n = len(s)
    if n < 2 or s.nunique() < 2:
        return pd.Series(0.0, index=s.index)
    return (s.rank(method="min") - 1) / (n - 1)


# ----------------------------------------------------------------------
# 1. Heuristic score
# ----------------------------------------------------------------------
def heuristic_trust(reviewers: pd.DataFrame, weights=None) -> pd.DataFrame:
    """Return per-component suspiciousness (s_*, 0-1) and the heuristic trust."""
    weights = dict(weights or DEFAULT_WEIGHTS)
    comp = pd.DataFrame(index=reviewers.index)

    for name, options in COMPONENT_SOURCE.items():
        col = next((c for c in options
                    if c in reviewers.columns and reviewers[c].notna().any()), None)
        if col is None:
            weights.pop(name, None)       # signal unavailable -> drop & renormalize
            continue
        if name == "unverified":
            comp["s_unverified"] = (1.0 - reviewers[col].astype(float)).fillna(0.0)
        else:
            comp[f"s_{name}"] = rank01(reviewers[col])

    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    suspicion = sum(comp[f"s_{k}"] * w for k, w in weights.items())
    comp["heuristic_trust"] = 100.0 * (1.0 - suspicion)
    comp.attrs["weights_used"] = weights
    return comp


# ----------------------------------------------------------------------
# 2. Learned score (Isolation Forest)
# ----------------------------------------------------------------------
def isolation_forest_trust(reviewers: pd.DataFrame, random_state=42) -> pd.Series:
    cols, data = [], {}
    for col, use_log in IF_FEATURES:
        if col in reviewers.columns and reviewers[col].notna().any():
            v = reviewers[col].astype(float)
            v = v.fillna(v.median())
            data[col] = np.log1p(v.clip(lower=0)) if use_log else v
            cols.append(col)
    X = StandardScaler().fit_transform(pd.DataFrame(data))
    model = IsolationForest(n_estimators=300, contamination="auto",
                            random_state=random_state).fit(X)
    anomaly = pd.Series(-model.score_samples(X), index=reviewers.index)  # higher = weirder
    return 100.0 * (1.0 - rank01(anomaly))


# ----------------------------------------------------------------------
# Combine
# ----------------------------------------------------------------------
def compute_trust_scores(reviewers: pd.DataFrame, weights=None, random_state=42) -> pd.DataFrame:
    comp = heuristic_trust(reviewers, weights)
    out = reviewers.copy()
    for c in comp.columns:
        out[c] = comp[c]
    out["learned_trust"] = isolation_forest_trust(reviewers, random_state)
    out["trust_score"] = (0.5 * out["heuristic_trust"] + 0.5 * out["learned_trust"]).round(1)
    out["heuristic_trust"] = out["heuristic_trust"].round(1)
    out["learned_trust"] = out["learned_trust"].round(1)
    out.attrs["weights_used"] = comp.attrs["weights_used"]
    return out


# ----------------------------------------------------------------------
# Planted-spammer validation
# ----------------------------------------------------------------------
PLANT_TEMPLATES = [
    "Absolutely love this! Best purchase ever, works perfectly and arrived fast. Highly recommend to everyone, five stars!",
    "Amazing quality, exceeded my expectations. Will definitely buy again, great seller, five stars!!!",
    "Perfect product, I love it so much. Everyone needs this, best thing I have ever bought!",
    "Great product great price, fast shipping, highly recommended, five stars, will buy again!",
]
PLANT_SUFFIXES = ["", " Thanks!", " Wow.", " Really happy.", " Love it."]


def plant_spammers(reviews: pd.DataFrame, n_spammers=50, seed=42) -> pd.DataFrame:
    """
    Build synthetic spammer reviews (half 'burst' posters, half 'template' posters).
    Their IDs start with PLANTED_ so we can find them again.
    """
    rng = np.random.default_rng(seed)
    products = reviews["product_id"].unique()
    last_date = pd.to_datetime(reviews["date"]).max()
    rows = []
    for i in range(n_spammers):
        kind = "burst" if i % 2 == 0 else "template"
        rid = f"PLANTED_{kind}_{i:03d}"
        template = PLANT_TEMPLATES[i % len(PLANT_TEMPLATES)]
        base_day = last_date - pd.Timedelta(days=int(rng.integers(0, 300)))
        for j in range(6 if kind == "burst" else 5):
            day = base_day if kind == "burst" else base_day - pd.Timedelta(days=int(rng.integers(0, 90)))
            rows.append({
                "review_id": f"{rid}_{j}", "reviewer_id": rid,
                "product_id": rng.choice(products),
                "rating": 5.0 if rng.random() < 0.9 else 4.0,
                "review_text": template + str(rng.choice(PLANT_SUFFIXES)),
                "date": day, "verified_purchase": bool(rng.random() < 0.2),
                "account_age_days": np.nan,
            })
    return pd.DataFrame(rows)


def run_planted_test(review_path: Path, n_planted=50, flag_pct=10, seed=42) -> pd.DataFrame:
    from preprocess import add_clean_text
    from features import (add_review_level_behavior, add_self_similarity,
                          build_reviewer_features)

    need = ["review_id", "reviewer_id", "product_id", "rating", "date",
            "verified_purchase", "clean_text", "account_age_days"]
    header = pd.read_csv(review_path, nrows=0).columns
    extra = [c for c in ("product_rating_sum", "product_rating_count") if c in header]
    base = pd.read_csv(review_path, usecols=need + extra, parse_dates=["date"])
    base["clean_text"] = base["clean_text"].fillna("")

    planted = add_clean_text(plant_spammers(base, n_planted, seed))
    allr = pd.concat([base, planted], ignore_index=True)
    allr = add_review_level_behavior(allr)
    allr = add_self_similarity(allr)
    scored = compute_trust_scores(build_reviewer_features(allr))

    scored["is_planted"] = scored["reviewer_id"].str.startswith("PLANTED_")
    k = max(1, int(round(len(scored) * flag_pct / 100)))
    rows = []
    for col in ("heuristic_trust", "learned_trust", "trust_score"):
        bottom = scored.nsmallest(k, col)
        lowest_n = scored.nsmallest(n_planted, col)   # K = number planted
        rows.append({
            "score": col,
            "roc_auc": roc_auc_score(scored["is_planted"], 100 - scored[col]),
            f"recall_in_bottom_{flag_pct}%": bottom["is_planted"].sum() / n_planted,
            f"hit_rate_in_lowest_{n_planted}": lowest_n["is_planted"].mean(),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def plot_distribution(scores: pd.DataFrame, flag_pct):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, label in [("heuristic_trust", "Heuristic"), ("learned_trust", "Isolation Forest"),
                       ("trust_score", "Final (average)")]:
        ax.hist(scores[col], bins=40, alpha=0.5, label=label)
    cutoff = scores["trust_score"].quantile(flag_pct / 100)
    ax.axvline(cutoff, color="red", ls="--", label=f"flag cutoff (bottom {flag_pct}%)")
    ax.set(xlabel="Trust score (0-100)", ylabel="Reviewers", title="Reviewer Trust Score Distribution")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_trust_score_distribution.png", dpi=150)
    plt.close()


def plot_agreement(scores: pd.DataFrame, rho: float):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(scores["heuristic_trust"], scores["learned_trust"], s=6, alpha=0.35)
    ax.set(xlabel="Heuristic trust", ylabel="Isolation Forest trust",
           title=f"Do the two methods agree? (Spearman = {rho:.2f})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_trust_score_agreement.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="amazon")
    ap.add_argument("--flag-pct", type=float, default=10, help="flag the bottom X%% of reviewers")
    ap.add_argument("--n-planted", type=int, default=50)
    ap.add_argument("--no-planted-test", action="store_true")
    args = ap.parse_args()

    reviewer_path = PROCESSED / f"{args.source}_reviewer_features.csv"
    review_path = PROCESSED / f"{args.source}_review_features.csv"
    if not reviewer_path.exists():
        raise FileNotFoundError(
            f"{reviewer_path} not found. Run `python src/build_features.py "
            f"--source {args.source} --path <file>` first.")

    reviewers = pd.read_csv(reviewer_path)
    scores = compute_trust_scores(reviewers)
    cutoff = scores["trust_score"].quantile(args.flag_pct / 100)
    scores["flagged"] = scores["trust_score"] <= cutoff

    out_path = PROCESSED / f"{args.source}_trust_scores.csv"
    scores.sort_values("trust_score").to_csv(out_path, index=False)

    rho = scores["heuristic_trust"].corr(scores["learned_trust"], method="spearman")
    plot_distribution(scores, args.flag_pct)
    plot_agreement(scores, rho)

    print(f"Scored {len(scores):,} reviewers -> {out_path.name}\n")
    print("Heuristic weights used:",
          {k: round(v, 3) for k, v in scores.attrs["weights_used"].items()})
    if "s_unverified" not in scores.columns:
        print("  (no verified-purchase data -> 'unverified' signal dropped)")
    print("\nTrust score distribution:")
    print(scores["trust_score"].describe().round(1).to_string())
    print(f"\nAgreement between heuristic and Isolation Forest (Spearman): {rho:.2f}")

    show = ["reviewer_id", "trust_score", "review_count", "pct_5star",
            "max_reviews_in_one_day", "avg_self_similarity", "verified_ratio"]
    show = [c for c in show if c in scores.columns]
    print(f"\n10 least trusted reviewers (inspect a few by hand for your report):")
    print(scores.sort_values("trust_score")[show].head(10).round(2).to_string(index=False))

    if not args.no_planted_test and review_path.exists():
        print("\nPlanted-spammer test (injecting fake spammers and re-running the pipeline)...")
        res = run_planted_test(review_path, args.n_planted, int(args.flag_pct))
        print(res.round(3).to_string(index=False))
        print("roc_auc: how well the score separates planted from real reviewers (1.0 = perfect).")
        print(f"hit_rate_in_lowest_N: of the N lowest-scored reviewers (N = number planted), "
              f"the share that are planted.")
        print("(Planted spammers are synthetic: this checks the pipeline can find such "
              "patterns, not that real fraud exists in the data.)")
    print("\nFigures: reports/figures/10_trust_score_distribution.png, 11_trust_score_agreement.png")


if __name__ == "__main__":
    main()
