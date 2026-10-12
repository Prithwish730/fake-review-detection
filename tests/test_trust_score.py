"""
Tests for the trust score module.

Run with pytest:   pytest tests/ -v
Or plain python:   python tests/test_trust_score.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from trust_score import compute_trust_scores, heuristic_trust, rank01


def _reviewers(n=200, seed=0):
    """Mostly ordinary reviewers plus 10 obvious spammers at the end."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "reviewer_id": [f"R{i}" for i in range(n)],
        "review_count": rng.integers(3, 8, n),
        "avg_self_similarity": rng.uniform(0.0, 0.2, n),
        "max_reviews_in_one_day": rng.integers(1, 3, n),
        "avg_abs_rating_deviation": rng.uniform(0.2, 1.0, n),
        "pct_5star": rng.uniform(0.2, 0.7, n),
        "review_velocity": rng.uniform(0.1, 2.0, n),
        "verified_ratio": rng.uniform(0.7, 1.0, n),
        "rating_std": rng.uniform(0.5, 1.5, n),
        "avg_gap_days": rng.uniform(10, 120, n),
    })
    df.loc[n - 10:, "avg_self_similarity"] = 0.95
    df.loc[n - 10:, "max_reviews_in_one_day"] = 6
    df.loc[n - 10:, "pct_5star"] = 1.0
    df.loc[n - 10:, "verified_ratio"] = 0.0
    df.loc[n - 10:, "review_velocity"] = 40.0
    df.loc[n - 10:, "avg_abs_rating_deviation"] = 2.5
    df.loc[n - 10:, "avg_gap_days"] = 0.0
    return df


def test_rank01_ties_at_lowest_value_get_zero():
    r = rank01(pd.Series([0, 0, 0, 1, 2]))
    assert (r.iloc[:3] == 0).all() and r.iloc[-1] == 1


def test_rank01_constant_column_is_all_zero():
    assert (rank01(pd.Series([5, 5, 5, 5])) == 0).all()


def test_scores_are_between_0_and_100():
    s = compute_trust_scores(_reviewers())
    for col in ("heuristic_trust", "learned_trust", "trust_score"):
        assert s[col].between(0, 100).all()


def test_obvious_spammers_score_lower_than_normal_reviewers():
    s = compute_trust_scores(_reviewers())
    spam, normal = s.iloc[-10:], s.iloc[:-10]
    assert spam["trust_score"].mean() < normal["trust_score"].mean() - 20
    # every spammer should be in the lowest 15% of reviewers
    cutoff = s["trust_score"].quantile(0.15)
    assert (spam["trust_score"] <= cutoff).all()


def test_missing_verified_column_drops_signal_and_renormalizes():
    df = _reviewers().assign(verified_ratio=np.nan)
    comp = heuristic_trust(df)
    assert "s_unverified" not in comp.columns
    assert abs(sum(comp.attrs["weights_used"].values()) - 1.0) < 1e-9
    assert comp["heuristic_trust"].between(0, 100).all()


def test_scoring_is_deterministic():
    a = compute_trust_scores(_reviewers())["trust_score"]
    b = compute_trust_scores(_reviewers())["trust_score"]
    assert (a == b).all()


def test_heuristic_prefers_cross_product_similarity_when_available():
    df = _reviewers()
    # same-product repeats give everyone high plain similarity, but only the last
    # 10 reviewers paste text across DIFFERENT products
    df["avg_self_similarity"] = 0.9
    df["avg_cross_product_similarity"] = 0.0
    df.loc[len(df) - 10:, "avg_cross_product_similarity"] = 0.95
    comp = heuristic_trust(df)
    assert comp["s_template_reuse"].iloc[-10:].min() > comp["s_template_reuse"].iloc[:-10].max()


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
