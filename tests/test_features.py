"""
Unit tests for Week 2 code.

Run with pytest (recommended):   pytest tests/ -v
Or without pytest:               python tests/test_features.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from preprocess import clean_text
from features import (
    _single_review_linguistic,
    add_review_level_behavior,
    add_self_similarity,
    flesch_reading_ease,
    sentiment_score,
)


def test_clean_text_removes_urls_html_and_punctuation():
    out = clean_text("Great!!! Visit http://x.com <b>now</b>")
    assert "http" not in out and "<" not in out and "!" not in out


def test_clean_text_keeps_negation():
    assert "not" in clean_text("This is not good")


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_exclamation_and_caps_features():
    f = _single_review_linguistic("AMAZING product!!!")
    assert f["exclamation_count"] == 3
    assert f["all_caps_word_ratio"] > 0
    assert abs(f["caps_ratio"] - 0.5) < 1e-9  # 7 capitals out of 14 letters


def test_empty_review_does_not_crash():
    f = _single_review_linguistic("")
    assert f["word_count"] == 0 and f["avg_word_length"] == 0.0


def test_sentiment_direction():
    assert sentiment_score("This is great and I love it") > 0
    assert sentiment_score("Terrible and awful, the worst") < 0


def test_readability_is_finite():
    assert np.isfinite(flesch_reading_ease("This is a short sentence. It reads easily."))


def test_rating_deviation_excludes_self():
    df = pd.DataFrame({
        "product_id": ["P1", "P1", "P1"],
        "rating": [5, 3, 1],
    })
    out = add_review_level_behavior(df)
    # review with rating 5: others average (3+1)/2 = 2 -> deviation +3
    assert out.loc[0, "rating_deviation"] == 3.0


def test_rating_deviation_uses_full_product_stats_when_available():
    # the product has 4 reviews in total (ratings sum to 10) but only 1 is in our sample
    df = pd.DataFrame({
        "product_id": ["P1"], "rating": [5.0],
        "product_rating_sum": [10.0], "product_rating_count": [4],
    })
    out = add_review_level_behavior(df)
    # others average (10 - 5) / 3 = 1.667 -> deviation 5 - 1.667 = 3.333
    assert abs(out.loc[0, "rating_deviation"] - (5 - 5 / 3)) < 1e-9


def test_cross_product_similarity_ignores_same_product_repeats():
    df = pd.DataFrame({
        "reviewer_id": ["A", "A", "B", "B"],
        "product_id":  ["P1", "P1", "P1", "P2"],
        "clean_text": ["love this brush set"] * 2 + ["great shea butter"] * 2,
    })
    out = add_self_similarity(df)
    # A repeated text on the SAME product -> similar overall, but not across products
    assert out.loc[0, "max_self_similarity"] > 0.99
    assert out.loc[0, "max_cross_product_similarity"] == 0.0
    # B pasted the same text on DIFFERENT products -> flagged by the cross-product feature
    assert out.loc[2, "max_cross_product_similarity"] > 0.99


def test_self_similarity_detects_duplicates():
    df = pd.DataFrame({
        "reviewer_id": ["A", "A", "B"],
        "clean_text": [
            "amazing product highly recommend",
            "amazing product highly recommend",
            "battery drains fast screen cracked",
        ],
    })
    out = add_self_similarity(df)
    assert out.loc[0, "max_self_similarity"] > 0.99   # duplicate text
    assert out.loc[2, "max_self_similarity"] == 0.0   # single review -> nothing to compare


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
