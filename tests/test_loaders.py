"""
Tests for the real-data loaders (use small temporary files, no downloads needed).

Run with pytest:   pytest tests/ -v
Or plain python:   python tests/test_loaders.py
"""

import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from data_loader import load_amazon_reviews, load_ott_corpus


def _write_ott_csv(tmp: Path) -> Path:
    p = tmp / "deceptive-opinion.csv"
    pd.DataFrame({
        "deceptive": ["truthful", "deceptive", "truthful", "deceptive"],
        "hotel": ["hilton", "hilton", "hyatt", "hyatt"],
        "polarity": ["positive", "positive", "negative", "negative"],
        "source": ["TripAdvisor", "MTurk", "Web", "MTurk"],
        "text": ["ok stay", "AMAZING!!!", "noisy room", "worst ever"],
    }).to_csv(p, index=False)
    return p


def test_ott_csv_labels_are_mapped_to_schema():
    with tempfile.TemporaryDirectory() as t:
        df = load_ott_corpus(_write_ott_csv(Path(t)))
    assert df["label"].tolist() == ["genuine", "fake", "genuine", "fake"]
    assert df["reviewer_id"].is_unique            # one synthetic reviewer per review
    assert df["rating"].isna().all()               # Ott has no ratings


def test_ott_folder_label_ignores_name_of_root_folder():
    # If the user's own folder is called "Deceptive Opinion Spam Corpus",
    # that must NOT make every file look deceptive.
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "Deceptive Opinion Spam Corpus" / "op_spam_v1.4"
        for kind, d in [("t", "truthful_from_Web"), ("d", "deceptive_from_MTurk")]:
            f = root / "negative_polarity" / d / "fold1"
            f.mkdir(parents=True)
            (f / f"{kind}_hilton_1.txt").write_text("some review text")
        (root / "README.txt").write_text("ignore me")
        df = load_ott_corpus(root)
    assert sorted(df["label"]) == ["fake", "genuine"]
    assert len(df) == 2                            # README skipped


def test_amazon_jsonl_2023_format_and_millisecond_timestamps():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "reviews.jsonl"
        ts = int(pd.Timestamp("2022-06-01").timestamp() * 1000)
        with open(p, "w") as f:
            for i in range(4):
                f.write(json.dumps({"user_id": "U1", "parent_asin": f"P{i}", "rating": 5.0,
                                    "text": "nice", "timestamp": ts + i * 86_400_000,
                                    "verified_purchase": True}) + "\n")
        df = load_amazon_reviews(p, min_reviews=3)
    assert df["date"].dt.year.eq(2022).all()       # ms detected, not read as seconds
    assert df["verified_purchase"].all()


def test_amazon_min_reviews_filter_drops_one_time_reviewers():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "Reviews.csv"
        rows = [{"UserId": "A", "ProductId": f"P{i}", "Score": 4, "Time": 1_600_000_000 + i,
                 "Text": "fine"} for i in range(3)]
        rows.append({"UserId": "B", "ProductId": "P9", "Score": 1, "Time": 1_600_000_100,
                     "Text": "bad"})
        pd.DataFrame(rows).to_csv(p, index=False)
        df = load_amazon_reviews(p, min_reviews=3)
    assert set(df["reviewer_id"]) == {"A"}
    assert df["verified_purchase"].isna().all()    # column absent in this format


def test_amazon_exact_duplicates_dropped_by_default_and_optional():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "Reviews.csv"
        rows = [{"UserId": "A", "ProductId": "P1", "Score": 5, "Time": 1_600_000_000,
                 "Text": "same text"}] * 3                       # 3 exact copies
        rows += [{"UserId": "A", "ProductId": f"P{i}", "Score": 4, "Time": 1_600_000_000 + i,
                  "Text": f"different {i}"} for i in range(2, 5)]
        pd.DataFrame(rows).to_csv(p, index=False)
        deduped = load_amazon_reviews(p, min_reviews=3)
        kept = load_amazon_reviews(p, min_reviews=3, drop_duplicates=False)
    assert len(deduped) == 4          # 3 copies collapsed to 1, plus 3 distinct reviews
    assert len(kept) == 6


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
