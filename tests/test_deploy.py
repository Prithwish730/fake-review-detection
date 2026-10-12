"""
Tests for the deployment helpers (use temporary folders, nothing real is touched).

Run with pytest:   pytest tests/ -v
Or plain python:   python tests/test_deploy.py
"""

import sys
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy"))

import joblib
import pandas as pd

from pin_requirements import MARKER, write_requirements
from prepare_deploy import build_assets


def _fake_project(tmp: Path):
    (tmp / "models").mkdir()
    (tmp / "data" / "processed").mkdir(parents=True)
    (tmp / "reports" / "figures").mkdir(parents=True)
    joblib.dump({"pretend": "model"}, tmp / "models" / "explainable_lr_ott.joblib")
    pd.DataFrame({"reviewer_id": ["A", "B"], "trust_score": [10.0, 90.0]}).to_csv(
        tmp / "data" / "processed" / "amazon_trust_scores.csv", index=False)
    pd.DataFrame({
        "reviewer_id": ["A", "A", "B"], "product_id": ["P1", "P2", "P3"], "rating": [5, 5, 4],
        "date": ["2020-01-01"] * 3, "verified_purchase": [False, False, True],
        "review_text": ["x" * 1000, "short   text\nwith   spaces", "ok"],
        "some_huge_column": [1, 2, 3],
    }).to_csv(tmp / "data" / "processed" / "amazon_review_features.csv", index=False)
    (tmp / "reports" / "figures" / "07_roc_curves_ott.png").write_bytes(b"png")


def test_assets_are_built_with_shortened_text_and_only_needed_columns():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        _fake_project(tmp)
        build_assets(tmp, max_chars=50)
        slim = pd.read_csv(tmp / "deploy_assets" / "amazon_reviews_slim.csv.gz")
        assert "some_huge_column" not in slim.columns
        assert slim["review_text"].str.len().max() <= 50
        assert "short text with spaces" in set(slim["review_text"])     # whitespace tidied
        assert (tmp / "deploy_assets" / "figures" / "07_roc_curves_ott.png").exists()
        assert joblib.load(tmp / "deploy_assets" / "explainable_lr_ott.joblib") == {"pretend": "model"}


def test_no_text_option_removes_review_text_entirely():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        _fake_project(tmp)
        build_assets(tmp, include_text=False)
        slim = pd.read_csv(tmp / "deploy_assets" / "amazon_reviews_slim.csv.gz")
        assert "review_text" not in slim.columns


def test_missing_input_gives_a_helpful_message():
    with tempfile.TemporaryDirectory() as t:
        try:
            build_assets(Path(t))
        except FileNotFoundError as e:
            assert "python src/" in str(e)         # tells the user which command creates it
        else:
            raise AssertionError("expected FileNotFoundError")


def test_requirements_are_pinned_and_old_list_is_preserved_once():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "requirements.txt").write_text("torch>=2.1\nxgboost>=2.0\n")
        write_requirements(tmp)
        text = (tmp / "requirements.txt").read_text()
        assert MARKER in text and "scikit-learn==" in text and "torch" not in text
        assert (tmp / "requirements-train.txt").read_text() == "torch>=2.1\nxgboost>=2.0\n"
        write_requirements(tmp)                     # running again must not clobber the backup
        assert "torch" in (tmp / "requirements-train.txt").read_text()


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
