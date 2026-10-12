"""
Tests for Week 3 training code.

Run with pytest:   pytest tests/ -v
Or plain python:   python tests/test_models.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from train_classical import compute_metrics, grouped_stratified_split


def test_metrics_perfect_prediction():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["f1"] == 1.0 and m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0


def test_metrics_all_wrong_has_zero_f1():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    m = compute_metrics(y_true, y_pred, np.array([0.9, 0.8, 0.2, 0.1]))
    assert m["f1"] == 0.0 and m["accuracy"] == 0.0


def test_metrics_accuracy_can_mislead_on_imbalance():
    # predicting "genuine" for everything: 90% accuracy but catches zero fakes
    y_true = np.array([0] * 9 + [1])
    y_pred = np.zeros(10, dtype=int)
    m = compute_metrics(y_true, y_pred, np.zeros(10) + 0.1)
    assert m["accuracy"] == 0.9 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_split_never_shares_reviewers_between_train_and_test():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"reviewer_id": [f"R{i % 40}" for i in range(400)]})
    y = (rng.random(400) < 0.2).astype(int)
    train_idx, test_idx = grouped_stratified_split(df, y)
    train_r = set(df.iloc[train_idx]["reviewer_id"])
    test_r = set(df.iloc[test_idx]["reviewer_id"])
    assert train_r.isdisjoint(test_r)
    assert len(train_idx) + len(test_idx) == 400


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
