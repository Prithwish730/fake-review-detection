"""
Tests for explain.py: the per-word contributions must add up to the model's own score.

Run with pytest:   pytest tests/ -v
Or plain python:   python tests/test_explain.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from explain import analyze_review, explain_row, linear_weights, top_words
from features import _single_review_linguistic
from preprocess import clean_text

FAKE = ["amazing perfect best hotel love it highly recommend", "best ever perfect love amazing stay",
        "perfect luxury amazing wonderful love recommend", "love this hotel best amazing perfect"] * 6
REAL = ["room small noisy elevator slow parking expensive", "paid for wifi walked to the train street noisy",
        "booked tuesday small clean bathroom towels late", "parking paid elevator broken breakfast fine"] * 6


def _train(clf):
    texts = FAKE + REAL
    y = [1] * len(FAKE) + [0] * len(REAL)
    df = pd.DataFrame([{**_single_review_linguistic(t), "clean_text": clean_text(t)} for t in texts])
    pipe = Pipeline([
        ("features", ColumnTransformer([("tfidf", TfidfVectorizer(), "clean_text")])),
        ("clf", clf),
    ]).fit(df, y)
    return pipe


def _row(text):
    return pd.DataFrame([{**_single_review_linguistic(text), "clean_text": clean_text(text)}])


def test_contributions_add_up_to_the_models_own_score_logreg_and_svm():
    for clf in (LogisticRegression(max_iter=1000), LinearSVC()):
        pipe = _train(clf)
        row = _row("amazing perfect hotel but parking paid")
        contribs, intercept = explain_row(pipe, row)
        assert abs(contribs["contribution"].sum() + intercept - pipe.decision_function(row)[0]) < 1e-9


def test_naive_bayes_contributions_match_log_odds():
    pipe = _train(MultinomialNB())
    row = _row("amazing perfect hotel but parking paid")
    contribs, intercept = explain_row(pipe, row)
    proba = pipe.predict_proba(row)[0]
    assert abs(contribs["contribution"].sum() + intercept - np.log(proba[1] / proba[0])) < 1e-6


def test_top_words_point_the_right_way():
    fake, genuine = top_words(_train(LogisticRegression(max_iter=1000)), n=5)
    assert {"amazing", "perfect", "love"} & set(fake["word"])
    assert {"parking", "elevator", "noisy"} & set(genuine["word"])


def test_analyze_review_direction_and_fields():
    pipe = _train(LogisticRegression(max_iter=1000))
    fake_like = analyze_review("Best hotel ever, amazing and perfect, I love it!!!", pipe)
    real_like = analyze_review("Small room, noisy elevator, paid for parking.", pipe)
    assert fake_like["p_fake"] > 0.5 > real_like["p_fake"]
    assert 0.0 <= fake_like["vocab_coverage"] <= 1.0
    assert "amazing" in set(fake_like["top_fake"]["feature"])
    assert "exclamation_count" in fake_like["style"]


def test_unknown_vocabulary_gives_low_coverage():
    res = analyze_review("zxqv blorf quuxle snarf", _train(LogisticRegression(max_iter=1000)))
    assert res["vocab_coverage"] == 0.0


def test_empty_text_does_not_crash():
    res = analyze_review("", _train(LogisticRegression(max_iter=1000)))
    assert res["n_words"] == 0 and 0.0 <= res["p_fake"] <= 1.0


def test_non_linear_model_is_rejected_with_a_clear_message():
    pipe = _train(RandomForestClassifier(n_estimators=10, random_state=0))
    try:
        linear_weights(pipe.named_steps["clf"])
    except ValueError as e:
        assert "linear" in str(e)
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
