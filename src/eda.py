"""
eda.py

Exploratory Data Analysis for the Fake Review Detection project.
Produces 6 figures into reports/figures/ and prints a text summary you can
paste directly into your report's EDA chapter.

Run:
    python src/eda.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display in this environment; saves straight to file
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_reviews

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

STOPWORDS = set("""a an the is are was were be been being to of and or in on
at for with as by this that it its it's i you he she they we not no do does
did have has had but if so just very can could would should""".split())


def simple_tokenize(text: str):
    return [w.strip(".,!?\"'()") .lower() for w in str(text).split()]


def plot_label_distribution(df):
    labeled = df.dropna(subset=["label"])
    fig, ax = plt.subplots(figsize=(5, 4))
    labeled["label"].value_counts().plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_title("Genuine vs Fake Review Counts")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_label_distribution.png", dpi=150)
    plt.close()


def plot_rating_distribution_by_label(df):
    labeled = df.dropna(subset=["label"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=labeled, x="rating", hue="label", multiple="dodge",
                 discrete=True, shrink=0.8, ax=ax)
    ax.set_title("Rating Distribution: Genuine vs Fake")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_rating_by_label.png", dpi=150)
    plt.close()


def plot_review_length_by_label(df):
    d = df.dropna(subset=["label"]).copy()
    d["word_count"] = d["review_text"].apply(lambda t: len(str(t).split()))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=d, x="label", y="word_count", ax=ax)
    ax.set_title("Review Length (words) by Label")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_review_length_by_label.png", dpi=150)
    plt.close()
    return d["word_count"]


def plot_reviews_per_reviewer(df):
    counts = df.groupby("reviewer_id").size()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(counts, bins=30, ax=ax)
    ax.set_title("Reviews per Reviewer (potential burst/spam signal)")
    ax.set_xlabel("Number of reviews")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_reviews_per_reviewer.png", dpi=150)
    plt.close()
    return counts


def plot_top_words(df, label, filename, top_n=15):
    d = df[df["label"] == label] if label else df
    freq = {}
    for text in d["review_text"]:
        for tok in simple_tokenize(text):
            if tok and tok not in STOPWORDS and len(tok) > 2:
                freq[tok] = freq.get(tok, 0) + 1
    top = sorted(freq.items(), key=lambda x: -x[1])[:top_n]
    words, counts = zip(*top) if top else ([], [])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=list(counts), y=list(words), ax=ax, color="#55A868")
    ax.set_title(f"Top {top_n} Words — {label or 'All Reviews'}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.close()


def plot_verified_purchase_rate(df):
    labeled = df.dropna(subset=["label"])
    rate = labeled.groupby("label")["verified_purchase"].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    rate.plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_title("Verified Purchase Rate by Label")
    ax.set_ylabel("Proportion Verified")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_verified_purchase_rate.png", dpi=150)
    plt.close()
    return rate


def main():
    df = load_reviews()
    print(f"Dataset: {len(df)} reviews | {df['reviewer_id'].nunique()} reviewers | "
          f"{df['product_id'].nunique()} products")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Missing values per column:\n{df.isna().sum()}\n")

    plot_label_distribution(df)
    plot_rating_distribution_by_label(df)
    word_counts = plot_review_length_by_label(df)
    reviewer_counts = plot_reviews_per_reviewer(df)
    plot_top_words(df, "genuine", "05a_top_words_genuine.png")
    plot_top_words(df, "fake", "05b_top_words_fake.png")
    verified_rate = plot_verified_purchase_rate(df)

    print("=" * 60)
    print("EDA SUMMARY (paste into your report)")
    print("=" * 60)
    labeled = df.dropna(subset=["label"])
    print(f"Label balance: {labeled['label'].value_counts(normalize=True).to_dict()}")
    print(f"Avg word count - genuine: "
          f"{word_counts[labeled['label']=='genuine'].mean():.1f}, "
          f"fake: {word_counts[labeled['label']=='fake'].mean():.1f}")
    print(f"Reviews per reviewer - median: {reviewer_counts.median():.0f}, "
          f"max: {reviewer_counts.max()}, "
          f"reviewers with >5 reviews: {(reviewer_counts > 5).sum()}")
    print(f"Verified purchase rate by label:\n{verified_rate}")
    print(f"\nFigures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
