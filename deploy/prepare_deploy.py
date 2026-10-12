"""
prepare_deploy.py

Collects the few small files the deployed dashboard needs into deploy_assets/,
because the deployed app can only use files that are in the GitHub repo
(your big raw/processed data stay out of it).

    deploy_assets/explainable_lr_ott.joblib   the text model
    deploy_assets/amazon_trust_scores.csv     reviewer trust scores
    deploy_assets/amazon_reviews_slim.csv.gz  reviewer history (text shortened)
    deploy_assets/model_comparison_ott.csv    results table
    deploy_assets/figures/*.png               report figures

Run (project root):
    python deploy/prepare_deploy.py
    python deploy/prepare_deploy.py --no-text        # leave review text out entirely
    python deploy/prepare_deploy.py --max-chars 200  # shorter text snippets

About review text: the Amazon data is published for research. Check the dataset page's
terms before putting review text in a PUBLIC repo; --no-text is the safe option.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ["07_roc_curves_ott.png", "08_pr_curves_ott.png", "09_confusion_matrix_best_ott.png",
           "10_trust_score_distribution.png", "11_trust_score_agreement.png",
           "12_top_words_ott.png"]
REVIEW_COLS = ["reviewer_id", "product_id", "rating", "date", "verified_purchase", "review_text"]


def build_assets(root: Path = ROOT, max_chars: int = 300, include_text: bool = True) -> list:
    sources = {
        "model": (root / "models" / "explainable_lr_ott.joblib",
                  "python src/train_classical.py --source ott"),
        "trust": (root / "data" / "processed" / "amazon_trust_scores.csv",
                  "python src/trust_score.py"),
        "reviews": (root / "data" / "processed" / "amazon_review_features.csv",
                    "python src/build_features.py --source amazon --path data/raw/All_Beauty.jsonl"),
    }
    for path, command in sources.values():
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Create it with:  {command}")

    out = root / "deploy_assets"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    written = []

    shutil.copy(sources["model"][0], out / "explainable_lr_ott.joblib")
    written.append(out / "explainable_lr_ott.joblib")
    shutil.copy(sources["trust"][0], out / "amazon_trust_scores.csv")
    written.append(out / "amazon_trust_scores.csv")

    reviews = pd.read_csv(sources["reviews"][0], usecols=lambda c: c in REVIEW_COLS)
    if include_text and "review_text" in reviews.columns:
        reviews["review_text"] = (reviews["review_text"].astype(str)
                                  .str.replace(r"\s+", " ", regex=True).str.slice(0, max_chars))
    else:
        reviews = reviews.drop(columns=["review_text"], errors="ignore")
    slim = out / "amazon_reviews_slim.csv.gz"
    reviews.to_csv(slim, index=False)
    written.append(slim)

    comparison = root / "reports" / "model_comparison_ott.csv"
    if comparison.exists():
        shutil.copy(comparison, out / "model_comparison_ott.csv")
        written.append(out / "model_comparison_ott.csv")
    for name in FIGURES:
        src = root / "reports" / "figures" / name
        if src.exists():
            shutil.copy(src, out / "figures" / name)
            written.append(out / "figures" / name)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=300, help="characters of review text kept")
    ap.add_argument("--no-text", action="store_true", help="do not include review text at all")
    args = ap.parse_args()

    files = build_assets(ROOT, args.max_chars, include_text=not args.no_text)
    total = 0
    print("deploy_assets/ now contains:")
    for f in files:
        size = f.stat().st_size
        total += size
        print(f"  {str(f.relative_to(ROOT)):<52} {size / 1024:>9,.0f} KB")
    print(f"\nTotal: {total / 1024 / 1024:.1f} MB "
          f"({'review text included, shortened' if not args.no_text else 'no review text'})")
    biggest = max(f.stat().st_size for f in files) / 1024 / 1024
    if biggest > 50:
        print(f"WARNING: one file is {biggest:.0f} MB; GitHub rejects files over 100 MB.")


if __name__ == "__main__":
    main()
