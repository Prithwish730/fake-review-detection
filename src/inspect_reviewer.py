"""
inspect_reviewer.py

Print the actual reviews written by specific reviewers, next to their trust
score. Use it to hand-check what the score flagged, and to pick real examples
for your report.

Examples:
    python src/inspect_reviewer.py --lowest 5
    python src/inspect_reviewer.py AEOX6BMEBEXAQA73AOCR27WRRN6Q
    python src/inspect_reviewer.py --lowest 3 --max-reviews 10 --width 200
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# review text often contains emoji/non-ASCII; don't let the Windows console crash on it
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reviewer_ids", nargs="*", help="one or more reviewer IDs")
    ap.add_argument("--lowest", type=int, default=0, help="show the N least-trusted reviewers")
    ap.add_argument("--source", default="amazon")
    ap.add_argument("--max-reviews", type=int, default=8)
    ap.add_argument("--width", type=int, default=160, help="characters of text to show")
    args = ap.parse_args()

    scores = pd.read_csv(PROCESSED / f"{args.source}_trust_scores.csv")
    reviews = pd.read_csv(PROCESSED / f"{args.source}_review_features.csv",
                          usecols=["reviewer_id", "product_id", "rating", "date",
                                   "verified_purchase", "review_text"])
    ids = list(args.reviewer_ids)
    if args.lowest:
        ids += scores.sort_values("trust_score")["reviewer_id"].head(args.lowest).tolist()
    if not ids:
        ap.error("give at least one reviewer ID, or use --lowest N")

    for rid in ids:
        row = scores[scores["reviewer_id"] == rid]
        if row.empty:
            print(f"\n{rid}: not found in trust scores\n")
            continue
        r = row.iloc[0]
        print("=" * 100)
        print(f"{rid}   trust score: {r['trust_score']}   "
              f"(heuristic {r['heuristic_trust']}, isolation forest {r['learned_trust']})")
        print(f"reviews: {int(r['review_count'])} | 5-star share: {r['pct_5star']:.2f} | "
              f"max reviews in one day: {int(r['max_reviews_in_one_day'])} | "
              f"text similarity (any product): {r['avg_self_similarity']:.2f} | "
              + (f"(different products): {r['avg_cross_product_similarity']:.2f} | "
                 if "avg_cross_product_similarity" in r else "")
              + f"verified: {r['verified_ratio']:.2f}")
        mine = reviews[reviews["reviewer_id"] == rid].sort_values("date")
        for _, rv in mine.head(args.max_reviews).iterrows():
            text = " ".join(str(rv["review_text"]).split())[: args.width]
            print(f"  {str(rv['date'])[:10]} | product {rv['product_id']} | "
                  f"{rv['rating']:.0f} stars | verified={rv['verified_purchase']} | {text}")
        if len(mine) > args.max_reviews:
            print(f"  ... and {len(mine) - args.max_reviews} more reviews")
    print()


if __name__ == "__main__":
    main()
