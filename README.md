# Fake Review Detection & Reviewer Trust Score

**Live demo:** [Open the app](https://fake-review-detector-app.streamlit.app/)

Final-year project — ML/NLP system to classify fake reviews and score reviewer trustworthiness.

## Status: classifier + trust score + explainability + dashboard + deployment

## Project structure
```
fake-review-detection/
├── data/
│   ├── raw/              # raw datasets (sample data lives here)
│   └── processed/        # cleaned/feature-engineered data (Week 2+)
├── src/
│   ├── generate_sample_data.py  # synthetic data for pipeline testing
│   ├── data_loader.py           # standardized data loading (SWAP REAL DATA HERE)
│   ├── eda.py                   # exploratory data analysis
│   ├── preprocess.py            # text cleaning
│   ├── features.py              # linguistic + behavioral feature functions
│   ├── build_features.py        # runs the full feature pipeline
│   ├── train_classical.py       # trains + compares 7 classical models
│   ├── trust_score.py           # reviewer trust score (heuristic + Isolation Forest)
│   ├── inspect_reviewer.py      # print a flagged reviewer's actual reviews
│   └── explain.py               # exact per-word explanations for the text model
├── tests/                       # unit tests (pytest)
├── notebooks/             # exploratory notebooks (optional, mirrors src/)
├── models/                 # saved trained models (Week 3+)
├── reports/
│   └── figures/            # EDA plots, auto-generated
├── app/app.py               # Streamlit dashboard
├── deploy/                  # pin_requirements.py, prepare_deploy.py (see Deployment)
└── deploy_assets/           # small files the deployed app needs (created by prepare_deploy.py)
```

## Setup
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run Week 1 pipeline
```bash
python src/generate_sample_data.py   # creates data/raw/reviews_sample.csv
python src/eda.py                     # creates reports/figures/*.png + prints summary
```


## Run Week 2 pipeline
```bash
python src/build_features.py     # creates data/processed/review_features.csv + reviewer_features.csv
python tests/test_features.py    # or: pytest tests/ -v
```

Optional (better text cleaning + sentiment) - run once in Python:
```python
import nltk
nltk.download("wordnet"); nltk.download("omw-1.4"); nltk.download("vader_lexicon")
```
The code falls back automatically if these aren't downloaded.

## Run Week 3 (model training)
```bash
python src/train_classical.py     # trains 7 models, writes reports/model_comparison.csv
python tests/test_models.py       # or: pytest tests/ -v
mlflow ui --backend-store-uri sqlite:///mlflow.db   # optional: browse experiments at http://127.0.0.1:5000
```
Outputs: `reports/model_comparison.csv`, ROC / PR / confusion-matrix figures, and
`models/best_classifier.joblib` (best model chosen by cross-validated F1 on the train set).

> Note: on the synthetic sample data every model scores ~1.0 because the fake reviews are
> deliberately easy to spot. Use real datasets (Ott et al.) for the numbers in your report.

## Using REAL data

Two datasets, two jobs:

| Dataset | Has | Lacks | Feeds |
|---|---|---|---|
| Ott et al. corpus (`deceptive-opinion.csv` or `op_spam_v1.4/`) | text + fake/genuine label | reviewer info, dates, ratings | the **classifier** |
| Amazon Reviews (`.jsonl` / `.csv`) | reviewer id, rating, date, verified flag | labels | the **trust score** |

```bash
# 1) classifier on Ott
python src/eda.py            --source ott --path data/raw/deceptive-opinion.csv
python src/build_features.py --source ott --path data/raw/deceptive-opinion.csv
python src/train_classical.py --source ott

# 2) behavioral features on Amazon
python src/build_features.py --source amazon --path data/raw/All_Beauty.jsonl --min-reviews 3
```
Outputs are prefixed by source (`ott_review_features.csv`, `amazon_reviewer_features.csv`,
`model_comparison_ott.csv`, ...), so datasets never overwrite each other.
Omit `--path` to auto-detect files placed in `data/raw/`.

Amazon options: `--nrows` (read first N rows of huge files), `--min-reviews` (default 3),
`--max-reviewers` (default 5000).

## Reviewer Trust Score
```bash
python src/build_features.py --source amazon --path data/raw/All_Beauty.jsonl --max-reviewers 10000
python src/trust_score.py
```
Two independent scores, averaged into `trust_score` (0-100, higher = more trustworthy, **relative to the
other reviewers in the dataset**):
1. **Heuristic**: weighted suspiciousness signals (template reuse, burst posting, rating deviation,
   5-star share, unverified purchases, velocity). Interpretable; weights in `DEFAULT_WEIGHTS`.
2. **Isolation Forest**: unsupervised anomaly detection on the same behavioral features.

Validation without labels: agreement between the two methods, plus a planted-spammer test.
Outputs: `data/processed/amazon_trust_scores.csv`, figures 10 and 11.

Data-quality notes (found by hand-inspecting the least-trusted reviewers):
- Exact duplicate reviews (same reviewer + product + text) are dropped by default; they are
  usually variant/double-submission artifacts. Use `--keep-duplicates` to disable.
- Template reuse is measured across *different products* only (`avg_cross_product_similarity`),
  because the same text pasted across products is the real spam signal.
- Inspect flagged reviewers: `python src/inspect_reviewer.py --lowest 10`

## Explainability + dashboard
```bash
python src/train_classical.py --source ott   # also saves models/explainable_lr_ott.joblib
python src/explain.py --source ott           # top fake/genuine words -> reports/top_words_ott.csv + figure 12
streamlit run app/app.py                     # opens the dashboard in your browser
```
Dashboard tabs: **Check a review** (probability + the words that drove it), **Reviewer trust lookup**,
**Flagged reviewers** (filter + CSV download), **Model results** (comparison table, curves, limitations).

Explanations use the TF-IDF Logistic Regression: for a linear model, `weight x TF-IDF value` per word is
an exact additive breakdown of the prediction (the same decomposition SHAP's linear explainer approximates).

## Deployment (Streamlit Community Cloud)
The server can only use files that are in the GitHub repo, so the repo carries small copies in `deploy_assets/`
and your big raw/processed data stay out (see `.gitignore`).
```bash
python deploy/pin_requirements.py      # slim requirements.txt, pinned to YOUR versions (old list -> requirements-train.txt)
python deploy/prepare_deploy.py        # builds deploy_assets/  (add --no-text to leave review text out)
git add .
git commit -m "Prepare deployment"
git push
```
Then at share.streamlit.io: sign in with GitHub, create a new app, pick this repo, branch `main`, main file
`app/app.py`, open **Advanced settings** and choose the **same Python version as your laptop**
(`python --version`), then Deploy.

Local development now uses `pip install -r requirements-train.txt` (the full list incl. torch, xgboost, mlflow).

## Week 1 EDA findings (from sample data — replace with real numbers once real data is loaded)
- Fake reviews cluster heavily at 5 stars (rating distribution mismatch vs. genuine)
- Fake reviews have a lower verified-purchase rate (~31% vs ~81% genuine)
- Fake reviews tend to be shorter and more repetitive/templated in wording
- A subset of reviewers post well above the median review count — a burst/frequency signal
  worth encoding as a behavioral feature for the trust score

## Roadmap
See `fake_review_detection_roadmap.md` for the full phase-by-phase plan and the
compressed 6-week (1.5 month) schedule.

- [x] Week 1 — Setup, data loading, EDA
- [x] Week 2 — Preprocessing + feature engineering (linguistic + behavioral)
- [x] Week 3 — Classical ML models (Logistic Regression, Naive Bayes, XGBoost)
- [x] Heuristic + Isolation Forest trust score (done early)
- [ ] Week 4 — BERT fine-tuning (Colab)
- [ ] Week 5 — Learned trust score + SHAP + Streamlit dashboard
- [ ] Week 6 — Deployment + report + slides
