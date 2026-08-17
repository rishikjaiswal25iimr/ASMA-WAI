# AI-Powered Social Media Content Optimization

### Predicting Engagement and Identifying the Drivers of High-Performing Content

**WAI Project — AI & Social Media Analytics — IIM Ranchi**

---

## 1. Business Problem

Marketers invest heavily in social-media content but often lack a systematic, data-driven way to
decide which combinations of content characteristics, sentiment, timing, and audience factors are
most likely to drive high engagement.

## 2. Objective

Build an AI-powered decision-support system that:
1. diagnoses historical engagement patterns,
2. identifies the characteristics of high-performing content,
3. uses machine learning to predict whether a post will achieve high engagement,
4. explains which variables drive that prediction (SHAP / explainable AI),
5. applies the trained model to unseen posts, and
6. converts the outputs into an actionable content-optimization playbook.

## 3. Dataset Description

Source file: `synthetic_social_media_engagement.csv` (20,000 synthetic post-level records).

Columns actually present in this dataset: `post_id, user_id, user_name, user_gender, user_age,
followers_count, following_count, account_creation_date, is_verified, location, topic,
post_content, content_length, hashtags, has_media, post_date, device, language, likes, comments,
shares, engagement_rate`.

**Notable dataset-specific characteristics (auto-detected, not assumed):**
- There is **no true multi-network "platform" field** (e.g., Instagram vs. X/Twitter). `device`
  (Android / iPhone / Web / Tablet) is the closest available channel-level dimension and is used
  in its place, with this substitution called out explicitly throughout the app.
- `post_date` has **no time-of-day component**, so posting-hour analysis is not possible; the
  project instead analyzes day-of-week and weekend/weekday patterns.
- There is **no pre-existing sentiment label** — `post_content` contains raw caption text, so
  genuine text-based sentiment analysis (TextBlob polarity/subjectivity) is performed instead of
  reusing a label that doesn't exist.
- `topic` (16 categories, e.g. Technology, Travel, Fashion, Gaming) is used as the content
  **category** dimension.

## 4. Dataset Split

| File | Records | Purpose |
|---|---|---|
| `data/training_data.csv` | First 15,000 rows | Model development: EDA, feature engineering, target definition, training, internal validation, hyperparameter tuning, threshold selection |
| `data/prediction_data.csv` | Last 5,000 rows | Completely unseen: final prediction / out-of-sample evaluation only |

The prediction set is **never** used for feature selection, model training, model comparison,
threshold selection, SHAP analysis, or optimization-rule creation. The high-engagement threshold
(75th percentile of `engagement_rate`) is computed **only** on the training data and then applied,
unchanged, to the prediction data.

## 5. Methodology

1. **Data discovery & quality** — automatic column-type detection and an alias-based column-mapping
   layer (`src/data_loader.py`) that never blindly renames the user's columns.
2. **Preprocessing** — duplicate removal, date parsing, defensive missing-value imputation
   (median for numeric, mode for categorical), reported before/after (`src/preprocessing.py`).
3. **Feature engineering** — derived features (sentiment polarity, hashtag count/band, follower
   band, content-length band, day-of-week, weekend flag, account age, follower/following ratio)
   plus an explicit **pre-publication vs. post-publication** classification table to prevent
   target leakage (`src/feature_engineering.py`).
4. **Target** — `high_engagement = 1` if `engagement_rate` is above the 75th percentile of the
   **training data only**.
5. **Modeling** — four independently-trained classifiers (Logistic Regression, Random Forest,
   XGBoost, SVM) inside `sklearn` pipelines with a shared preprocessing `ColumnTransformer`,
   compared with an 80:20 stratified train/validation split and light `GridSearchCV` tuning
   (`src/modeling.py`).
6. **Evaluation & threshold selection** — Accuracy, Precision, Recall, F1, ROC-AUC, confusion
   matrices; best model chosen by F1 (not raw accuracy, since the target is imbalanced); decision
   threshold tuned on the validation split (`src/evaluation.py`).
7. **Prediction** — the winning model is retrained on the full 15,000-row training set and applied
   once to the 5,000-row unseen prediction set (`src/prediction.py`).
8. **Explainability** — SHAP `TreeExplainer` for tree-based winners (Random Forest / XGBoost);
   standardized coefficients for Logistic Regression, since SHAP's model-agnostic explainer is
   comparatively slow for linear/kernel models (`src/explainability.py`).
9. **Content optimization** — the final model scores a grid of controllable-variable combinations
   (category, device, sentiment, media use, hashtag band, content-length band, day of week) while
   holding non-controllable variables at representative values (`src/optimization.py`).
10. **Managerial recommendations** — Finding / Evidence / Interpretation / Business Implication /
    Caveat cards generated from the actual calculated statistics, never hardcoded
    (`src/recommendations.py`).

## 6. AI/ML Models

| Model | Role |
|---|---|
| Logistic Regression | Interpretable linear baseline |
| Random Forest | Nonlinear relationships, feature interactions |
| XGBoost | Gradient boosting, typically the strongest raw predictive performance |
| SVM (RBF) | Classical margin-based classifier; trained on a stratified subsample for runtime |

## 7. Dashboard Functionality (10 pages)

1. Executive Overview — KPIs, dataset snapshot, auto-detected limitations
2. Data Quality — dimensions, missingness, duplicates, column mapping
3. Descriptive Analytics — category/device/timing/audience/engagement, with filters
4. NLP / Sentiment — TextBlob sentiment distribution and cross-tabs, top vocabulary
5. Feature Engineering — target construction, engineered features, leakage-check table
6. Model Training & Comparison — metrics table, ROC curves, confusion matrices, threshold table
7. Explainable AI — SHAP global importance + individual prediction explanations
8. Prediction — scored unseen 5,000-record set, out-of-sample metrics, downloadable CSV
9. Content Optimization — model-scored controllable-variable scenario playbook
10. Managerial Recommendations — insight cards + short/long-term actions + ethics/limitations

## 8. How to Run Locally

```bash
git clone <this-repo-url>
cd social-media-content-optimization
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app defaults to the bundled `data/training_data.csv` and `data/prediction_data.csv`; you can
also upload your own files (CSV or Excel) from the sidebar.

## 9. How to Deploy on Streamlit Community Cloud

1. Push this folder to a public (or Streamlit-Cloud-connected private) GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), click **New app**, select the repo,
   branch, and `app.py` as the entry point.
3. Streamlit Cloud installs `requirements.txt` automatically and launches the app.

## 10. File Structure

```text
social-media-content-optimization/
├── app.py                     # Streamlit application (10-page dashboard)
├── requirements.txt
├── README.md
├── data/
│   ├── training_data.csv      # first 15,000 records
│   └── prediction_data.csv    # last 5,000 records (unseen)
├── src/
│   ├── data_loader.py         # Layer 1 — discovery, column mapping, quality report
│   ├── preprocessing.py       # cleaning, missing-value treatment
│   ├── eda.py                 # Layer 2 — descriptive analytics / charts
│   ├── nlp_analysis.py        # Layer 3 — sentiment (TextBlob), n-grams
│   ├── feature_engineering.py # Layer 4 — derived features, target, leakage table
│   ├── modeling.py            # Layer 5 — 4-model training pipeline
│   ├── evaluation.py          # comparison metrics, ROC, confusion matrix, threshold
│   ├── explainability.py      # Layer 7 — SHAP / coefficient explanations
│   ├── prediction.py          # Layer 6 — apply final model to unseen data
│   ├── optimization.py        # Layer 8 — content optimization scenario search
│   └── recommendations.py     # Layer 9 — managerial insight-card generation
├── tests/
│   └── test_pipeline.py       # end-to-end smoke test of the full analytical pipeline
├── outputs/                   # figures/tables/predictions written at runtime (gitignored contents)
└── models/saved_models/       # optional location for persisted trained models
```

## 11. Limitations

- **Synthetic data:** results describe patterns in this synthetic dataset and are for coursework
  demonstration; they should not be treated as real-world marketing fact without validation.
- **No true platform field:** cross-network platform comparisons (Instagram vs. X, etc.) are not
  possible with this data; `device` is used as the closest available substitute and labelled as such.
- **No time-of-day field:** posting-hour optimization is not possible; only day-of-week/weekend
  patterns are analyzed.
- **Association, not causation:** all findings are reported as "associated with" / "predictive of,"
  never as proven causal effects.
- **Class imbalance:** the high-engagement class is ~25% of posts; F1/ROC-AUC are prioritized over
  raw accuracy for model selection.

## 12. Ethical Considerations

See the in-app **Managerial Recommendations → Ethical AI & limitations** section for the full,
auto-generated discussion of dataset, model, and AI limitations, privacy, and bias considerations.

---

*Built with Generative AI assistance (code generation and review) as part of an MBA WAI
(Working with AI) project. All analytical results in the app are computed live from the uploaded
data — none are hardcoded or fabricated.*
