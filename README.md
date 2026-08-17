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
   layer that never blindly renames the user's columns.
2. **Preprocessing** — duplicate removal, date parsing, defensive missing-value imputation
   (median for numeric, mode for categorical), reported before/after.
3. **Feature engineering** — derived features (sentiment polarity, hashtag count/band, follower
   band, content-length band, day-of-week, weekend flag, account age, follower/following ratio)
   plus an explicit **pre-publication vs. post-publication** classification table to prevent
   target leakage. This step is **disk-cached** (see Section 13) so the relatively slow
   per-caption sentiment scoring only runs once per unique dataset.
4. **Target** — `high_engagement = 1` if `engagement_rate` is above the 75th percentile of the
   **training data only**.
5. **Modeling** — four independently-trained classifiers (Logistic Regression, Random Forest,
   XGBoost, SVM) inside `sklearn` pipelines with a shared preprocessing `ColumnTransformer`,
   compared with an 80:20 stratified train/validation split. By default each model trains **once**
   with an already-good fixed hyperparameter set (fast); optional `GridSearchCV` tuning is
   available as an explicit, opt-in "slower" toggle in the sidebar (see Section 13).
6. **Evaluation & threshold selection** — Accuracy, Precision, Recall, F1, ROC-AUC, confusion
   matrices; best model chosen by F1 (not raw accuracy, since the target is imbalanced); decision
   threshold tuned on the validation split.
7. **Prediction** — the winning model is retrained on the full 15,000-row training set and applied
   once to the 5,000-row unseen prediction set.
8. **Explainability** — SHAP `TreeExplainer` for tree-based winners (Random Forest / XGBoost);
   standardized coefficients for Logistic Regression, since SHAP's model-agnostic explainer is
   comparatively slow for linear/kernel models.
9. **Content optimization** — the final model scores a grid of controllable-variable combinations
   (category, device, sentiment, media use, hashtag band, content-length band, day of week) while
   holding non-controllable variables at representative values.
10. **Managerial recommendations** — Finding / Evidence / Interpretation / Business Implication /
    Caveat cards generated from the actual calculated statistics, never hardcoded.

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
4. See **Section 13** below if the app is slow to start or gets its CPU allocation throttled —
   the defaults already applied in this repo (untuned fast model fits, a capped worker count,
   and disk-cached feature engineering) are specifically tuned to avoid that on the free tier.

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
│   ├── __init__.py
│   └── core.py                # ALL analytical logic (Layers 1-9), consolidated into one module —
│                               # see the module docstring at the top of core.py for what used to
│                               # be separate files (data_loader, preprocessing, eda, nlp_analysis,
│                               # feature_engineering, modeling, evaluation, explainability,
│                               # prediction, optimization, recommendations) and why it was merged
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

## 13. Performance & Deployment Notes (Streamlit Cloud CPU throttling fix)

An earlier version of this app was slow to start and had its CPU allocation throttled by Streamlit
Community Cloud. Two concrete, mechanical causes were found and fixed — not just "wait longer":

1. **CPU oversubscription from `n_jobs=-1`.** Hosted free-tier containers grant a small CPU
   *quota* (often less than 1 full vCPU), but Python's `os.cpu_count()` reports the **host
   machine's** core count, not that quota. The old code set `n_jobs=-1` on scikit-learn/XGBoost
   estimators **and** on the surrounding `GridSearchCV` at the same time — nested parallelism that
   spawns far more worker processes/threads than the container is actually allowed to run at once.
   The OS then repeatedly pauses ("throttles") those workers, which makes the *wall-clock* time
   **worse**, not better, than just using 1-2 workers. The app now uses a single `SAFE_N_JOBS`
   constant (`src/core.py`, default **2**, overridable via the `APP_MAX_WORKERS` environment
   variable) and only ever parallelizes in one place at a time.
2. **Hyperparameter search ran automatically, every time.** The old app always ran `GridSearchCV`
   (multiple folds x multiple hyperparameter combinations, x 4 models) before the very first page
   render. Tuning is now **off by default** — every model trains **once** with an already-good
   fixed hyperparameter set — and is only ever run as an explicit, clearly-labelled "slower"
   opt-in from the sidebar's **⚙️ Performance settings** expander.
3. **Slow per-caption sentiment scoring reran on every cold start.** `engineer_features()` (which
   calls TextBlob once per caption, over ~20,000 rows) is now wrapped with a `joblib.Memory` disk
   cache, so that work only happens once per unique dataset content — including across app
   restarts / the container "waking up" from Streamlit Cloud's sleep-on-inactivity — instead of
   once per user session.
4. **Models train one at a time, with visible progress**, instead of behind one long silent
   spinner (`app.py`'s `_train_models_with_progress`), and **each model is cached individually**
   so changing the model selection only (re)trains the model(s) that actually changed.

If the app is still slow on a very constrained host, lower `APP_MAX_WORKERS` to `1` (via the
Streamlit Cloud app's "Secrets"/environment settings) and keep hyperparameter tuning off.

---

*Built with Generative AI assistance (code generation and review) as part of an MBA WAI
(Working with AI) project. All analytical results in the app are computed live from the uploaded
data — none are hardcoded or fabricated.*
