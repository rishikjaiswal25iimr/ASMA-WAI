"""
core.py
=======
AI-Powered Social Media Content Optimization — consolidated core logic.

This single module replaces the previous multi-file `src/` package
(data_loader.py, preprocessing.py, eda.py, nlp_analysis.py,
feature_engineering.py, modeling.py, evaluation.py, explainability.py,
prediction.py, optimization.py, recommendations.py). All analytical logic
is unchanged; the functions and their signatures are the same so the rest
of the app / tests barely had to change. Sections are separated by banner
comments below, in the same order as the original layer numbering used
throughout the project (Layer 1 -> Layer 9).

--------------------------------------------------------------------------
WHY THIS FILE ALSO FIXES THE STREAMLIT-CLOUD "STUCK AT STARTUP / CPU
THROTTLED" PROBLEM
--------------------------------------------------------------------------
Two real, mechanical causes were found and fixed here (not just "load
things slower" band-aids):

1. **CPU oversubscription (`n_jobs=-1` everywhere).**
   `os.cpu_count()` inside a container reports the HOST's core count, not
   the small CPU *quota* Streamlit Community Cloud actually grants the
   container (cgroup-limited, often <=1 vCPU). scikit-learn/XGBoost's
   `n_jobs=-1` spawns a worker per host core. On a quota-limited box this
   causes constant CFS-quota throttling: many more threads/processes are
   created than the container is allowed to run at once, so the OS
   scheduler keeps pausing them, and the actual wall-clock time gets much
   *worse* than using 1-2 workers, not better. On top of that, the old
   code used `n_jobs=-1` on the estimator AND `n_jobs=-1` on the
   surrounding `GridSearchCV` at the same time — a classic *nested*
   parallelism bug that multiplies the number of processes spawned.
   Fix: a single `SAFE_N_JOBS` constant (default 2, overridable via the
   `APP_MAX_WORKERS` environment variable) is used everywhere, and
   parallelism is only ever applied in ONE place at a time (never both the
   estimator and the search wrapper simultaneously).

2. **Hyperparameter search was always on.**
   The previous code always ran `GridSearchCV` (multiple folds x multiple
   hyperparameter combinations) for all 4 models on every fresh app
   session. That is a lot of model fits before a single pixel is drawn.
   Fix: `tune=False` is now the default. Each model trains ONCE with a
   sensible, already-good fixed hyperparameter set. Full grid-search
   tuning is still available (`tune=True`) as an explicit, opt-in,
   clearly-labelled "slower" option in the sidebar.

3. **Expensive per-row TextBlob sentiment scoring reran on every cold
   start.** `engineer_features()` is now wrapped with a `joblib.Memory`
   disk cache (see `_MEMORY` below) so the (comparatively slow, pure
   Python) sentiment scoring over ~20,000 captions only ever runs once
   per unique dataset content — including across app restarts / Streamlit
   Cloud "waking up" from sleep — instead of once per session.

4. **Models are trained ONE AT A TIME with visible progress** via
   `train_models_sequential()`, a generator the Streamlit app consumes to
   update a progress bar/status line as each model finishes, and each
   model is cached independently (see app.py) so re-running after only
   adding one more model to the selection does not retrain the others.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
import time
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    precision_recall_curve, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except Exception:
    _TEXTBLOB_AVAILABLE = False

try:
    import shap
    _SHAP_AVAILABLE = True
except Exception:
    _SHAP_AVAILABLE = False

try:
    from joblib import Memory
    _cache_dir = os.environ.get("APP_CACHE_DIR", os.path.join(tempfile.gettempdir(), "content_opt_cache"))
    os.makedirs(_cache_dir, exist_ok=True)
    _MEMORY = Memory(location=_cache_dir, verbose=0)
except Exception:
    # If the filesystem is read-only or unavailable for any reason, degrade
    # gracefully to "no disk cache" rather than crashing the app.
    class _NoOpMemory:
        def cache(self, func=None, **kwargs):
            if func is None:
                return lambda f: f
            return func

    _MEMORY = _NoOpMemory()

RANDOM_SEED = 42

# Safe worker count for a CPU-quota-limited cloud container. Overridable via
# the APP_MAX_WORKERS environment variable if you know your host's actual
# quota (e.g. set it to 1 on the smallest Streamlit Community Cloud tier).
SAFE_N_JOBS = max(1, int(os.environ.get("APP_MAX_WORKERS", "2")))


# ==========================================================================
# LAYER 1 — DATA DISCOVERY  (was data_loader.py)
# ==========================================================================

CANONICAL_ALIASES = {
    "post_id": ["post_id", "id", "postid"],
    "user_id": ["user_id", "userid", "author_id"],
    "user_name": ["user_name", "username", "handle"],
    "gender": ["user_gender", "gender"],
    "age": ["user_age", "age"],
    "followers_count": ["followers_count", "follower_count", "followers", "n_followers"],
    "following_count": ["following_count", "follower_following", "following"],
    "account_creation_date": ["account_creation_date", "account_created", "signup_date", "join_date"],
    "is_verified": ["is_verified", "verified", "verified_status"],
    "location": ["location", "country", "city"],
    "category": ["topic", "category", "content_category", "niche"],
    "post_content": ["post_content", "caption", "text", "content", "post_text"],
    "content_length": ["content_length", "caption_length", "text_length"],
    "hashtags": ["hashtags", "hashtag", "tags"],
    "has_media": ["has_media", "media_present", "is_media"],
    "post_date": ["post_date", "timestamp", "post_time", "datetime", "date", "published_at"],
    "device": ["device", "platform", "social_platform", "channel"],
    "language": ["language", "lang"],
    "likes": ["likes", "like_count", "n_likes"],
    "comments": ["comments", "comment_count", "n_comments"],
    "shares": ["shares", "share_count", "n_shares", "retweets"],
    "views": ["views", "view_count", "impressions"],
    "saves": ["saves", "save_count", "bookmarks"],
    "engagement_rate": ["engagement_rate", "engagement", "eng_rate"],
    "sentiment_label": ["sentiment", "sentiment_label"],
}

POST_PUBLICATION_ROLES = {"likes", "comments", "shares", "views", "saves", "engagement_rate"}


def load_dataset(uploaded_file) -> pd.DataFrame:
    """Load a CSV or Excel file (path string or file-like object) into a DataFrame."""
    if isinstance(uploaded_file, (str,)):
        name = uploaded_file
    else:
        name = getattr(uploaded_file, "name", "uploaded_file.csv")

    if name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    return df


def build_column_mapping(df: pd.DataFrame) -> dict:
    """Map canonical analytical roles -> actual column name in df (or None)."""
    lower_cols = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for role, aliases in CANONICAL_ALIASES.items():
        found = None
        for alias in aliases:
            if alias in lower_cols:
                found = lower_cols[alias]
                break
        if found is None:
            for lc, orig in lower_cols.items():
                if any(alias in lc for alias in aliases):
                    found = orig
                    break
        mapping[role] = found
    return mapping


def classify_columns(df: pd.DataFrame) -> dict:
    """Classify every column into numeric / categorical / datetime / text / boolean."""
    numeric, categorical, datetime_cols, text, boolean = [], [], [], [], []

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            boolean.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
        elif _looks_like_date(s):
            datetime_cols.append(col)
        else:
            avg_len = s.astype(str).str.len().mean()
            n_unique = s.nunique()
            if avg_len > 40 and n_unique > 0.5 * len(s):
                text.append(col)
            else:
                categorical.append(col)

    return {
        "numeric": numeric, "categorical": categorical, "datetime": datetime_cols,
        "text": text, "boolean": boolean,
    }


def _looks_like_date(series: pd.Series, sample: int = 50) -> bool:
    sample_vals = series.dropna().astype(str).head(sample)
    if sample_vals.empty:
        return False
    try:
        parsed = pd.to_datetime(sample_vals, errors="coerce", format="mixed")
        return parsed.notna().mean() > 0.8
    except Exception:
        return False


def data_quality_report(df: pd.DataFrame) -> dict:
    """Return a dictionary summarizing dataset dimensions, missingness, duplicates, dtypes."""
    col_types = classify_columns(df)
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    report = {
        "n_records": len(df),
        "n_variables": df.shape[1],
        "n_numeric": len(col_types["numeric"]),
        "n_categorical": len(col_types["categorical"]),
        "n_text": len(col_types["text"]),
        "n_datetime": len(col_types["datetime"]),
        "n_boolean": len(col_types["boolean"]),
        "n_duplicates": int(df.duplicated().sum()),
        "missing_table": pd.DataFrame({
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "missing_count": missing.values,
            "missing_pct": missing_pct.values,
            "n_unique": [df[c].nunique() for c in df.columns],
        }),
        "column_types": col_types,
    }
    return report


# ==========================================================================
# LAYER 1/4 support — CLEANING  (was preprocessing.py)
# ==========================================================================

def clean_dataset(df: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, dict]:
    """Duplicate removal, date parsing, defensive missing-value imputation."""
    report = {"steps": []}
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        report["steps"].append(f"Removed {before - after} exact duplicate rows.")

    for role in ("post_date", "account_creation_date"):
        col = mapping.get(role)
        if col and col in df.columns:
            missing_before = df[col].isna().sum()
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
            missing_after = df[col].isna().sum()
            if missing_after > missing_before:
                report["steps"].append(
                    f"Column '{col}' ({role}): {missing_after - missing_before} values could not be "
                    f"parsed as dates and were set to missing."
                )

    missing_report_rows = []
    for col in df.columns:
        n_missing_before = df[col].isna().sum()
        if n_missing_before == 0:
            continue
        pct_missing = n_missing_before / len(df) * 100

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)
            treatment = f"median imputation ({fill_value:.3f})"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            treatment = "left as missing (datetime; no safe imputation)"
        else:
            fill_value = df[col].mode(dropna=True)
            fill_value = fill_value.iloc[0] if not fill_value.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)
            treatment = f"mode imputation ('{fill_value}')"

        n_missing_after = df[col].isna().sum()
        missing_report_rows.append({
            "column": col, "missing_before": int(n_missing_before),
            "pct_missing": round(pct_missing, 2), "treatment": treatment,
            "missing_after": int(n_missing_after),
        })

    report["missing_value_treatment"] = pd.DataFrame(missing_report_rows) if missing_report_rows else pd.DataFrame(
        columns=["column", "missing_before", "pct_missing", "treatment", "missing_after"]
    )

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    return df, report


def detect_outliers_iqr(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append({
            "column": col, "lower_bound": round(lower, 3), "upper_bound": round(upper, 3),
            "n_outliers": int(n_outliers), "pct_outliers": round(n_outliers / len(df) * 100, 2),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# LAYER 2 — DESCRIPTIVE / EXPLORATORY ANALYTICS  (was eda.py)
# ==========================================================================

def group_engagement(df: pd.DataFrame, group_col: str, engagement_col: str) -> pd.DataFrame:
    g = (df.groupby(group_col, observed=True)[engagement_col]
         .agg(["mean", "median", "count", "std"])
         .reset_index()
         .sort_values("mean", ascending=False))
    g.columns = [group_col, "avg_engagement", "median_engagement", "n_posts", "std_engagement"]
    return g


def bar_chart(summary_df: pd.DataFrame, x: str, y: str, title: str, color=None):
    fig = px.bar(summary_df, x=x, y=y, title=title, color=color or x, text_auto=".3f")
    fig.update_layout(showlegend=False, xaxis_title=x.replace("_", " ").title(),
                       yaxis_title=y.replace("_", " ").title())
    return fig


def box_plot(df: pd.DataFrame, x: str, y: str, title: str):
    fig = px.box(df, x=x, y=y, title=title, points=False)
    fig.update_layout(xaxis_title=x.replace("_", " ").title(), yaxis_title=y.replace("_", " ").title())
    return fig


def correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str]):
    corr = df[numeric_cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Correlation Heatmap — Numeric Variables")
    return fig, corr


def scatter_plot(df: pd.DataFrame, x: str, y: str, title: str, color=None, trendline=None):
    kwargs = {}
    if trendline:
        kwargs["trendline"] = trendline
    fig = px.scatter(df, x=x, y=y, title=title, color=color, opacity=0.5, **kwargs)
    return fig


def high_vs_low_distribution(df: pd.DataFrame, target_col: str):
    counts = df[target_col].value_counts().rename({0: "Low Engagement", 1: "High Engagement"})
    fig = px.pie(values=counts.values, names=counts.index.astype(str),
                 title="High vs. Low Engagement Distribution (Training Data)", hole=0.4)
    return fig, counts


def weekend_vs_weekday(df: pd.DataFrame, engagement_col: str):
    if "is_weekend" not in df.columns:
        return None, None
    g = df.groupby("is_weekend", observed=True)[engagement_col].mean().rename({True: "Weekend", False: "Weekday"})
    fig = px.bar(x=g.index.astype(str), y=g.values, title="Weekend vs. Weekday Average Engagement",
                 labels={"x": "", "y": "Avg Engagement Rate"}, text_auto=".4f")
    return fig, g


# ==========================================================================
# LAYER 3 — NLP / SENTIMENT ANALYTICS  (was nlp_analysis.py)
# ==========================================================================

STOPWORDS = set("""a an the and or but if while is are was were be been being to of in on for with
as by at from this that these those it its it's he she they them his her their our your you i we
not no nor so than then too very can will just don should now""".split())


def compute_sentiment(text: str) -> tuple[float, float]:
    """Return (polarity, subjectivity) for one string. Polarity in [-1, 1]."""
    if not _TEXTBLOB_AVAILABLE or not isinstance(text, str) or not text.strip():
        return 0.0, 0.0
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def _add_sentiment_columns_uncached(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df = df.copy()
    pols, subs = [], []
    for t in df[text_col].astype(str):
        p, s = compute_sentiment(t)
        pols.append(p)
        subs.append(s)
    df["sentiment_polarity"] = pols
    df["sentiment_subjectivity"] = subs
    df["sentiment_label"] = pd.cut(
        df["sentiment_polarity"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["Negative", "Neutral", "Positive"],
    ).astype(str)
    return df


# Disk-cached: per-row TextBlob scoring over the full 15k/5k datasets is the
# single slowest step in the whole pipeline. Caching it to disk (keyed on
# the actual text content) means it only ever runs once per unique dataset,
# not once per Streamlit session / cold start.
add_sentiment_columns = _MEMORY.cache(_add_sentiment_columns_uncached)


def extract_hashtag_count(df: pd.DataFrame, hashtag_col: str) -> pd.Series:
    def _count(val):
        if not isinstance(val, str) or not val.strip():
            return 0
        tokens = re.split(r"[,\s]+", val.strip())
        return sum(1 for t in tokens if t.startswith("#"))
    return df[hashtag_col].apply(_count)


def top_ngrams(text_series: pd.Series, n: int = 1, top_k: int = 15) -> pd.DataFrame:
    counter = Counter()
    for text in text_series.astype(str):
        tokens = [w.lower() for w in re.findall(r"[a-zA-Z']+", text) if w.lower() not in STOPWORDS and len(w) > 2]
        grams = zip(*[tokens[i:] for i in range(n)])
        for g in grams:
            counter[" ".join(g)] += 1
    top = counter.most_common(top_k)
    return pd.DataFrame(top, columns=["ngram", "frequency"])


def sentiment_vs_dimension(df: pd.DataFrame, sentiment_col: str, dimension_col: str,
                            engagement_col: str) -> pd.DataFrame:
    return (
        df.groupby([dimension_col, sentiment_col], observed=True)[engagement_col]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_engagement", "count": "n_posts"})
    )


# ==========================================================================
# LAYER 4 — FEATURE ENGINEERING + TARGET  (was feature_engineering.py)
# ==========================================================================

def follower_band(followers: pd.Series) -> pd.Series:
    bins = [-1, 1000, 10000, 50000, 500000, np.inf]
    labels = ["Nano (<1K)", "Micro (1K-10K)", "Mid (10K-50K)", "Macro (50K-500K)", "Mega (500K+)"]
    return pd.cut(followers, bins=bins, labels=labels)


def content_length_band(length: pd.Series) -> pd.Series:
    bins = [-1, 75, 150, 250, np.inf]
    labels = ["Short (<75)", "Medium (75-150)", "Long (150-250)", "Very Long (250+)"]
    return pd.cut(length, bins=bins, labels=labels)


def hashtag_band(n: pd.Series) -> pd.Series:
    bins = [-1, 0, 2, 4, np.inf]
    labels = ["None", "Low (1-2)", "Medium (3-4)", "High (5+)"]
    return pd.cut(n, bins=bins, labels=labels)


def _engineer_features_uncached(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Add all derived, leakage-safe (pre-publication) engineered features."""
    df = df.copy()

    text_col = mapping.get("post_content")
    if text_col and text_col in df.columns:
        df = add_sentiment_columns(df, text_col)

    hashtag_col = mapping.get("hashtags")
    if hashtag_col and hashtag_col in df.columns:
        df["hashtag_count"] = extract_hashtag_count(df, hashtag_col)
        df["hashtag_band"] = hashtag_band(df["hashtag_count"])

    followers_col = mapping.get("followers_count")
    if followers_col and followers_col in df.columns:
        df["follower_band"] = follower_band(df[followers_col])

    length_col = mapping.get("content_length")
    if length_col and length_col in df.columns:
        df["content_length_band"] = content_length_band(df[length_col])

    date_col = mapping.get("post_date")
    if date_col and date_col in df.columns:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        df["post_day_of_week"] = dt.dt.day_name()
        df["post_month"] = dt.dt.month_name()
        df["is_weekend"] = dt.dt.dayofweek.isin([5, 6])

    creation_col = mapping.get("account_creation_date")
    if creation_col and creation_col in df.columns and date_col and date_col in df.columns:
        created = pd.to_datetime(df[creation_col], errors="coerce")
        posted = pd.to_datetime(df[date_col], errors="coerce")
        df["account_age_days"] = (posted - created).dt.days.clip(lower=0)

    following_col = mapping.get("following_count")
    if followers_col and following_col and followers_col in df.columns and following_col in df.columns:
        df["follower_following_ratio"] = df[followers_col] / df[following_col].replace(0, 1)

    return df


# Disk-cached for the same reason as add_sentiment_columns above — this is
# the function that calls it, so caching here covers the whole engineered
# feature set (sentiment + bands + date features) in one cached artifact.
engineer_features = _MEMORY.cache(_engineer_features_uncached)


def build_target(train_df: pd.DataFrame, other_dfs: list[pd.DataFrame], engagement_col: str,
                  percentile: float = 0.75) -> tuple[pd.DataFrame, list[pd.DataFrame], float]:
    """
    Threshold = the given percentile of engagement_rate computed ONLY on train_df,
    then applied unchanged to other_dfs (e.g. the prediction set).
    """
    threshold = train_df[engagement_col].quantile(percentile)

    train_df = train_df.copy()
    train_df["high_engagement"] = (train_df[engagement_col] > threshold).astype(int)

    out_others = []
    for d in other_dfs:
        d = d.copy()
        if engagement_col in d.columns:
            d["high_engagement"] = (d[engagement_col] > threshold).astype(int)
        out_others.append(d)

    return train_df, out_others, threshold


def classify_variable_roles(df_columns: list[str], mapping: dict) -> pd.DataFrame:
    """Leakage-check table: | Variable | Role | Used for Prediction? | Reason |"""
    post_pub_roles = {"likes", "comments", "shares", "views", "saves", "engagement_rate"}
    id_roles = {"post_id", "user_id", "user_name"}

    role_by_col = {}
    for role, col in mapping.items():
        if col:
            role_by_col[col] = role

    rows = []
    for col in df_columns:
        role = role_by_col.get(col, "unmapped / derived")
        if role in post_pub_roles:
            rows.append({"Variable": col, "Role": role, "Used for Prediction?": "No",
                         "Reason": "Post-publication outcome — would leak the target."})
        elif role in id_roles:
            rows.append({"Variable": col, "Role": role, "Used for Prediction?": "No",
                         "Reason": "Identifier / free-text label with no predictive meaning."})
        elif role == "high_engagement":
            rows.append({"Variable": col, "Role": "target", "Used for Prediction?": "N/A",
                         "Reason": "This is the prediction target itself."})
        elif role in ("post_content",):
            rows.append({"Variable": col, "Role": role, "Used for Prediction?": "No (raw form)",
                         "Reason": "Raw text not fed directly to tabular models; used to derive sentiment_polarity instead."})
        else:
            rows.append({"Variable": col, "Role": role, "Used for Prediction?": "Yes",
                         "Reason": "Known before publication; legitimate predictor."})
    return pd.DataFrame(rows)


PRE_PUBLICATION_ROLES = [
    "gender", "age", "is_verified", "location", "category", "device", "language",
    "content_length", "followers_count", "following_count", "has_media",
]

ENGINEERED_FEATURE_COLUMNS = [
    "hashtag_count", "follower_band", "hashtag_band", "content_length_band",
    "post_day_of_week", "post_month", "is_weekend", "account_age_days",
    "follower_following_ratio", "sentiment_polarity", "sentiment_subjectivity",
]

POST_PUBLICATION_OUTCOMES = ["likes", "comments", "shares", "views", "saves", "engagement_rate"]


def get_pre_publication_feature_columns(df: pd.DataFrame, mapping: dict) -> list[str]:
    cols = []
    for role in PRE_PUBLICATION_ROLES:
        actual = mapping.get(role)
        if actual and actual in df.columns:
            cols.append(actual)
    for col in ENGINEERED_FEATURE_COLUMNS:
        if col in df.columns:
            cols.append(col)
    return cols


# ==========================================================================
# LAYER 5 — MACHINE LEARNING  (was modeling.py)
# ==========================================================================
# See module docstring for why n_jobs is capped at SAFE_N_JOBS and tuning
# defaults to off. `MODEL_DEFINITIONS[name]["default_params"]` is a single,
# already-reasonable hyperparameter set used for the fast, untuned path;
# `param_grid` is only used when the caller explicitly opts into tuning.

def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipeline = "passthrough"
    if scale_numeric:
        num_pipeline = StandardScaler()

    cat_pipeline = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipeline, categorical_cols))

    return ColumnTransformer(transformers, remainder="drop")


def split_train_validation(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y)


def _model_definitions():
    """Built lazily so SAFE_N_JOBS (which can be overridden via env var) is honored."""
    return {
        "Logistic Regression": {
            "scale_numeric": True,
            # n_jobs deliberately omitted: recent scikit-learn versions no
            # longer use it for LogisticRegression's default solver and emit
            # a FutureWarning if it's passed at all.
            "make_estimator": lambda n_jobs: LogisticRegression(
                max_iter=1000, random_state=RANDOM_SEED, C=1.0),
            "default_params": {"C": 1.0},
            "param_grid": {"clf__C": [0.1, 1.0, 10.0]},
        },
        "Random Forest": {
            "scale_numeric": False,
            "make_estimator": lambda n_jobs: RandomForestClassifier(
                random_state=RANDOM_SEED, n_jobs=n_jobs, n_estimators=200, max_depth=16),
            "default_params": {"n_estimators": 200, "max_depth": 16},
            # smaller than before — light tuning, not an exhaustive search
            "param_grid": {"clf__n_estimators": [150, 300], "clf__max_depth": [10, 20]},
        },
        "XGBoost": {
            "scale_numeric": False,
            "make_estimator": lambda n_jobs: XGBClassifier(
                random_state=RANDOM_SEED, eval_metric="logloss", n_jobs=n_jobs,
                n_estimators=200, max_depth=4, learning_rate=0.1),
            "default_params": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1},
            "param_grid": {"clf__n_estimators": [150, 300], "clf__max_depth": [3, 5], "clf__learning_rate": [0.05, 0.1]},
        },
        "SVM": {
            # SVC has no n_jobs param; trained on a stratified subsample to
            # keep runtime bounded regardless of tuning mode.
            "scale_numeric": True,
            "make_estimator": lambda n_jobs: SVC(probability=True, random_state=RANDOM_SEED, C=1.0, kernel="rbf"),
            "default_params": {"C": 1.0, "kernel": "rbf"},
            "param_grid": {"clf__C": [1.0, 10.0], "clf__kernel": ["rbf"]},
        },
    }


MODEL_DEFINITIONS = _model_definitions()
MODEL_NAMES = list(MODEL_DEFINITIONS.keys())


def train_one_model(name: str, X_train: pd.DataFrame, y_train: pd.Series, tune: bool = False,
                     cv_folds: int = 2, svm_subsample: int = 4000):
    """
    Train (+ optionally tune) a single model. Returns (fitted_pipeline, train_seconds, params_used).

    tune=False (default, fast path): a single fit using an already-sensible
        fixed hyperparameter set — no GridSearchCV, no nested parallelism.
    tune=True (slow path, opt-in): GridSearchCV over a small grid. To avoid
        oversubscribing a CPU-quota-limited container, the estimator itself
        runs single-threaded (n_jobs=1) while GridSearchCV does the
        parallelism (n_jobs=SAFE_N_JOBS) — never both at once.
    """
    definition = MODEL_DEFINITIONS[name]

    X_fit, y_fit = X_train, y_train
    if name == "SVM" and len(X_train) > svm_subsample:
        X_fit, _, y_fit, _ = train_test_split(
            X_train, y_train, train_size=svm_subsample, stratify=y_train, random_state=RANDOM_SEED
        )

    preprocessor = build_preprocessor(X_fit, scale_numeric=definition["scale_numeric"])

    start = time.time()
    if not tune:
        estimator = definition["make_estimator"](SAFE_N_JOBS)
        pipe = Pipeline([("prep", preprocessor), ("clf", estimator)])
        pipe.fit(X_fit, y_fit)
        elapsed = time.time() - start
        return pipe, elapsed, definition["default_params"]

    # Tuned path: estimator single-threaded, search parallel (not both).
    estimator = definition["make_estimator"](1)
    pipe = Pipeline([("prep", preprocessor), ("clf", estimator)])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(pipe, definition["param_grid"], cv=cv, scoring="f1", n_jobs=SAFE_N_JOBS)
    search.fit(X_fit, y_fit)
    elapsed = time.time() - start
    return search.best_estimator_, elapsed, search.best_params_


def train_models_sequential(X_train, y_train, model_names=None, tune: bool = False,
                             cv_folds: int = 2, svm_subsample: int = 4000):
    """
    Generator: trains models ONE AT A TIME, yielding progress after each so
    a UI (e.g. Streamlit) can render a live progress bar / status line
    instead of blocking silently behind one big spinner. This is the
    "load models slowly" behavior — visible, incremental, and each result
    is independently cacheable by the caller.

    Yields dicts: {"name", "index", "total", "model", "train_seconds", "params"}
    """
    if model_names is None:
        model_names = MODEL_NAMES
    total = len(model_names)
    for i, name in enumerate(model_names, start=1):
        model, secs, params = train_one_model(
            name, X_train, y_train, tune=tune, cv_folds=cv_folds, svm_subsample=svm_subsample
        )
        yield {"name": name, "index": i, "total": total, "model": model,
               "train_seconds": secs, "params": params}


def train_all_models(X_train, y_train, model_names=None, tune: bool = False,
                      cv_folds: int = 2, svm_subsample: int = 4000):
    """Non-generator convenience wrapper (used by the smoke test): trains all
    requested models and returns dict[name] = {'model', 'train_seconds', 'best_params'}."""
    results = {}
    for item in train_models_sequential(X_train, y_train, model_names, tune, cv_folds, svm_subsample):
        results[item["name"]] = {
            "model": item["model"], "train_seconds": item["train_seconds"], "best_params": item["params"],
        }
    return results


# ==========================================================================
# LAYER 5/6 — EVALUATION  (was evaluation.py)
# ==========================================================================

def evaluate_model(model, X_val, y_val) -> dict:
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, zero_division=0),
        "recall": recall_score(y_val, preds, zero_division=0),
        "f1": f1_score(y_val, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_val, proba),
        "confusion_matrix": confusion_matrix(y_val, preds),
        "y_proba": proba,
    }


def build_comparison_table(results: dict, X_val, y_val) -> tuple[pd.DataFrame, dict]:
    rows = []
    eval_cache = {}
    for name, info in results.items():
        ev = evaluate_model(info["model"], X_val, y_val)
        eval_cache[name] = ev
        rows.append({
            "Model": name,
            "Accuracy": round(ev["accuracy"], 4),
            "Precision": round(ev["precision"], 4),
            "Recall": round(ev["recall"], 4),
            "F1 Score": round(ev["f1"], 4),
            "ROC-AUC": round(ev["roc_auc"], 4),
            "Training Time (s)": round(info["train_seconds"], 2),
        })
    table = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    return table, eval_cache


def select_best_model(comparison_table: pd.DataFrame, primary_metric: str = "F1 Score") -> str:
    return comparison_table.sort_values(primary_metric, ascending=False).iloc[0]["Model"]


def roc_curve_figure(eval_cache: dict, y_val) -> go.Figure:
    fig = go.Figure()
    for name, ev in eval_cache.items():
        fpr, tpr, _ = roc_curve(y_val, ev["y_proba"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={ev['roc_auc']:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
    fig.update_layout(title="ROC Curves — Model Comparison", xaxis_title="False Positive Rate",
                       yaxis_title="True Positive Rate")
    return fig


def confusion_matrix_figure(cm: np.ndarray, model_name: str) -> go.Figure:
    labels = ["Low Engagement", "High Engagement"]
    fig = go.Figure(data=go.Heatmap(z=cm, x=labels, y=labels, colorscale="Blues", text=cm, texttemplate="%{text}"))
    fig.update_layout(title=f"Confusion Matrix — {model_name}", xaxis_title="Predicted", yaxis_title="Actual")
    return fig


def select_threshold(y_val, y_proba, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.30, 0.71, 0.05)
    rows = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 2),
            "precision": round(precision_score(y_val, preds, zero_division=0), 4),
            "recall": round(recall_score(y_val, preds, zero_division=0), 4),
            "f1": round(f1_score(y_val, preds, zero_division=0), 4),
        })
    return pd.DataFrame(rows)


def best_threshold_by_f1(threshold_table: pd.DataFrame) -> float:
    return float(threshold_table.sort_values("f1", ascending=False).iloc[0]["threshold"])


# ==========================================================================
# LAYER 6 — PREDICTION ON UNSEEN DATA  (was prediction.py)
# ==========================================================================

def predict_on_new_data(pipeline, X_new: pd.DataFrame, threshold: float) -> pd.DataFrame:
    proba = pipeline.predict_proba(X_new)[:, 1]
    preds = (proba >= threshold).astype(int)
    out = pd.DataFrame({
        "predicted_high_engagement": preds,
        "probability_high_engagement": proba,
        "probability_low_engagement": 1 - proba,
    }, index=X_new.index)
    out["predicted_label"] = out["predicted_high_engagement"].map({1: "High Engagement", 0: "Low Engagement"})
    out["engagement_rank"] = out["probability_high_engagement"].rank(ascending=False, method="min").astype(int)
    return out


def evaluate_out_of_sample(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def assemble_prediction_output(original_df: pd.DataFrame, engineered_df: pd.DataFrame,
                                prediction_df: pd.DataFrame) -> pd.DataFrame:
    keep_engineered = [c for c in [
        "sentiment_polarity", "sentiment_label", "hashtag_count", "follower_band",
        "content_length_band", "post_day_of_week", "is_weekend", "account_age_days",
    ] if c in engineered_df.columns]

    out = original_df.copy()
    for c in keep_engineered:
        out[c] = engineered_df[c].values
    for c in prediction_df.columns:
        out[c] = prediction_df[c].values
    return out.sort_values("engagement_rank")


# ==========================================================================
# LAYER 7 — EXPLAINABLE AI  (was explainability.py)
# ==========================================================================

def get_feature_names(preprocessor) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def explain_tree_model(pipeline, X_sample: pd.DataFrame, max_rows: int = 400):
    """Return (shap_values, feature_names, X_transformed_sample, X_sample) for a tree-based pipeline.
    max_rows defaults lower than before (400, was 500) — SHAP cost scales
    with sample size and this keeps Page 7 responsive on a throttled CPU."""
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    X_sample = X_sample.sample(min(max_rows, len(X_sample)), random_state=42)
    X_trans = prep.transform(X_sample)
    feature_names = get_feature_names(prep)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_trans)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return shap_values, feature_names, X_trans, X_sample.reset_index(drop=True)


def global_importance_table(shap_values, feature_names) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    table = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    table = table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return table


def logistic_coefficient_importance(pipeline) -> pd.DataFrame:
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    feature_names = get_feature_names(prep)
    coefs = clf.coef_[0]
    table = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def explain_single_prediction(shap_values, feature_names, row_idx: int, top_k: int = 6) -> pd.DataFrame:
    row = shap_values[row_idx]
    table = pd.DataFrame({"feature": feature_names, "shap_value": row})
    table["direction"] = np.where(table["shap_value"] >= 0, "Positive contributor", "Negative contributor")
    table["abs_value"] = table["shap_value"].abs()
    return table.sort_values("abs_value", ascending=False).head(top_k)


# ==========================================================================
# LAYER 8 — CONTENT OPTIMIZATION PLAYBOOK  (was optimization.py)
# ==========================================================================

CONTROLLABLE_FIELDS = {
    "category": None,
    "device": None,
    "sentiment_label": ["Positive", "Neutral", "Negative"],
    "has_media": [True, False],
    "hashtag_band": ["None", "Low (1-2)", "Medium (3-4)", "High (5+)"],
    "content_length_band": ["Short (<75)", "Medium (75-150)", "Long (150-250)", "Very Long (250+)"],
    "post_day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
}

CONTEXTUAL_FIELDS_NOTE = (
    "followers_count, following_count, account_age_days, is_verified, gender, "
    "language, location, follower_band, follower_following_ratio, age"
)


def build_representative_context(train_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    context = {}
    for col in feature_cols:
        if col not in train_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]) or pd.api.types.is_bool_dtype(train_df[col]):
            context[col] = train_df[col].median()
        else:
            context[col] = train_df[col].mode(dropna=True).iloc[0]
    return context


def generate_scenarios(train_df: pd.DataFrame, feature_cols: list[str], max_combinations: int = 300) -> pd.DataFrame:
    import itertools

    search_space = {}
    if "category" in feature_cols and "category" in train_df.columns:
        search_space["category"] = sorted(train_df["category"].dropna().unique().tolist())
    if "device" in feature_cols and "device" in train_df.columns:
        search_space["device"] = sorted(train_df["device"].dropna().unique().tolist())
    for field in ("sentiment_label", "has_media", "hashtag_band", "content_length_band", "post_day_of_week"):
        if field in feature_cols and field in train_df.columns:
            vals = CONTROLLABLE_FIELDS.get(field) or sorted(train_df[field].dropna().astype(str).unique().tolist())
            search_space[field] = vals

    context = build_representative_context(train_df, [c for c in feature_cols if c not in search_space])

    keys = list(search_space.keys())
    combos = list(itertools.product(*[search_space[k] for k in keys]))
    if len(combos) > max_combinations:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(combos), size=max_combinations, replace=False)
        combos = [combos[i] for i in idx]

    rows = []
    for combo in combos:
        row = dict(context)
        row.update(dict(zip(keys, combo)))
        rows.append(row)

    scenario_df = pd.DataFrame(rows)
    return scenario_df[feature_cols]


def score_scenarios(pipeline, scenario_df: pd.DataFrame) -> pd.DataFrame:
    proba = pipeline.predict_proba(scenario_df)[:, 1]
    out = scenario_df.copy()
    out["predicted_probability_high_engagement"] = proba
    return out.sort_values("predicted_probability_high_engagement", ascending=False).reset_index(drop=True)


# ==========================================================================
# LAYER 9 — MANAGERIAL RECOMMENDATIONS  (was recommendations.py)
# ==========================================================================

def insight_card(finding: str, evidence: str, interpretation: str, implication: str, caveat: str) -> dict:
    return {
        "Finding": finding, "Evidence": evidence, "Interpretation": interpretation,
        "Business Implication": implication, "Caveat": caveat,
    }


def category_insight(category_summary: pd.DataFrame, category_col: str) -> dict:
    top = category_summary.iloc[0]
    bottom = category_summary.iloc[-1]
    return insight_card(
        finding=f"'{top[category_col]}' content shows the highest average engagement rate "
                f"({top['avg_engagement']:.4f}), while '{bottom[category_col]}' shows the lowest "
                f"({bottom['avg_engagement']:.4f}).",
        evidence="Computed from grouped mean engagement_rate on the training dataset.",
        interpretation="Certain content categories are more consistently associated with stronger audience response.",
        implication=f"Prioritize production effort toward '{top[category_col]}'-style content where strategically relevant.",
        caveat="Association, not causation — audience composition and category may be confounded.",
    )


def device_insight(device_summary: pd.DataFrame, device_col: str) -> dict:
    top = device_summary.iloc[0]
    return insight_card(
        finding=f"Posts published via '{top[device_col]}' show the highest average engagement rate "
                f"({top['avg_engagement']:.4f}) in this dataset.",
        evidence="Computed from grouped mean engagement_rate by device/channel on the training dataset.",
        interpretation="This dataset does not include a true multi-network 'platform' field (e.g. Instagram vs. X); "
                        "'device' is the closest available channel-level dimension.",
        implication="Treat this as a directional signal about the posting channel, not a cross-platform strategy conclusion.",
        caveat="No platform variable exists in the uploaded data — this finding is limited to device/channel, not social network.",
    )


def sentiment_insight(sentiment_summary: pd.DataFrame) -> dict:
    top = sentiment_summary.sort_values("avg_engagement", ascending=False).iloc[0]
    return insight_card(
        finding=f"Captions with '{top['sentiment_label']}' sentiment show the highest average engagement "
                f"({top['avg_engagement']:.4f}) among sentiment classes derived from caption text.",
        evidence="Sentiment computed via TextBlob polarity on post_content; grouped mean engagement_rate.",
        interpretation="Tone of the caption is predictive of, and associated with, audience engagement.",
        implication="Favor caption tones consistent with the top-performing sentiment class, tested via A/B experiments.",
        caveat="Sentiment is algorithmically inferred (TextBlob), not human-labeled; some misclassification is expected.",
    )


def timing_insight(weekend_series: pd.Series) -> dict:
    better = weekend_series.idxmax()
    return insight_card(
        finding=f"{better} posts show higher average engagement than the alternative in the training data.",
        evidence="Grouped mean engagement_rate by weekend/weekday flag derived from post_date.",
        interpretation="Publishing day may relate to audience availability/attention.",
        implication=f"Weight the content calendar toward {better.lower()} publishing where feasible.",
        caveat="No time-of-day field exists in this dataset, so intra-day timing cannot be assessed — a documented limitation.",
    )


def model_insight(best_model: str, metrics_row: pd.Series) -> dict:
    return insight_card(
        finding=f"{best_model} was selected as the best-performing model (F1={metrics_row['F1 Score']:.3f}, "
                f"ROC-AUC={metrics_row['ROC-AUC']:.3f}) on the held-out validation split.",
        evidence="Computed via stratified train/validation split within the 15,000-record training dataset.",
        interpretation="This model best balances precision and recall for identifying high-engagement posts.",
        implication="Use this model's probability score to rank draft content ideas before publishing.",
        caveat="Performance reflects historical/synthetic patterns and may not generalize to entirely new content strategies.",
    )
