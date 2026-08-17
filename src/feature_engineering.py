"""
feature_engineering.py
=======================
Layer 4 — Feature Engineering + target construction + explicit
pre-publication vs post-publication variable classification (data-leakage
guard, section 39 of the master prompt).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from src.nlp_analysis import add_sentiment_columns, extract_hashtag_count

RANDOM_SEED = 42


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


def engineer_features(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Add all derived, leakage-safe (pre-publication) engineered features."""
    df = df.copy()

    # --- Sentiment from raw caption text (pre-publication: author writes it before posting) ---
    text_col = mapping.get("post_content")
    if text_col and text_col in df.columns:
        df = add_sentiment_columns(df, text_col)

    # --- Hashtag count from hashtags field ---
    hashtag_col = mapping.get("hashtags")
    if hashtag_col and hashtag_col in df.columns:
        df["hashtag_count"] = extract_hashtag_count(df, hashtag_col)
        df["hashtag_band"] = hashtag_band(df["hashtag_count"])

    # --- Follower band (contextual, non-controllable, but legitimate pre-pub predictor) ---
    followers_col = mapping.get("followers_count")
    if followers_col and followers_col in df.columns:
        df["follower_band"] = follower_band(df[followers_col])

    # --- Content length band ---
    length_col = mapping.get("content_length")
    if length_col and length_col in df.columns:
        df["content_length_band"] = content_length_band(df[length_col])

    # --- Date-derived features (day of week / weekend). NOTE: this dataset's
    # post_date has no time-of-day component, so "posting hour" cannot be
    # computed — this is documented as a limitation in the app rather than
    # fabricated. ---
    date_col = mapping.get("post_date")
    if date_col and date_col in df.columns:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        df["post_day_of_week"] = dt.dt.day_name()
        df["post_month"] = dt.dt.month_name()
        df["is_weekend"] = dt.dt.dayofweek.isin([5, 6])

    # --- Account age (days) at time of posting: contextual predictor ---
    creation_col = mapping.get("account_creation_date")
    if creation_col and creation_col in df.columns and date_col and date_col in df.columns:
        created = pd.to_datetime(df[creation_col], errors="coerce")
        posted = pd.to_datetime(df[date_col], errors="coerce")
        df["account_age_days"] = (posted - created).dt.days.clip(lower=0)

    # --- Follower/following ratio ---
    following_col = mapping.get("following_count")
    if followers_col and following_col and followers_col in df.columns and following_col in df.columns:
        df["follower_following_ratio"] = df[followers_col] / df[following_col].replace(0, 1)

    return df


def build_target(train_df: pd.DataFrame, other_dfs: list[pd.DataFrame], engagement_col: str,
                  percentile: float = 0.75) -> tuple[pd.DataFrame, list[pd.DataFrame], float]:
    """
    Create the binary High Engagement target.
    Threshold = the given percentile of engagement_rate computed ONLY on train_df.
    The SAME threshold value is then applied to any other_dfs (e.g. the prediction set)
    so the prediction set never influences its own threshold (section 8 / 18 requirement).
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
    """
    Build the leakage-check table required by section 39:
    | Variable | Role | Used for Prediction? | Reason |
    """
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


# Canonical ROLES (not literal column names — resolved via the mapping, since
# mapping.py deliberately never renames the user's actual columns) that are
# legitimate pre-publication predictors, plus the fixed names of columns this
# module itself creates during engineer_features().
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
    """
    Resolve the canonical PRE_PUBLICATION_ROLES + ENGINEERED_FEATURE_COLUMNS into the
    ACTUAL column names present in df (using the mapping built by data_loader), skipping
    any role/column that isn't available. This is the single source of truth for which
    columns feed the ML models — used identically for training and prediction data.
    """
    cols = []
    for role in PRE_PUBLICATION_ROLES:
        actual = mapping.get(role)
        if actual and actual in df.columns:
            cols.append(actual)
    for col in ENGINEERED_FEATURE_COLUMNS:
        if col in df.columns:
            cols.append(col)
    return cols
