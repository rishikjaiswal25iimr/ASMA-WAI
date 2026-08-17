"""
data_loader.py
==============
Layer 1 — Data Discovery.

Handles:
- reading uploaded CSV/XLSX files
- automatic column-type detection (numeric, categorical, datetime, text, boolean)
- intelligent (non-destructive) column-name mapping to canonical analytical roles
- a data-quality report used across the Streamlit app

IMPORTANT: this module NEVER blindly renames the user's columns. It builds a
mapping dictionary (canonical_role -> actual_column_name) that the rest of the
pipeline consults. If a canonical role cannot be found, the value is None and
downstream code must handle that gracefully (skip the analysis, warn the user).
"""

from __future__ import annotations
import io
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# Canonical roles this project cares about, with a list of aliases used for
# fuzzy / substring matching against the actual uploaded column names.
# This list was built FROM the actual schema of this project's dataset
# (post_id, user_id, user_name, user_gender, user_age, followers_count,
# following_count, account_creation_date, is_verified, location, topic,
# post_content, content_length, hashtags, has_media, post_date, device,
# language, likes, comments, shares, engagement_rate) plus common
# variations a marketer might upload (likes/like_count, platform, etc.)
# ----------------------------------------------------------------------
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

# Roles that represent outcomes only known AFTER a post is published.
# These must NEVER be used as predictive features for the pre-publication
# content-optimization model (see feature_engineering.py).
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
            # substring fallback
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
            # crude heuristic: long average string length => free text, else categorical
            avg_len = s.astype(str).str.len().mean()
            n_unique = s.nunique()
            if avg_len > 40 and n_unique > 0.5 * len(s):
                text.append(col)
            else:
                categorical.append(col)

    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime_cols,
        "text": text,
        "boolean": boolean,
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
