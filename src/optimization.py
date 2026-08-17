"""
optimization.py
================
Layer 8 — Content Optimization Playbook.

Uses the trained model to score COMBINATIONS of controllable, pre-publication
variables (category, device, sentiment, has_media, hashtag_band, content
length band, day of week) while holding non-controllable / contextual
variables (followers, account age, verified status, gender, language,
location) fixed at representative values drawn from the training data
(median for numeric, mode for categorical).

This produces MODEL-BASED, HYPOTHETICAL scenarios — never presented as
historical fact.
"""

from __future__ import annotations
import itertools
import pandas as pd
import numpy as np

CONTROLLABLE_FIELDS = {
    "category": None,           # filled dynamically from training data uniques
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
    """Median (numeric) / mode (categorical) values for every feature NOT in the search space."""
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
    """Build a grid of controllable-variable combinations, holding contextual vars at representative values."""
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
