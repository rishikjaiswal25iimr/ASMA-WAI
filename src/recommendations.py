"""
recommendations.py
===================
Layer 9 — Managerial Recommendations, generated ONLY from actual calculated
statistics passed in (never hardcoded). Each function returns a small
dictionary with Finding / Evidence / Interpretation / Business Implication /
Caveat, as required by section 24.
"""

from __future__ import annotations
import pandas as pd


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
