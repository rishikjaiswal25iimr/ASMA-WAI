"""
prediction.py
=============
Layer 6 — Final Prediction on the completely unseen 5,000-record prediction
dataset. The model was selected AND its threshold tuned using only the
15,000-record training dataset; this module only ever *applies* that frozen
artifact to new data.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)


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
    """Only called when the prediction dataset actually contains ground-truth labels."""
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
    """Combine original fields + key engineered fields + predictions, without overwriting originals."""
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
