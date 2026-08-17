"""
evaluation.py
=============
Layers 5-6 helper — model comparison metrics, ROC curves, confusion matrices,
and business-aware threshold selection. All computed on the TRAIN/VALIDATION
split only (never on the 5,000-record prediction set).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)


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


def build_comparison_table(results: dict, X_val, y_val) -> pd.DataFrame:
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
    """Best model chosen by F1 (balances precision/recall) — NOT raw accuracy, per section 17."""
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
    """Evaluate a grid of thresholds on validation data and return precision/recall/F1 per threshold."""
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
