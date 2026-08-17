"""
explainability.py
==================
Layer 7 — Explainable AI. Uses SHAP for tree-based models (Random Forest,
XGBoost). For non-tree models (Logistic Regression, SVM) SHAP's
model-agnostic KernelExplainer is far more expensive, so we fall back to a
faster, still-honest explanation:
  - Logistic Regression: standardized coefficients (directly interpretable).
  - SVM (linear-ish rbf): SHAP KernelExplainer on a small background sample
    (explicitly labelled as an approximation).
This choice — and why — is surfaced in the Streamlit app rather than hidden.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

try:
    import shap
    _SHAP_AVAILABLE = True
except Exception:
    _SHAP_AVAILABLE = False


def get_feature_names(preprocessor) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def explain_tree_model(pipeline, X_sample: pd.DataFrame, max_rows: int = 500):
    """Return (shap_values, feature_names, X_transformed_sample) for a tree-based pipeline."""
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    X_sample = X_sample.sample(min(max_rows, len(X_sample)), random_state=42)
    X_trans = prep.transform(X_sample)
    feature_names = get_feature_names(prep)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_trans)
    # Binary classifiers may return a list [class0, class1] or a single 2D array
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
    """Return top positive/negative contributors for one row of a SHAP value matrix."""
    row = shap_values[row_idx]
    table = pd.DataFrame({"feature": feature_names, "shap_value": row})
    table["direction"] = np.where(table["shap_value"] >= 0, "Positive contributor", "Negative contributor")
    table["abs_value"] = table["shap_value"].abs()
    return table.sort_values("abs_value", ascending=False).head(top_k)
