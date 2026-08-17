"""
preprocessing.py
=================
Layer 1/4 support — data cleaning shared by both the training dataset and the
prediction dataset. Applied IDENTICALLY (same logic, no leakage of statistics
computed on one dataset into the other) to both, except where noted.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def clean_dataset(df: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, dict]:
    """
    Perform automatic, defensive cleaning:
      - drop exact duplicate rows
      - parse date columns
      - coerce obviously-numeric columns
      - report missing-value treatment applied

    Returns (cleaned_df, report_dict)
    """
    report = {"steps": []}
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        report["steps"].append(f"Removed {before - after} exact duplicate rows.")

    # Parse date-like columns identified via mapping
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

    # Missing-value treatment, column by column
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

    # Strip whitespace on object/string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    return df, report


def detect_outliers_iqr(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Return a summary table of IQR-based outlier counts per numeric column (inspection only)."""
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
