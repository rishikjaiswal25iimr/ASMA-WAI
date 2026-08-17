"""
eda.py
======
Layer 2 — Descriptive / Exploratory Analytics.
Every function returns data (DataFrame) and/or a Plotly figure so the
Streamlit app can render it consistently, and so it can also be unit-tested
without Streamlit running.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


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
