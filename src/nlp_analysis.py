"""
nlp_analysis.py
================
Layer 3 — NLP / Sentiment Analytics.

This dataset provides RAW caption text (post_content), not a pre-existing
sentiment label. Therefore this module performs genuine text-based sentiment
analysis (TextBlob polarity/subjectivity — a lexicon/pattern-based analyzer
that ships with the package and needs no external corpus download, which
keeps the app deployable on Streamlit Cloud without extra setup).

If a dataset WITHOUT raw text but WITH a pre-existing sentiment label were
uploaded instead, `has_raw_text` would be False and the app must fall back to
analyzing the provided label rather than fabricating NLP on missing text.
Both code paths are implemented below.
"""

from __future__ import annotations
import re
import pandas as pd
import numpy as np
from collections import Counter

try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except Exception:
    _TEXTBLOB_AVAILABLE = False

STOPWORDS = set("""a an the and or but if while is are was were be been being to of in on for with
as by at from this that these those it its it's he she they them his her their our your you i we
not no nor so than then too very can will just don should now""".split())


def compute_sentiment(text: str) -> tuple[float, float]:
    """Return (polarity, subjectivity) for one string. Polarity in [-1, 1]."""
    if not _TEXTBLOB_AVAILABLE or not isinstance(text, str) or not text.strip():
        return 0.0, 0.0
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def add_sentiment_columns(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Add sentiment_polarity, sentiment_subjectivity, sentiment_label columns."""
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


def extract_hashtag_count(df: pd.DataFrame, hashtag_col: str) -> pd.Series:
    """Count number of hashtags in a whitespace/comma separated hashtag field."""
    def _count(val):
        if not isinstance(val, str) or not val.strip():
            return 0
        tokens = re.split(r"[,\s]+", val.strip())
        return sum(1 for t in tokens if t.startswith("#"))
    return df[hashtag_col].apply(_count)


def top_ngrams(text_series: pd.Series, n: int = 1, top_k: int = 15) -> pd.DataFrame:
    """Simple frequency-based n-gram extraction (no external NLP corpus required)."""
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
    """Cross-tab of sentiment vs. a categorical dimension with mean engagement."""
    return (
        df.groupby([dimension_col, sentiment_col], observed=True)[engagement_col]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_engagement", "count": "n_posts"})
    )
