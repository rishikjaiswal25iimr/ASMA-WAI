import io
import re
import time
import itertools
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier

try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except ImportError:
    _TEXTBLOB_AVAILABLE = False

RANDOM_SEED = 42

# ==============================================================================
# LAYER 1: DATA DISCOVERY & PREPROCESSING
# ==============================================================================
CANONICAL_ALIASES = {
    "post_id": ["post_id", "id", "postid"], "user_id": ["user_id", "userid", "author_id"],
    "user_name": ["user_name", "username", "handle"], "gender": ["user_gender", "gender"],
    "age": ["user_age", "age"], "followers_count": ["followers_count", "follower_count", "followers"],
    "following_count": ["following_count", "follower_following", "following"],
    "account_creation_date": ["account_creation_date", "account_created", "signup_date"],
    "is_verified": ["is_verified", "verified", "verified_status"],
    "location": ["location", "country", "city"], "category": ["topic", "category", "niche"],
    "post_content": ["post_content", "caption", "text", "content"],
    "content_length": ["content_length", "caption_length"], "hashtags": ["hashtags", "hashtag", "tags"],
    "has_media": ["has_media", "media_present"], "post_date": ["post_date", "timestamp", "published_at"],
    "device": ["device", "platform", "channel"], "language": ["language", "lang"],
    "likes": ["likes", "like_count"], "comments": ["comments", "comment_count"],
    "shares": ["shares", "share_count", "retweets"], "views": ["views", "view_count", "impressions"],
    "saves": ["saves", "save_count", "bookmarks"], "engagement_rate": ["engagement_rate", "engagement"],
}

def load_dataset(file_path_or_obj) -> pd.DataFrame:
    if isinstance(file_path_or_obj, str) and file_path_or_obj.lower().endswith(".xlsx"):
        return pd.read_excel(file_path_or_obj)
    return pd.read_csv(file_path_or_obj)

def build_column_mapping(df: pd.DataFrame) -> dict:
    lower_cols = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for role, aliases in CANONICAL_ALIASES.items():
        found = next((lower_cols[a] for a in aliases if a in lower_cols), None)
        if not found:
            found = next((orig for lc, orig in lower_cols.items() if any(a in lc for a in aliases)), None)
        mapping[role] = found
    return mapping

def classify_columns(df: pd.DataFrame) -> dict:
    num, cat, dt, txt, bool_c = [], [], [], [], []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s): bool_c.append(c)
        elif pd.api.types.is_numeric_dtype(s): num.append(c)
        elif pd.to_datetime(s.dropna().astype(str).head(50), errors="coerce").notna().mean() > 0.8: dt.append(c)
        elif s.astype(str).str.len().mean() > 40 and s.nunique() > 0.5 * len(s): txt.append(c)
        else: cat.append(c)
    return {"numeric": num, "categorical": cat, "datetime": dt, "text": txt, "boolean": bool_c}

def data_quality_report(df: pd.DataFrame) -> dict:
    types = classify_columns(df)
    miss = df.isna().sum()
    return {
        "n_records": len(df), "n_variables": df.shape[1], "n_numeric": len(types["numeric"]),
        "n_categorical": len(types["categorical"]), "n_text": len(types["text"]),
        "n_datetime": len(types["datetime"]), "n_duplicates": int(df.duplicated().sum()),
        "missing_table": pd.DataFrame({"column": df.columns, "missing_count": miss.values, "missing_pct": (miss/len(df)*100).round(2)})
    }

def clean_dataset(df: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, dict]:
    report = {"steps": []}
    df = df.copy().drop_duplicates()
    for role in ("post_date", "account_creation_date"):
        col = mapping.get(role)
        if col and col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    m_rows = []
    for col in df.columns:
        nb = df[col].isna().sum()
        if nb > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                fill = df[col].median()
                df[col] = df[col].fillna(fill)
                m_rows.append({"column": col, "missing_before": nb, "treatment": f"median ({fill:.3f})"})
            elif not pd.api.types.is_datetime64_any_dtype(df[col]):
                fill = df[col].mode(dropna=True)[0] if not df[col].mode(dropna=True).empty else "Unknown"
                df[col] = df[col].fillna(fill)
                m_rows.append({"column": col, "missing_before": nb, "treatment": f"mode ('{fill}')"})
    report["missing_value_treatment"] = pd.DataFrame(m_rows) if m_rows else pd.DataFrame()
    return df, report

# ==============================================================================
# LAYER 2: DESCRIPTIVE ANALYTICS
# ==============================================================================
def group_engagement(df: pd.DataFrame, group_col: str, eng_col: str) -> pd.DataFrame:
    g = df.groupby(group_col, observed=True)[eng_col].agg(["mean", "median", "count"]).reset_index()
    g.columns = [group_col, "avg_engagement", "median_engagement", "n_posts"]
    return g.sort_values("avg_engagement", ascending=False)

def bar_chart(df: pd.DataFrame, x: str, y: str, title: str):
    return px.bar(df, x=x, y=y, title=title, color=x, text_auto=".3f").update_layout(showlegend=False)

def correlation_heatmap(df: pd.DataFrame, num_cols: list[str]):
    corr = df[num_cols].corr(numeric_only=True)
    return px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), corr

def scatter_plot(df: pd.DataFrame, x: str, y: str, title: str, trendline=False):
    fig = px.scatter(df, x=x, y=y, title=title, opacity=0.5)
    if trendline: # Custom NumPy polyfit replaces statsmodels lowess dependency
        v = df[[x, y]].dropna()
        if not v.empty and pd.api.types.is_numeric_dtype(v[x]) and pd.api.types.is_numeric_dtype(v[y]):
            m, b = np.polyfit(v[x], v[y], 1)
            fig.add_trace(go.Scatter(x=v[x], y=m*v[x]+b, mode="lines", name="Trend", line=dict(color="red")))
    return fig

def weekend_vs_weekday(df: pd.DataFrame, eng_col: str):
    if "is_weekend" not in df.columns: return None, None
    g = df.groupby("is_weekend", observed=True)[eng_col].mean().rename({True: "Weekend", False: "Weekday"})
    return px.bar(x=g.index.astype(str), y=g.values, title="Weekend vs. Weekday", text_auto=".4f"), g

# ==============================================================================
# LAYER 3 & 4: NLP, SENTIMENT & FEATURE ENGINEERING
# ==============================================================================
def add_sentiment_columns(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df = df.copy()
    pols, subs = [], []
    for t in df[text_col].astype(str):
        if _TEXTBLOB_AVAILABLE and t.strip():
            b = TextBlob(t)
            pols.append(b.sentiment.polarity)
            subs.append(b.sentiment.subjectivity)
        else:
            pols.append(0.0)
            subs.append(0.0)
    df["sentiment_polarity"] = pols
    df["sentiment_subjectivity"] = subs
    df["sentiment_label"] = pd.cut(df["sentiment_polarity"], bins=[-1.01, -0.05, 0.05, 1.01], labels=["Negative", "Neutral", "Positive"]).astype(str)
    return df

def top_ngrams(s: pd.Series, top_k: int = 15) -> pd.DataFrame:
    stops = set("a an the and or but if while is are was were be to of in on for with as by at from it its he she they them his her their not no very can will just".split())
    c = Counter()
    for txt in s.astype(str):
        c.update(w.lower() for w in re.findall(r"[a-zA-Z']+", txt) if len(w) > 2 and w.lower() not in stops)
    return pd.DataFrame(c.most_common(top_k), columns=["ngram", "frequency"])

def engineer_features(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    df = df.copy()
    if mapping.get("post_content") in df.columns:
        df = add_sentiment_columns(df, mapping["post_content"])
    if mapping.get("hashtags") in df.columns:
        df["hashtag_count"] = df[mapping["hashtags"]].apply(lambda x: sum(1 for t in re.split(r"[,\s]+", str(x)) if t.startswith("#")))
        df["hashtag_band"] = pd.cut(df["hashtag_count"], bins=[-1, 0, 2, 4, np.inf], labels=["None", "Low (1-2)", "Medium (3-4)", "High (5+)"])
    if mapping.get("followers_count") in df.columns:
        df["follower_band"] = pd.cut(df[mapping["followers_count"]], bins=[-1, 1000, 10000, 50000, 500000, np.inf], labels=["Nano", "Micro", "Mid", "Macro", "Mega"])
    if mapping.get("content_length") in df.columns:
        df["content_length_band"] = pd.cut(df[mapping["content_length"]], bins=[-1, 75, 150, 250, np.inf], labels=["Short", "Medium", "Long", "Very Long"])
    if mapping.get("post_date") in df.columns:
        dt = pd.to_datetime(df[mapping["post_date"]], errors="coerce")
        df["post_day_of_week"] = dt.dt.day_name()
        df["is_weekend"] = dt.dt.dayofweek.isin([5, 6])
    if mapping.get("account_creation_date") in df.columns and mapping.get("post_date") in df.columns:
        df["account_age_days"] = (pd.to_datetime(df[mapping["post_date"]]) - pd.to_datetime(df[mapping["account_creation_date"]])).dt.days.clip(lower=0)
    return df

def build_target(train_df: pd.DataFrame, eng_col: str, pct: float = 0.75):
    """STRICT DATA SEPARATION: Applies threshold creation ONLY using training set."""
    thresh = train_df[eng_col].quantile(pct)
    train_df = train_df.copy()
    train_df["high_engagement"] = (train_df[eng_col] > thresh).astype(int)
    return train_df, thresh

def classify_variable_roles(cols: list[str], mapping: dict) -> pd.DataFrame:
    roles = {v: k for k, v in mapping.items() if v}
    rows = []
    for c in cols:
        r = roles.get(c, "derived")
        if r in {"likes", "comments", "shares", "views", "saves", "engagement_rate"}:
            rows.append({"Variable": c, "Role": r, "Predictor?": "No", "Reason": "Post-publication leakage"})
        elif r in {"post_id", "user_id", "user_name"}:
            rows.append({"Variable": c, "Role": r, "Predictor?": "No", "Reason": "Identifier"})
        else:
            rows.append({"Variable": c, "Role": r, "Predictor?": "Yes", "Reason": "Pre-publication predictor"})
    return pd.DataFrame(rows)

def get_feature_columns(df: pd.DataFrame, mapping: dict) -> list[str]:
    pre = ["gender", "age", "is_verified", "location", "category", "device", "language", "content_length", "followers_count", "following_count", "has_media"]
    eng = ["hashtag_count", "follower_band", "hashtag_band", "content_length_band", "post_day_of_week", "is_weekend", "account_age_days", "sentiment_polarity"]
    cols = [mapping.get(r) for r in pre if mapping.get(r) in df.columns] + [c for c in eng if c in df.columns]
    return cols

# ==============================================================================
# LAYER 5 & 6: MACHINE LEARNING & PREDICTION
# ==============================================================================
def split_train_validation(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

MODEL_DEFS = {
    "Logistic Regression": {"scale": True, "est": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED), "grid": {"clf__C": [0.1, 1.0]}},
    "Random Forest": {"scale": False, "est": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=2), "grid": {"clf__n_estimators": [100, 200], "clf__max_depth": [8, 16]}},
    "XGBoost": {"scale": False, "est": XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss", n_jobs=2), "grid": {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5]}},
    "SVM": {"scale": True, "est": SVC(probability=True, random_state=RANDOM_SEED), "grid": {"clf__C": [1.0]}}
}

def train_all_models(X_train, y_train, names):
    results = {}
    for name in names:
        d = MODEL_DEFS[name]
        num_c = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        cat_c = [c for c in X_train.columns if c not in num_c]
        prep = ColumnTransformer([("num", StandardScaler() if d["scale"] else "passthrough", num_c),
                                  ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_c)])
        pipe = Pipeline([("prep", prep), ("clf", d["est"])])
        
        # Subsample SVM for Streamlit performance
        X_fit, y_fit = (X_train, y_train)
        if name == "SVM" and len(X_train) > 6000:
            X_fit, _, y_fit, _ = train_test_split(X_train, y_train, train_size=6000, stratify=y_train, random_state=RANDOM_SEED)
        
        # CPU Safe Grid Search (n_jobs=1 to avoid overriding model n_jobs=2 limit)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        search = GridSearchCV(pipe, d["grid"], cv=cv, scoring="f1", n_jobs=1)
        
        st_t = time.time()
        search.fit(X_fit, y_fit)
        results[name] = {"model": search.best_estimator_, "time": time.time()-st_t}
    return results

def evaluate(results, X_val, y_val):
    rows = []
    cache = {}
    for n, r in results.items():
        proba = r["model"].predict_proba(X_val)[:, 1]
        preds = (proba >= 0.5).astype(int)
        cache[n] = {"proba": proba, "preds": preds}
        rows.append({
            "Model": n, 
            "Accuracy": round(accuracy_score(y_val, preds), 4),
            "Precision": round(precision_score(y_val, preds, zero_division=0), 4),
            "Recall": round(recall_score(y_val, preds, zero_division=0), 4),
            "F1 Score": round(f1_score(y_val, preds, zero_division=0), 4), 
            "ROC-AUC": round(roc_auc_score(y_val, proba), 4), 
            "Time (s)": round(r["time"], 2)
        })
    return pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True), cache

def get_best_threshold(y_val, proba):
    ts = np.arange(0.3, 0.71, 0.05)
    best_t, best_f = 0.5, 0.0
    for t in ts:
        f = f1_score(y_val, (proba >= t).astype(int), zero_division=0)
        if f > best_f: best_t, best_f = t, f
    return best_t

def predict_new(model, X_pred, thresh, raw):
    proba = model.predict_proba(X_pred)[:, 1]
    out = raw.copy()
    out["predicted_probability_high_engagement"] = proba
    out["predicted_high_engagement"] = (proba >= thresh).astype(int)
    out["predicted_label"] = out["predicted_high_engagement"].map({1: "High", 0: "Low"})
    return out.sort_values("predicted_probability_high_engagement", ascending=False)

def evaluate_out_of_sample(y_true, y_pred, y_proba):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_proba), 4)
    }

# ==============================================================================
# LAYER 7, 8, 9: EXPLAINABILITY, OPTIMIZATION & RECOMMENDATIONS
# ==============================================================================
def explain_model(pipeline, X_samp, is_tree):
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    X_trans = pd.DataFrame(prep.transform(X_samp), columns=prep.get_feature_names_out())
    
    if is_tree:
        sv = shap.TreeExplainer(clf).shap_values(X_trans)
        if isinstance(sv, list): sv = sv[1]
        df = pd.DataFrame({"feature": X_trans.columns, "importance": np.abs(sv).mean(axis=0)})
    else:
        df = pd.DataFrame({"feature": X_trans.columns, "importance": np.abs(clf.coef_[0])})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)

def generate_scenarios(train_fe, cols, max_n=300):
    cats = train_fe["category"].dropna().unique() if "category" in train_fe else ["Default"]
    devs = train_fe["device"].dropna().unique() if "device" in train_fe else ["Default"]
    sents = ["Positive", "Neutral", "Negative"]
    days = ["Monday", "Wednesday", "Saturday"]
    lens = ["Short", "Medium", "Long"]
    
    grid = list(itertools.product(cats, devs, sents, days, lens))
    if len(grid) > max_n:
        grid = [grid[i] for i in np.random.choice(len(grid), max_n, replace=False)]
    
    df = pd.DataFrame(grid, columns=["category", "device", "sentiment_label", "post_day_of_week", "content_length_band"])
    for c in cols:
        if c not in df.columns:
            df[c] = train_fe[c].median() if pd.api.types.is_numeric_dtype(train_fe[c]) else train_fe[c].mode()[0]
    return df

def score_scenarios(pipeline, df):
    df = df.copy()
    df["prob"] = pipeline.predict_proba(df[pipeline.named_steps["prep"].feature_names_in_])[:, 1]
    return df.sort_values("prob", ascending=False)

def category_insight(cat_summary: pd.DataFrame, cat_col: str) -> dict:
    if cat_summary.empty: return {}
    top = cat_summary.iloc[0]
    return {
        "Finding": f"'{top[cat_col]}' content shows the highest average engagement ({top['avg_engagement']:.4f}).",
        "Evidence": "Computed from grouped mean engagement_rate on the training dataset.",
        "Interpretation": "Certain categories are consistently associated with stronger audience response.",
        "Business Implication": f"Prioritize production toward '{top[cat_col]}'-style content.",
        "Caveat": "Association, not causation — audience composition and category may be confounded."
    }

def device_insight(dev_summary: pd.DataFrame, dev_col: str) -> dict:
    if dev_summary.empty: return {}
    top = dev_summary.iloc[0]
    return {
         "Finding": f"Posts published via '{top[dev_col]}' show highest avg engagement ({top['avg_engagement']:.4f}).",
         "Evidence": "Computed from grouped mean engagement_rate by device.",
         "Interpretation": "Device is functioning as a proxy for platform/channel.",
         "Business Implication": "Treat this as a directional signal about the posting channel.",
         "Caveat": "No platform variable exists — this is limited to device/channel."
    }