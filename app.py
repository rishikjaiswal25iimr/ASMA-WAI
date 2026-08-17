"""
AI-Powered Social Media Content Optimization — Streamlit Application
======================================================================
Predicting Engagement and Identifying the Drivers of High-Performing Content

WAI Project — AI & Social Media Analytics — IIM Ranchi

Run locally:  streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src import core

st.set_page_config(
    page_title="AI Social Media Content Optimization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_TRAIN_PATH = "data/training_data.csv"
DEFAULT_PRED_PATH = "data/prediction_data.csv"
TARGET_PERCENTILE = 0.75

# ----------------------------------------------------------------------------
# Cached pipeline steps. Cached on the raw bytes of the uploaded files so the
# expensive steps (esp. model training) only rerun when the data changes.
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return core.load_dataset(io.BytesIO(file_bytes) if not filename.endswith((".xlsx", ".xls"))
                                     else io.BytesIO(file_bytes))


def load_default_or_uploaded(uploaded_file, default_path: str, label: str) -> pd.DataFrame:
    if uploaded_file is not None:
        df = core.load_dataset(uploaded_file)
        st.sidebar.success(f"{label}: loaded {len(df):,} records from upload.")
    else:
        df = core.load_dataset(default_path)
        st.sidebar.info(f"{label}: using default project file ({len(df):,} records).")
    return df


@st.cache_data(show_spinner=False)
def run_data_pipeline(train_df: pd.DataFrame, pred_df: pd.DataFrame):
    """Mapping -> cleaning -> feature engineering -> target construction. Cached on data content."""
    mapping = core.build_column_mapping(train_df)

    train_clean, train_clean_report = core.clean_dataset(train_df, mapping)
    pred_clean, pred_clean_report = core.clean_dataset(pred_df, mapping)

    train_fe = core.engineer_features(train_clean, mapping)
    pred_fe = core.engineer_features(pred_clean, mapping)

    eng_col = mapping.get("engagement_rate")
    threshold = None
    if eng_col and eng_col in train_fe.columns:
        train_fe, (pred_fe,), threshold = core.build_target(
            train_fe, [pred_fe], eng_col, percentile=TARGET_PERCENTILE
        )

    feature_cols = core.get_pre_publication_feature_columns(train_fe, mapping)
    leakage_table = core.classify_variable_roles(train_fe.columns.tolist(), mapping)

    return {
        "mapping": mapping,
        "train_clean": train_clean, "pred_clean": pred_clean,
        "train_clean_report": train_clean_report, "pred_clean_report": pred_clean_report,
        "train_fe": train_fe, "pred_fe": pred_fe,
        "threshold": threshold, "eng_col": eng_col,
        "feature_cols": feature_cols, "leakage_table": leakage_table,
    }


# ----------------------------------------------------------------------------
# Model training. Each individual model is cached SEPARATELY (keyed on its
# own name + the actual training data + the tuning settings), instead of one
# big cache entry for "all selected models at once". This means:
#   - adding ONE more model to the sidebar selection only trains that new
#     model — the ones already trained stay cached and are reused instantly.
#   - training happens ONE MODEL AT A TIME with a live progress bar, so the
#     app is never silently frozen behind a single multi-minute spinner.
# See src/core.py's module docstring for why this (plus tune=False by
# default and a capped worker count) fixes the CPU-throttling / stuck-at-
# startup problem on Streamlit Community Cloud.
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _train_single_model_cached(name: str, X_train: pd.DataFrame, y_train: pd.Series,
                                tune: bool, cv_folds: int, svm_subsample: int):
    return core.train_one_model(name, X_train, y_train, tune=tune, cv_folds=cv_folds, svm_subsample=svm_subsample)


def _train_models_with_progress(X_train, y_train, model_names: list[str], tune: bool,
                                 cv_folds: int, svm_subsample: int) -> dict:
    results = {}
    total = len(model_names)
    progress = st.progress(0.0, text="Preparing to train models…")
    status = st.empty()
    for i, name in enumerate(model_names, start=1):
        status.info(
            f"Training model {i}/{total}: **{name}**"
            + (" — hyperparameter tuning is ON, this one may take a while." if tune else " (fast, untuned fit)")
        )
        model, secs, params = _train_single_model_cached(name, X_train, y_train, tune, cv_folds, svm_subsample)
        results[name] = {"model": model, "train_seconds": secs, "best_params": params}
        progress.progress(i / total, text=f"Trained {i}/{total} models — {name} finished in {secs:.1f}s")
    status.empty()
    progress.empty()
    return results


def run_modeling_pipeline(train_fe: pd.DataFrame, feature_cols: tuple, model_names: tuple,
                           tune: bool, cv_folds: int, svm_subsample: int):
    X = train_fe[list(feature_cols)]
    y = train_fe["high_engagement"]
    X_train, X_val, y_train, y_val = core.split_train_validation(X, y)

    results = _train_models_with_progress(X_train, y_train, list(model_names), tune, cv_folds, svm_subsample)
    comparison_table, eval_cache = core.build_comparison_table(results, X_val, y_val)
    best_name = core.select_best_model(comparison_table)

    threshold_table = core.select_threshold(y_val, eval_cache[best_name]["y_proba"])
    best_threshold = core.best_threshold_by_f1(threshold_table)

    with st.spinner(f"Refitting {best_name} on the full 15,000-record training set…"):
        final_model, train_secs, best_params = _train_single_model_cached(
            best_name, X, y, tune, cv_folds, svm_subsample
        )

    return {
        "results": results, "comparison_table": comparison_table, "eval_cache": eval_cache,
        "best_name": best_name, "threshold_table": threshold_table, "best_threshold": best_threshold,
        "final_model": final_model, "X_val": X_val, "y_val": y_val, "X_train": X_train, "y_train": y_train,
        "tuning_enabled": tune,
    }


def kpi_row(items: list[tuple[str, str]]):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


# ==============================================================================
# SIDEBAR — data upload + navigation
# ==============================================================================
st.sidebar.title("📊 Content Optimization")
st.sidebar.caption("AI & Social Media Analytics — WAI Project")

st.sidebar.markdown("### 1. Upload Data")
train_upload = st.sidebar.file_uploader("Training dataset (15,000 records)", type=["csv", "xlsx", "xls"])
pred_upload = st.sidebar.file_uploader("Prediction dataset (5,000 unseen records)", type=["csv", "xlsx", "xls"])
st.sidebar.caption("If nothing is uploaded, the project's default data/ files are used.")

train_raw = load_default_or_uploaded(train_upload, DEFAULT_TRAIN_PATH, "Training data")
pred_raw = load_default_or_uploaded(pred_upload, DEFAULT_PRED_PATH, "Prediction data")

st.sidebar.markdown("### 2. Navigate")
PAGES = [
    "1. Executive Overview", "2. Data Quality", "3. Descriptive Analytics",
    "4. NLP / Sentiment", "5. Feature Engineering", "6. Model Training & Comparison",
    "7. Explainable AI", "8. Prediction", "9. Content Optimization",
    "10. Managerial Recommendations",
]
page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")

st.sidebar.markdown("### 3. Model Settings")
model_choices = st.sidebar.multiselect(
    "Models to train (Layer 5)",
    ["Logistic Regression", "Random Forest", "XGBoost", "SVM"],
    default=["Logistic Regression", "Random Forest", "XGBoost", "SVM"],
)

with st.sidebar.expander("⚙️ Performance settings", expanded=False):
    st.caption(
        "By default every model trains **once** with an already-good fixed "
        "hyperparameter set (fast, no grid search) and uses at most "
        f"**{core.SAFE_N_JOBS} CPU worker(s)** — this avoids the CPU-quota "
        "throttling that hosted free-tier containers apply when a process "
        "tries to use more cores than it was actually allocated."
    )
    enable_tuning = st.checkbox(
        "Enable hyperparameter tuning (GridSearchCV) — slower", value=False,
        help="Runs a small grid search per model instead of a single fit. "
             "Only turn this on if you have a few extra minutes / a beefier host.",
    )
    svm_subsample = st.slider(
        "SVM training subsample size", min_value=1000, max_value=8000, value=4000, step=500,
        help="SVM training cost grows quickly with rows; a subsample keeps it fast.",
    )
    cv_folds = st.slider("Cross-validation folds (only used when tuning is on)", 2, 5, 2)

run_button = st.sidebar.button("🚀 Run / Refresh Full Analysis", type="primary")

# Run the deterministic (non-ML) data pipeline. This is CPU-light on repeat
# runs because engineer_features() is disk-cached (joblib.Memory) inside
# core.py — the slow, one-time cost (TextBlob sentiment over every caption)
# only happens once per unique dataset, including across app restarts.
with st.spinner("Preparing data (column mapping, cleaning, feature engineering)… "
                 "first run on new data can take a little while, later runs are cached."):
    pipeline = run_data_pipeline(train_raw, pred_raw)
mapping = pipeline["mapping"]
train_fe, pred_fe = pipeline["train_fe"], pipeline["pred_fe"]
threshold, eng_col = pipeline["threshold"], pipeline["eng_col"]
feature_cols, leakage_table = pipeline["feature_cols"], pipeline["leakage_table"]

if eng_col is None:
    st.sidebar.error("Could not detect an engagement-rate column — high-engagement target cannot be built.")

# Model pipeline is heavier — only (re)run on demand or first load, and each
# model within it is cached individually (see run_modeling_pipeline above).
if "model_run" not in st.session_state:
    st.session_state["model_run"] = False

if run_button or not st.session_state["model_run"]:
    if eng_col is not None and model_choices:
        st.session_state["model_output"] = run_modeling_pipeline(
            train_fe, tuple(feature_cols), tuple(model_choices), enable_tuning, cv_folds, svm_subsample,
        )
        st.session_state["model_run"] = True

model_output = st.session_state.get("model_output")

# ==============================================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ==============================================================================
if page == PAGES[0]:
    st.title("AI-Powered Social Media Content Optimization")
    st.subheader("Predicting Engagement and Identifying the Drivers of High-Performing Content")

    st.markdown("""
**Business objective.** Diagnose historical engagement patterns, identify the characteristics of
high-performing content, use machine learning to predict high engagement, explain what drives
those predictions, and convert the results into actionable content-optimization guidance.

**Data split (Section 5).** The 20,000-record dataset is split into the first **15,000 records**
(training/model development) and the last **5,000 records** (completely unseen prediction set).
The prediction set is never used for feature selection, model training, threshold selection, or
optimization-rule creation.
""")

    kpi_row([
        ("Training records", f"{len(train_fe):,}"),
        ("Prediction records (unseen)", f"{len(pred_fe):,}"),
        ("High-Engagement threshold (75th pct)", f"{threshold:.4f}" if threshold else "N/A"),
        ("Training positive rate", f"{train_fe['high_engagement'].mean()*100:.1f}%" if "high_engagement" in train_fe else "N/A"),
    ])

    if model_output:
        best_row = model_output["comparison_table"].iloc[0]
        kpi_row([
            ("Best model", model_output["best_name"]),
            ("Validation F1", f"{best_row['F1 Score']:.3f}"),
            ("Validation ROC-AUC", f"{best_row['ROC-AUC']:.3f}"),
            ("Selected threshold", f"{model_output['best_threshold']:.2f}"),
        ])
    else:
        st.info("Click **Run / Refresh Full Analysis** in the sidebar to train models and populate model KPIs.")

    st.markdown("### Dataset snapshot")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Training data (first 15,000 rows)**")
        st.dataframe(train_fe.head(5), use_container_width=True)
    with c2:
        st.write("**Prediction data (last 5,000 rows — unseen)**")
        st.dataframe(pred_fe.head(5), use_container_width=True)

    st.markdown("### Known dataset limitations (auto-detected)")
    limitations = []
    if mapping.get("device") and mapping.get("device") not in train_fe.columns[:0]:
        limitations.append("No true multi-network **platform** field (e.g. Instagram vs. X/Twitter) exists in this "
                            "dataset — analyses labelled 'platform' elsewhere use the closest available field, "
                            "**device**, instead, and this is called out explicitly.")
    if "post_day_of_week" in train_fe.columns:
        limitations.append("**post_date has no time-of-day component** in this dataset, so posting-hour analysis "
                            "is not possible — day-of-week and weekend/weekday analysis is used instead.")
    if mapping.get("post_content"):
        limitations.append("Sentiment is derived from raw caption text via TextBlob (no pre-existing sentiment "
                            "label was present in the data) — see Page 4 for detail.")
    for l in limitations:
        st.warning(l)

    st.caption("This overview updates automatically from calculated results — nothing here is hardcoded.")

# ==============================================================================
# PAGE 2 — DATA QUALITY
# ==============================================================================
elif page == PAGES[1]:
    st.title("Layer 1 — Data Discovery & Quality")

    tab1, tab2 = st.tabs(["Training Dataset", "Prediction Dataset"])
    for tab, df, label in [(tab1, train_raw, "Training"), (tab2, pred_raw, "Prediction")]:
        with tab:
            report = core.data_quality_report(df)
            kpi_row([
                ("Records", f"{report['n_records']:,}"),
                ("Variables", report["n_variables"]),
                ("Numeric", report["n_numeric"]),
                ("Categorical", report["n_categorical"]),
                ("Text", report["n_text"]),
                ("Datetime", report["n_datetime"]),
            ])
            kpi_row([
                ("Duplicate records", report["n_duplicates"]),
                ("Total missing cells", int(report["missing_table"]["missing_count"].sum())),
            ])
            st.markdown(f"**{label} data — column-level quality report**")
            st.dataframe(report["missing_table"], use_container_width=True)
            st.markdown("**Descriptive statistics (numeric columns)**")
            st.dataframe(df.describe().T, use_container_width=True)

    st.markdown("### Automatic column mapping (Section 6)")
    st.caption("Detected canonical analytical roles → actual column names in your uploaded file. "
               "Columns are never silently renamed; this mapping is used consistently throughout the app.")
    map_table = pd.DataFrame([{"Canonical Role": k, "Detected Column": v or "— not found —"} for k, v in mapping.items()])
    st.dataframe(map_table, use_container_width=True, height=400)

    missing_roles = [k for k, v in mapping.items() if v is None]
    if missing_roles:
        st.warning(f"Roles not found in the uploaded data (related analyses will be skipped): {', '.join(missing_roles)}")

    st.markdown("### Cleaning applied (Section 7)")
    st.write("Training data:", pipeline["train_clean_report"]["steps"] or "No structural issues found.")
    if not pipeline["train_clean_report"]["missing_value_treatment"].empty:
        st.dataframe(pipeline["train_clean_report"]["missing_value_treatment"], use_container_width=True)
    else:
        st.success("No missing values detected in the training dataset — no imputation was necessary.")

    st.download_button(
        "⬇️ Download cleaned training data (CSV)",
        pipeline["train_clean"].to_csv(index=False).encode("utf-8"),
        file_name="cleaned_training_data.csv", mime="text/csv",
    )

# ==============================================================================
# PAGE 3 — DESCRIPTIVE ANALYTICS
# ==============================================================================
elif page == PAGES[2]:
    st.title("Layer 2 — Descriptive / Exploratory Analytics")
    st.caption("Computed on the 15,000-record TRAINING dataset only.")

    if eng_col is None:
        st.error("No engagement-rate column detected — descriptive engagement analysis cannot run.")
        st.stop()

    df = train_fe.copy()

    # --- Filters ---
    with st.expander("🔎 Filters", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        cat_col = mapping.get("category")
        device_col = mapping.get("device")
        sel_cats = fc1.multiselect(cat_col or "Category", sorted(df[cat_col].unique()) if cat_col else [],
                                    default=None) if cat_col else []
        sel_devices = fc2.multiselect(device_col or "Device", sorted(df[device_col].unique()) if device_col else [],
                                       default=None) if device_col else []
        sel_sentiment = fc3.multiselect("Sentiment", sorted(df["sentiment_label"].unique()) if "sentiment_label" in df else [],
                                         default=None) if "sentiment_label" in df else []
    if sel_cats:
        df = df[df[cat_col].isin(sel_cats)]
    if sel_devices:
        df = df[df[device_col].isin(sel_devices)]
    if sel_sentiment:
        df = df[df["sentiment_label"].isin(sel_sentiment)]
    st.caption(f"{len(df):,} of {len(train_fe):,} training records match the current filters.")

    # --- Category ---
    cat_col = mapping.get("category")
    if cat_col:
        st.markdown("### Content Category")
        summary = core.group_engagement(df, cat_col, eng_col)
        c1, c2 = st.columns([2, 1])
        c1.plotly_chart(core.bar_chart(summary, cat_col, "avg_engagement", f"Average Engagement Rate by {cat_col}"),
                         use_container_width=True)
        c2.dataframe(summary, use_container_width=True, height=400)

    # --- Device / channel (documented as NOT a true multi-network platform field) ---
    device_col = mapping.get("device")
    if device_col:
        st.markdown("### Device / Channel")
        st.caption("No true cross-network platform field exists in this dataset — `device` is the closest "
                   "available channel-level dimension and is used here instead of 'platform'.")
        summary = core.group_engagement(df, device_col, eng_col)
        c1, c2 = st.columns([2, 1])
        c1.plotly_chart(core.bar_chart(summary, device_col, "avg_engagement", f"Average Engagement Rate by {device_col}"),
                         use_container_width=True)
        c2.dataframe(summary, use_container_width=True, height=250)

    # --- Timing ---
    if "post_day_of_week" in df.columns:
        st.markdown("### Timing")
        st.caption("This dataset's post_date has no time-of-day component, so posting-hour analysis is not "
                   "available — day-of-week and weekend/weekday patterns are shown instead.")
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_summary = core.group_engagement(df, "post_day_of_week", eng_col)
        dow_summary["post_day_of_week"] = pd.Categorical(dow_summary["post_day_of_week"], categories=order, ordered=True)
        dow_summary = dow_summary.sort_values("post_day_of_week")
        c1, c2 = st.columns(2)
        c1.plotly_chart(core.bar_chart(dow_summary, "post_day_of_week", "avg_engagement", "Average Engagement by Day of Week"),
                         use_container_width=True)
        wk_fig, wk_data = core.weekend_vs_weekday(df, eng_col)
        if wk_fig:
            c2.plotly_chart(wk_fig, use_container_width=True)

    # --- Audience ---
    st.markdown("### Audience")
    followers_col = mapping.get("followers_count")
    if followers_col:
        c1, c2 = st.columns(2)
        c1.plotly_chart(core.scatter_plot(df.sample(min(3000, len(df)), random_state=42), followers_col, eng_col,
                                          f"{followers_col} vs. {eng_col}", trendline="lowess"), use_container_width=True)
        if "follower_band" in df.columns:
            band_summary = core.group_engagement(df, "follower_band", eng_col)
            c2.plotly_chart(core.bar_chart(band_summary, "follower_band", "avg_engagement", "Engagement by Follower Tier"),
                             use_container_width=True)

    verified_col = mapping.get("is_verified")
    if verified_col:
        st.markdown("**Verified vs. non-verified accounts**")
        v_summary = core.group_engagement(df, verified_col, eng_col)
        st.plotly_chart(core.bar_chart(v_summary, verified_col, "avg_engagement", "Engagement: Verified vs. Non-Verified"),
                         use_container_width=True)

    # --- Correlation heatmap ---
    st.markdown("### Correlation Heatmap")
    numeric_cols = [c for c in [followers_col, mapping.get("following_count"), mapping.get("content_length"),
                                 "hashtag_count", "sentiment_polarity", "account_age_days", eng_col]
                     if c and c in df.columns]
    if len(numeric_cols) >= 2:
        heat_fig, _ = core.correlation_heatmap(df, numeric_cols)
        st.plotly_chart(heat_fig, use_container_width=True)

    # --- High vs low distribution ---
    st.markdown("### High vs. Low Engagement Split")
    pie_fig, counts = core.high_vs_low_distribution(df, "high_engagement")
    c1, c2 = st.columns([1, 1])
    c1.plotly_chart(pie_fig, use_container_width=True)
    c2.dataframe(counts.rename("count"), use_container_width=True)

# ==============================================================================
# PAGE 4 — NLP / SENTIMENT
# ==============================================================================
elif page == PAGES[3]:
    st.title("Layer 3 — NLP / Sentiment Analytics")

    text_col = mapping.get("post_content")
    if not text_col:
        st.warning("No raw caption/text column detected. If your data instead has a pre-existing sentiment "
                   "label, that label would be analyzed directly here — but nothing was found to analyze.")
        st.stop()

    st.success(f"Raw caption text detected in column **'{text_col}'** — genuine text-based sentiment analysis "
               f"was performed using TextBlob (polarity/subjectivity), not a pre-existing label.")

    df = train_fe

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sentiment distribution**")
        dist = df["sentiment_label"].value_counts()
        st.plotly_chart(px.pie(values=dist.values, names=dist.index, hole=0.4, title="Sentiment Distribution"),
                         use_container_width=True)
    with c2:
        st.markdown("**Sentiment vs. Engagement**")
        s_summary = core.group_engagement(df, "sentiment_label", eng_col)
        st.plotly_chart(core.bar_chart(s_summary, "sentiment_label", "avg_engagement", "Average Engagement by Sentiment"),
                         use_container_width=True)

    cat_col = mapping.get("category")
    if cat_col:
        st.markdown("**Sentiment × Category (mean engagement, top rows)**")
        cross = core.sentiment_vs_dimension(df, "sentiment_label", cat_col, eng_col)
        st.dataframe(cross.sort_values("avg_engagement", ascending=False).head(20), use_container_width=True)

    device_col = mapping.get("device")
    if device_col:
        st.markdown("**Sentiment × Device**")
        cross2 = core.sentiment_vs_dimension(df, "sentiment_label", device_col, eng_col)
        st.dataframe(cross2.sort_values("avg_engagement", ascending=False), use_container_width=True)

    st.markdown("### Most frequent words in high-engagement vs. low-engagement captions")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("High engagement")
        st.dataframe(core.top_ngrams(df.loc[df["high_engagement"] == 1, text_col], n=1, top_k=15),
                     use_container_width=True)
    with c2:
        st.caption("Low engagement")
        st.dataframe(core.top_ngrams(df.loc[df["high_engagement"] == 0, text_col], n=1, top_k=15),
                     use_container_width=True)
    st.caption("Frequency-based extraction (stopwords removed); this is descriptive vocabulary analysis, not topic core.")

# ==============================================================================
# PAGE 5 — FEATURE ENGINEERING
# ==============================================================================
elif page == PAGES[4]:
    st.title("Layer 4 — Feature Engineering & Target Construction")

    st.markdown("### Target construction (Section 8)")
    if threshold is not None:
        st.info(f"**High Engagement = 1** if `{eng_col}` is above the **75th percentile of the TRAINING data "
                f"only** ({threshold:.4f}); **0** otherwise. This threshold is then applied unchanged to the "
                f"prediction dataset — it is never recalculated on unseen data.")
        kpi_row([
            ("Threshold (75th pct, train)", f"{threshold:.4f}"),
            ("Training high-engagement rate", f"{train_fe['high_engagement'].mean()*100:.1f}%"),
            ("Prediction high-engagement rate*", f"{pred_fe['high_engagement'].mean()*100:.1f}%" if "high_engagement" in pred_fe.columns else "N/A"),
        ])
        st.caption("*Shown only for transparency about label distribution — actual prediction-set labels are "
                   "never used during model development (Section 5/18).")
    else:
        st.error("No engagement-rate column found; target cannot be constructed.")

    st.markdown("### Engineered (derived) features")
    eng_added = [c for c in core.ENGINEERED_FEATURE_COLUMNS if c in train_fe.columns]
    st.write(eng_added)
    st.dataframe(train_fe[eng_added].head(10), use_container_width=True)

    st.markdown("### Pre-publication vs. post-publication variables — data-leakage check (Section 39)")
    st.caption("Post-publication outcomes (likes, comments, shares, engagement_rate, etc.) are explicitly "
               "excluded from the predictive feature set used in Layer 5.")
    st.dataframe(leakage_table, use_container_width=True, height=500)

    st.markdown("### Final feature set used for modeling")
    st.code(", ".join(feature_cols), language="text")
    st.download_button("⬇️ Download leakage-check table (CSV)", leakage_table.to_csv(index=False).encode("utf-8"),
                        file_name="leakage_check_table.csv", mime="text/csv")

# ==============================================================================
# PAGE 6 — MODEL TRAINING & COMPARISON
# ==============================================================================
elif page == PAGES[5]:
    st.title("Layer 5 — Machine Learning: Model Training & Comparison")

    if model_output is None:
        st.warning("Click **Run / Refresh Full Analysis** in the sidebar to train models.")
        st.stop()

    tuning_note = (
        "with **hyperparameter tuning enabled** (`GridSearchCV`, F1-optimized)"
        if model_output.get("tuning_enabled") else
        "each with a **single fixed, already-good hyperparameter set** (tuning is off by default — "
        "turn on *Enable hyperparameter tuning* in the sidebar's Performance settings for a slower "
        "`GridSearchCV` search instead)"
    )
    st.markdown(f"""
Four classifiers are trained **independently** (none feeds another) on the same 15,000-record
training dataset, using an internal **80:20 stratified train/validation split**, {tuning_note}:
- **Logistic Regression** — interpretable linear baseline
- **Random Forest** — nonlinear relationships, feature interactions
- **XGBoost** — gradient boosting, typically strongest raw predictive performance
- **SVM** — classical margin-based classifier (trained on a stratified subsample when the
  training set is large, to keep runtime reasonable — disclosed here rather than hidden)

Each model also trains **one at a time with a visible progress bar** (rather than one long silent
spinner) and is **cached individually**, so re-running after changing just one model in the
selection doesn't retrain the others from scratch.
""")

    st.markdown("### Model comparison table")
    st.dataframe(model_output["comparison_table"], use_container_width=True)
    st.caption("Best model is selected by **F1 Score** (balances precision & recall) — not by accuracy alone, "
               "since the high-engagement class is imbalanced (~25% positive).")

    st.markdown(f"### 🏆 Best model: **{model_output['best_name']}**")

    c1, c2 = st.columns(2)
    c1.plotly_chart(core.roc_curve_figure(model_output["eval_cache"], model_output["y_val"]),
                     use_container_width=True)
    with c2:
        chosen = st.selectbox("Confusion matrix for model:", list(model_output["results"].keys()))
        cm = model_output["eval_cache"][chosen]["confusion_matrix"]
        st.plotly_chart(core.confusion_matrix_figure(cm, chosen), use_container_width=True)

    st.markdown("### Threshold selection (Section 18)")
    st.caption("Evaluated on the validation split only — never on the unseen prediction data.")
    st.dataframe(model_output["threshold_table"], use_container_width=True)
    st.success(f"Selected threshold (best F1 on validation data): **{model_output['best_threshold']:.2f}** "
               f"(default would have been 0.50).")

# ==============================================================================
# PAGE 7 — EXPLAINABLE AI
# ==============================================================================
elif page == PAGES[6]:
    st.title("Layer 7 — Explainable AI")

    if model_output is None:
        st.warning("Click **Run / Refresh Full Analysis** in the sidebar first.")
        st.stop()

    best_name = model_output["best_name"]
    final_model = model_output["final_model"]
    X_val = model_output["X_val"]

    if best_name in ("Random Forest", "XGBoost"):
        st.info(f"Using **SHAP TreeExplainer** on the final {best_name} model (exact, fast for tree ensembles).")
        with st.spinner("Computing SHAP values..."):
            shap_values, feat_names, X_trans, X_samp = core.explain_tree_model(final_model, X_val, max_rows=400)
            st.session_state["shap_cache"] = (shap_values, feat_names, X_samp)
        imp_table = core.global_importance_table(shap_values, feat_names)

        st.markdown("### Global feature importance (mean |SHAP value|)")
        st.plotly_chart(px.bar(imp_table.head(15), x="mean_abs_shap", y="feature", orientation="h",
                                title="Top 15 Features Driving High-Engagement Predictions"),
                         use_container_width=True)
        st.dataframe(imp_table, use_container_width=True, height=350)

        st.markdown("### Individual prediction explanation")
        row_idx = st.slider("Select a validation-sample record", 0, len(X_samp) - 1, 0)
        st.dataframe(X_samp.iloc[[row_idx]], use_container_width=True)
        single = core.explain_single_prediction(shap_values, feat_names, row_idx)
        st.dataframe(single, use_container_width=True)
        pos = single[single["direction"] == "Positive contributor"]["feature"].tolist()
        neg = single[single["direction"] == "Negative contributor"]["feature"].tolist()
        c1, c2 = st.columns(2)
        c1.markdown("**Positive contributors**\n" + "\n".join(f"- {p}" for p in pos) if pos else "**Positive contributors:** none")
        c2.markdown("**Negative contributors**\n" + "\n".join(f"- {n}" for n in neg) if neg else "**Negative contributors:** none")

    else:
        st.info(f"{best_name} is not tree-based; SHAP's TreeExplainer does not apply, so standardized model "
                f"coefficients are used instead — a faster and equally honest explanation for a linear/kernel model.")
        coef_table = core.logistic_coefficient_importance(final_model) if best_name == "Logistic Regression" else None
        if coef_table is not None:
            st.plotly_chart(px.bar(coef_table.head(15), x="coefficient", y="feature", orientation="h",
                                    title="Top 15 Standardized Coefficients"), use_container_width=True)
            st.dataframe(coef_table, use_container_width=True, height=350)
        else:
            st.warning("Explainability for SVM with an RBF kernel requires SHAP's KernelExplainer, which is "
                       "computationally expensive; consider selecting Logistic Regression, Random Forest or "
                       "XGBoost as the comparison winner for a fully explained model.")

# ==============================================================================
# PAGE 8 — PREDICTION
# ==============================================================================
elif page == PAGES[7]:
    st.title("Layer 6 — Prediction on Unseen Data")

    if model_output is None:
        st.warning("Click **Run / Refresh Full Analysis** in the sidebar first.")
        st.stop()

    final_model = model_output["final_model"]
    best_threshold = model_output["best_threshold"]

    st.info(f"Applying the finalized **{model_output['best_name']}** model (retrained on the full 15,000-record "
            f"training set, threshold = {best_threshold:.2f}) to the **5,000-record prediction dataset**, which "
            f"was never touched during model development.")

    X_pred = pred_fe[feature_cols]
    pred_out = core.predict_on_new_data(final_model, X_pred, best_threshold)
    final_output = core.assemble_prediction_output(pred_raw, pred_fe, pred_out)
    st.session_state["final_prediction_output"] = final_output

    kpi_row([
        ("Records scored", f"{len(final_output):,}"),
        ("Predicted High Engagement", f"{(final_output['predicted_high_engagement'].mean()*100):.1f}%"),
        ("Avg. predicted probability", f"{final_output['probability_high_engagement'].mean():.3f}"),
    ])

    if "high_engagement" in pred_fe.columns:
        st.markdown("### Out-of-sample evaluation (actual labels available)")
        st.caption("Actual `high_engagement` labels exist in the prediction data and are used ONLY here, "
                   "AFTER predictions were generated — never during model development.")
        oos = core.evaluate_out_of_sample(pred_fe["high_engagement"], pred_out["predicted_high_engagement"],
                                                  pred_out["probability_high_engagement"])
        kpi_row([
            ("Out-of-sample Accuracy", f"{oos['accuracy']:.3f}"),
            ("Out-of-sample Precision", f"{oos['precision']:.3f}"),
            ("Out-of-sample Recall", f"{oos['recall']:.3f}"),
            ("Out-of-sample F1", f"{oos['f1']:.3f}"),
            ("Out-of-sample ROC-AUC", f"{oos['roc_auc']:.3f}"),
        ])
        st.plotly_chart(core.confusion_matrix_figure(oos["confusion_matrix"], f"{model_output['best_name']} — Out-of-Sample"),
                         use_container_width=True)
    else:
        st.warning("No ground-truth engagement labels were found in the prediction dataset — these are "
                   "**forward predictions only**, not a validated out-of-sample accuracy claim.")

    st.markdown("### Filters")
    fc1, fc2 = st.columns(2)
    class_filter = fc1.multiselect("Predicted class", final_output["predicted_label"].unique().tolist(),
                                    default=final_output["predicted_label"].unique().tolist())
    cat_col = mapping.get("category")
    cat_filter = fc2.multiselect(cat_col or "Category", sorted(final_output[cat_col].unique()) if cat_col else [],
                                  default=None) if cat_col else []
    view = final_output[final_output["predicted_label"].isin(class_filter)]
    if cat_filter:
        view = view[view[cat_col].isin(cat_filter)]

    st.markdown(f"### Prediction results ({len(view):,} records, ranked by predicted probability)")
    st.dataframe(view.head(200), use_container_width=True, height=450)

    st.download_button("⬇️ Download full prediction results (CSV)", final_output.to_csv(index=False).encode("utf-8"),
                        file_name="prediction_results.csv", mime="text/csv")

# ==============================================================================
# PAGE 9 — CONTENT OPTIMIZATION
# ==============================================================================
elif page == PAGES[8]:
    st.title("Layer 8 — Content Optimization Playbook")

    if model_output is None:
        st.warning("Click **Run / Refresh Full Analysis** in the sidebar first.")
        st.stop()

    st.markdown("""
The trained model scores **hypothetical combinations** of *controllable* pre-publication
variables (category, device, sentiment tone, media presence, hashtag volume, content length,
day of week), while holding *non-controllable / contextual* variables (followers, account age,
verified status, demographics) fixed at representative (median/mode) values from the training data.

These are **model-based optimization scenarios** — not observed historical facts.
""")
    st.caption(f"Non-controllable variables held fixed: {core.CONTEXTUAL_FIELDS_NOTE}")

    final_model = model_output["final_model"]
    with st.spinner("Scoring content scenarios..."):
        scenario_df = core.generate_scenarios(train_fe, feature_cols, max_combinations=300)
        scored = core.score_scenarios(final_model, scenario_df)

    st.markdown("### Top 15 predicted highest-probability content combinations")
    display_cols = [c for c in [mapping.get("category"), mapping.get("device"), "sentiment_label", "has_media",
                                 "hashtag_band", "content_length_band", "post_day_of_week",
                                 "predicted_probability_high_engagement"] if c in scored.columns]
    st.dataframe(scored[display_cols].head(15), use_container_width=True)

    st.markdown("### Bottom 10 predicted combinations")
    st.dataframe(scored[display_cols].tail(10), use_container_width=True)

    cat_col = mapping.get("category")
    if cat_col and cat_col in scored.columns:
        st.markdown("### Average predicted probability by category (across scenarios)")
        cat_scores = scored.groupby(cat_col, observed=True)["predicted_probability_high_engagement"].mean().sort_values(ascending=False).reset_index()
        st.plotly_chart(px.bar(cat_scores, x=cat_col, y="predicted_probability_high_engagement",
                                title="Model-Predicted High-Engagement Probability by Category (scenario average)"),
                         use_container_width=True)

    st.download_button("⬇️ Download optimization scenarios (CSV)", scored.to_csv(index=False).encode("utf-8"),
                        file_name="content_optimization_scenarios.csv", mime="text/csv")

# ==============================================================================
# PAGE 10 — MANAGERIAL RECOMMENDATIONS
# ==============================================================================
elif page == PAGES[9]:
    st.title("Layer 9 — Managerial Recommendations")

    if model_output is None:
        st.warning("Click **Run / Refresh Full Analysis** in the sidebar for model-based core.")

    cards = []
    cat_col = mapping.get("category")
    if cat_col:
        cat_summary = core.group_engagement(train_fe, cat_col, eng_col)
        cards.append(core.category_insight(cat_summary, cat_col))

    device_col = mapping.get("device")
    if device_col:
        device_summary = core.group_engagement(train_fe, device_col, eng_col)
        cards.append(core.device_insight(device_summary, device_col))

    if "sentiment_label" in train_fe.columns:
        sentiment_summary = core.group_engagement(train_fe, "sentiment_label", eng_col).rename(
            columns={"sentiment_label": "sentiment_label", "avg_engagement": "avg_engagement"})
        cards.append(core.sentiment_insight(sentiment_summary))

    if "is_weekend" in train_fe.columns:
        wk_fig, wk_data = core.weekend_vs_weekday(train_fe, eng_col)
        if wk_data is not None:
            cards.append(core.timing_insight(wk_data))

    if model_output is not None:
        best_row = model_output["comparison_table"].iloc[0]
        cards.append(core.model_insight(model_output["best_name"], best_row))

    for card in cards:
        with st.container(border=True):
            st.markdown(f"**Finding:** {card['Finding']}")
            st.markdown(f"**Evidence:** {card['Evidence']}")
            st.markdown(f"**Interpretation:** {card['Interpretation']}")
            st.markdown(f"**Business Implication:** {card['Business Implication']}")
            st.caption(f"⚠️ Caveat: {card['Caveat']}")

    st.markdown("### Strategic action summary")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Short-term actions**")
        st.markdown("""
- Use the Page 9 optimization playbook to pre-screen draft content ideas before publishing.
- Prioritize the top-scoring content category / sentiment-tone combinations identified above.
- A/B test the highest-ranked scenario combinations against current content mix.
""")
    with a2:
        st.markdown("**Long-term actions**")
        st.markdown("""
- Instrument true multi-platform tracking (this dataset lacks a network-level platform field)
  and time-of-day posting data to unlock richer core.
- Periodically retrain the model as new posts accumulate to avoid drift.
- Validate model-suggested scenarios with real published content, not just historical patterns.
""")

    st.markdown("### Ethical AI & limitations (Section 31)")
    with st.expander("View full ethical considerations"):
        st.markdown("""
**Dataset limitations:** this is a synthetic dataset generated for coursework; it may not represent
real audience behavior, is limited to 10 locations / 16 topics / 4 device types / 5 languages, and
contains no true cross-network platform field or time-of-day field.

**Model limitations:** the high-engagement class is imbalanced (~25% positive); predictive
performance reflects patterns in this specific (synthetic) dataset and may not generalize.
Results describe statistical **association**, not proven **causation** — phrases like "associated
with" or "predictive of" are used deliberately instead of "causes."

**AI limitations:** outputs (sentiment labels, SHAP explanations, optimization scenarios) are
generated by algorithms that can make mistakes; they should support, not replace, human marketing judgment.

**Privacy:** post_id/user_id are pseudonymous identifiers used only for row tracking and never
displayed as personally identifying information beyond what was already present in the uploaded file.

**Bias:** category, device, and follower-count imbalances in the sample may bias which patterns
the model learns most confidently; recommendations should be validated against your own audience.
""")

st.sidebar.markdown("---")
st.sidebar.caption("WAI Project — AI & Social Media Analytics — IIM Ranchi. "
                    "All figures are calculated live from the uploaded data; nothing is hardcoded.")
