import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import core 

st.set_page_config(page_title="AI Social Media Optimizer", layout="wide", initial_sidebar_state="expanded")

# ==============================================================================
# STRICT ISOLATION LAZY LOADERS
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_train_data():
    return core.load_dataset("data/training_data.csv")

@st.cache_data(show_spinner=False)
def load_prediction_data():
    return core.load_dataset("data/prediction_data.csv")

@st.cache_data
def prep_train_data(train_raw, mapping):
    """Processes ONLY the training dataset. Prediction dataset is strictly isolated."""
    tc, _ = core.clean_dataset(train_raw, mapping)
    tf = core.engineer_features(tc, mapping)
    
    ec = mapping.get("engagement_rate")
    th = None
    if ec in tf.columns:
        tf, th = core.build_target(tf, ec)
        
    fc = core.get_feature_columns(tf, mapping)
    return tf, fc, th, ec

@st.cache_resource
def run_models(tf, fc, choices):
    """Trains models purely on the processed training framework."""
    X, y = tf[fc], tf["high_engagement"]
    X_tr, X_va, y_tr, y_va = core.split_train_validation(X, y)
    res = core.train_all_models(X_tr, y_tr, choices)
    comp, cache = core.evaluate(res, X_va, y_va)
    best = comp.iloc[0]["Model"]
    th = core.get_best_threshold(y_va, cache[best]["proba"])
    
    # Refit final model
    d = core.MODEL_DEFS[best]
    pipe = core.Pipeline([
        ("prep", core.ColumnTransformer([
            ("num", core.StandardScaler() if d["scale"] else "passthrough", X.select_dtypes(include=[np.number, "bool"]).columns.tolist()),
            ("cat", core.OneHotEncoder(handle_unknown="ignore", sparse_output=False), [c for c in X.columns if c not in X.select_dtypes(include=[np.number]).columns])
        ])),
        ("clf", d["est"])
    ])
    pipe.fit(X, y)
    return {"best": best, "model": pipe, "comp": comp, "thresh": th, "X_va": X_va, "y_va": y_va, "cache": cache}

# ==============================================================================
# UI STRUCTURE (Lazy execution protects startup rendering)
# ==============================================================================
st.sidebar.title("📊 Optimization System")
PAGES = [
    "1. Overview", "2. Quality", "3. Descriptive", "4. NLP", "5. Features", 
    "6. Modeling", "7. Explainability", "8. Prediction", "9. Optimization", "10. Recommendations"
]
page = st.sidebar.radio("Navigate", PAGES)
choices = st.sidebar.multiselect("Models", ["Logistic Regression", "Random Forest", "XGBoost", "SVM"], default=["Logistic Regression", "Random Forest", "XGBoost", "SVM"])

# Instant load for metadata, no computation.
train_raw = load_train_data()
mapping = core.build_column_mapping(train_raw)

if page == PAGES[0]:
    st.title("AI Content Optimization (Executive Overview)")
    pred_meta = load_prediction_data()
    c1, c2, c3 = st.columns(3)
    c1.metric("Training Records", f"{len(train_raw):,}")
    c2.metric("Prediction Records (held-out)", f"{len(pred_meta):,}")
    c3.metric("Data Separation", "Strict Holdout")
    st.markdown("Use the sidebar to navigate to the respective ML analysis layers. Heavy processing runs lazily as you navigate.")
    
elif page == PAGES[1]:
    st.title("Data Quality & Discovery")
    st.write("Evaluating Training Dataset Quality:")
    st.dataframe(core.data_quality_report(train_raw)["missing_table"], use_container_width=True)

else:
    # Trigger feature engineering on train data ONLY if needed by page 3+
    with st.spinner("Processing training features..."):
        tf, fc, th, ec = prep_train_data(train_raw, mapping)
        
    if page == PAGES[2]:
        st.title("Descriptive Analytics")
        if ec:
            c1, c2 = st.columns(2)
            c1.plotly_chart(core.bar_chart(core.group_engagement(tf, mapping.get("category", "category"), ec).head(10), mapping.get("category", "category"), "avg_engagement", "Engagement by Category"), use_container_width=True)
            c2.plotly_chart(core.scatter_plot(tf, mapping.get("followers_count"), ec, "Followers vs Engagement", trendline=True), use_container_width=True)
            
    elif page == PAGES[3]:
        st.title("NLP & Sentiment Analytics")
        if "sentiment_label" in tf:
            st.plotly_chart(core.bar_chart(core.group_engagement(tf, "sentiment_label", ec), "sentiment_label", "avg_engagement", "Engagement by Sentiment"), use_container_width=True)
            st.dataframe(core.top_ngrams(tf[mapping["post_content"]]), use_container_width=True)

    elif page == PAGES[4]:
        st.title("Feature Engineering & Data Segregation")
        st.write("Ensuring post-publication labels don't leak into predictive variables.")
        st.dataframe(core.classify_variable_roles(tf.columns, mapping), use_container_width=True)

    # Trigger ML training ONLY on page 6+
    else:
        with st.spinner("Training ML Models on Training Data..."):
            mod = run_models(tf, fc, choices)
            
        if page == PAGES[5]:
            st.title("Model Training & Comparison")
            st.dataframe(mod["comp"], use_container_width=True)
            st.success(f"Best Model Selected via F1 Score: {mod['best']}")
            
            c1, c2 = st.columns(2)
            # Conf Matrix
            chosen = st.selectbox("View Confusion Matrix:", choices)
            if chosen in mod["cache"]:
                cm = confusion_matrix(mod["y_va"], mod["cache"][chosen]["preds"])
                fig = go.Figure(data=go.Heatmap(z=cm, x=["Low", "High"], y=["Low", "High"], colorscale="Blues", text=cm, texttemplate="%{text}"))
                c1.plotly_chart(fig, use_container_width=True)

        elif page == PAGES[6]:
            st.title("Explainable AI (SHAP / Coefs)")
            is_tree = mod["best"] in ["Random Forest", "XGBoost"]
            st.write(f"Explaining {mod['best']} predictions using {'TreeExplainer' if is_tree else 'Standardized Coefficients'}")
            st.dataframe(core.explain_model(mod["model"], mod["X_va"].sample(200, random_state=42), is_tree), use_container_width=True)

        elif page == PAGES[7]:
            st.title("Prediction on Unseen Records")
            st.warning("Prediction Dataset (5,000 records) is only processed now.")
            with st.spinner("Processing unseen prediction data..."):
                pred_raw = load_prediction_data()
                pc, _ = core.clean_dataset(pred_raw, mapping)
                pf = core.engineer_features(pc, mapping)
                
            out = core.predict_new(mod["model"], pf[fc], mod["thresh"], pred_raw)
            
            if ec and ec in pf.columns:
                st.subheader("Out-Of-Sample Validation")
                pf["high_engagement"] = (pf[ec] > th).astype(int)
                oos = core.evaluate_out_of_sample(pf["high_engagement"], out["predicted_high_engagement"], out["predicted_probability_high_engagement"])
                st.json(oos)
                
            st.dataframe(out.head(100), use_container_width=True)

        elif page == PAGES[8]:
            st.title("Content Optimization Matrix")
            with st.spinner("Scoring generated scenarios..."):
                scen = core.generate_scenarios(tf, fc)
                scored = core.score_scenarios(mod["model"], scen)
            st.dataframe(scored.head(25), use_container_width=True)

        elif page == PAGES[9]:
            st.title("Managerial Recommendations")
            cards = []
            if mapping.get("category"):
                cards.append(core.category_insight(core.group_engagement(tf, mapping.get("category"), ec), mapping.get("category")))
            if mapping.get("device"):
                cards.append(core.device_insight(core.group_engagement(tf, mapping.get("device"), ec), mapping.get("device")))
                
            for card in cards:
                if card:
                    with st.container(border=True):
                        st.markdown(f"Finding:")
                        st.markdown(f"Implication:")
                        st.caption(f"Caveat: {card['Caveat']}")