"""
tests/test_pipeline.py
=======================
End-to-end smoke test of the full analytical pipeline, now driven by the
single consolidated src/core.py module (was previously spread across
src/data_loader.py, preprocessing.py, feature_engineering.py, modeling.py,
evaluation.py, explainability.py, prediction.py, optimization.py,
recommendations.py, eda.py, nlp_analysis.py).

Run from the project root:  python tests/test_pipeline.py
"""
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src import core

t0 = time.time()
train_raw = core.load_dataset("data/training_data.csv")
pred_raw = core.load_dataset("data/prediction_data.csv")
print("loaded", train_raw.shape, pred_raw.shape)

mapping = core.build_column_mapping(train_raw)
print("mapping sample:", {k: v for k, v in list(mapping.items())[:8]})

report = core.data_quality_report(train_raw)
print("quality:", {k: v for k, v in report.items() if k not in ("missing_table", "column_types")})

train_clean, rep1 = core.clean_dataset(train_raw, mapping)
pred_clean, rep2 = core.clean_dataset(pred_raw, mapping)
print("clean steps:", rep1["steps"])

# NOTE: engineer_features() is disk-cached (joblib.Memory, see core.py) — the
# first call over the real dataset does the (slow) TextBlob sentiment scoring
# once; subsequent calls / test runs reuse the cached result instantly.
train_fe = core.engineer_features(train_clean, mapping)
pred_fe = core.engineer_features(pred_clean, mapping)
print("engineered cols added:", [c for c in train_fe.columns if c not in train_clean.columns])

eng_col = mapping["engagement_rate"]
train_t, (pred_t,), threshold = core.build_target(train_fe, [pred_fe], eng_col)
print("threshold:", threshold, "train positive rate:", train_t["high_engagement"].mean())

leak_table = core.classify_variable_roles(train_t.columns.tolist(), mapping)
print(leak_table.head(10))

feature_cols = core.get_pre_publication_feature_columns(train_t, mapping)
print("feature cols:", feature_cols)

X = train_t[feature_cols]
y = train_t["high_engagement"]
X_train, X_val, y_train, y_val = core.split_train_validation(X, y)
print("split sizes:", X_train.shape, X_val.shape)

# Fast path (default in the app too): tune=False -> single fixed-hyperparameter
# fit per model, no GridSearchCV, capped worker count (core.SAFE_N_JOBS).
# Set TUNE=1 in the environment to smoke-test the GridSearchCV path instead.
TUNE = os.environ.get("TUNE", "0") == "1"
results = core.train_all_models(X_train, y_train, tune=TUNE, cv_folds=2)
print("trained models:", list(results.keys()))

table, eval_cache = core.build_comparison_table(results, X_val, y_val)
print(table)

best_name = core.select_best_model(table)
print("best:", best_name)

best_model = results[best_name]["model"]
thresh_table = core.select_threshold(y_val, eval_cache[best_name]["y_proba"])
best_thresh = core.best_threshold_by_f1(thresh_table)
print("best threshold:", best_thresh)

# refit best model on full training data
final_model, secs, params = core.train_one_model(best_name, X, y, tune=TUNE, cv_folds=2)
print("refit done in", secs)

X_pred = pred_t[feature_cols]
pred_out = core.predict_on_new_data(final_model, X_pred, best_thresh)
print(pred_out.head())

if "high_engagement" in pred_t.columns:
    oos = core.evaluate_out_of_sample(
        pred_t["high_engagement"], pred_out["predicted_high_engagement"], pred_out["probability_high_engagement"]
    )
    print("out of sample:", oos)

final_output = core.assemble_prediction_output(pred_raw, pred_t, pred_out)
print(final_output.head(3))

# explainability - only if best model is tree based
if best_name in ("Random Forest", "XGBoost"):
    shap_values, feat_names, X_trans, X_samp = core.explain_tree_model(final_model, X_pred, max_rows=200)
    imp_table = core.global_importance_table(shap_values, feat_names)
    print(imp_table.head(10))
    single = core.explain_single_prediction(shap_values, feat_names, 0)
    print(single)
else:
    coef_table = core.logistic_coefficient_importance(final_model)
    print(coef_table.head(10))

# optimization
scenario_df = core.generate_scenarios(train_t, feature_cols, max_combinations=50)
scored = core.score_scenarios(final_model, scenario_df)
print(scored.head(5))

# eda
cat_col = mapping["category"]
cat_summary = core.group_engagement(train_t, cat_col, eng_col)
print(cat_summary.head())
ins = core.category_insight(cat_summary, cat_col)
print(ins)

print("ALL OK in", time.time() - t0, "seconds")
