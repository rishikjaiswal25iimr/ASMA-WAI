import sys, time
sys.path.insert(0, "/home/claude/social-media-content-optimization")
import pandas as pd
from src import data_loader, preprocessing, feature_engineering, modeling, evaluation, explainability, prediction, optimization, recommendations, eda, nlp_analysis

t0 = time.time()
train_raw = data_loader.load_dataset("data/training_data.csv")
pred_raw = data_loader.load_dataset("data/prediction_data.csv")
print("loaded", train_raw.shape, pred_raw.shape)

mapping = data_loader.build_column_mapping(train_raw)
print("mapping sample:", {k: v for k, v in list(mapping.items())[:8]})

report = data_loader.data_quality_report(train_raw)
print("quality:", {k: v for k, v in report.items() if k not in ("missing_table", "column_types")})

train_clean, rep1 = preprocessing.clean_dataset(train_raw, mapping)
pred_clean, rep2 = preprocessing.clean_dataset(pred_raw, mapping)
print("clean steps:", rep1["steps"])

train_fe = feature_engineering.engineer_features(train_clean, mapping)
pred_fe = feature_engineering.engineer_features(pred_clean, mapping)
print("engineered cols added:", [c for c in train_fe.columns if c not in train_clean.columns])

eng_col = mapping["engagement_rate"]
train_t, (pred_t,), threshold = feature_engineering.build_target(train_fe, [pred_fe], eng_col)
print("threshold:", threshold, "train positive rate:", train_t["high_engagement"].mean())

leak_table = feature_engineering.classify_variable_roles(train_t.columns.tolist(), mapping)
print(leak_table.head(10))

feature_cols = feature_engineering.get_pre_publication_feature_columns(train_t, mapping)
print("feature cols:", feature_cols)

X = train_t[feature_cols]
y = train_t["high_engagement"]
X_train, X_val, y_train, y_val = modeling.split_train_validation(X, y)
print("split sizes:", X_train.shape, X_val.shape)

# quick test with reduced grids for speed - monkeypatch small grids
modeling.MODEL_DEFINITIONS["Logistic Regression"]["param_grid"] = {"clf__C": [1.0]}
modeling.MODEL_DEFINITIONS["Random Forest"]["param_grid"] = {"clf__n_estimators": [100], "clf__max_depth": [10]}
modeling.MODEL_DEFINITIONS["XGBoost"]["param_grid"] = {"clf__n_estimators": [100], "clf__max_depth": [4], "clf__learning_rate": [0.1]}
modeling.MODEL_DEFINITIONS["SVM"]["param_grid"] = {"clf__C": [1.0], "clf__kernel": ["rbf"]}

results = modeling.train_all_models(X_train, y_train)
print("trained models:", list(results.keys()))

table, eval_cache = evaluation.build_comparison_table(results, X_val, y_val)
print(table)

best_name = evaluation.select_best_model(table)
print("best:", best_name)

best_model = results[best_name]["model"]
thresh_table = evaluation.select_threshold(y_val, eval_cache[best_name]["y_proba"])
best_thresh = evaluation.best_threshold_by_f1(thresh_table)
print("best threshold:", best_thresh)

# refit best model on full training data
final_model, secs, params = modeling.train_one_model(best_name, X, y)
print("refit done in", secs)

X_pred = pred_t[feature_cols]
pred_out = prediction.predict_on_new_data(final_model, X_pred, best_thresh)
print(pred_out.head())

if "high_engagement" in pred_t.columns:
    oos = prediction.evaluate_out_of_sample(pred_t["high_engagement"], pred_out["predicted_high_engagement"], pred_out["probability_high_engagement"])
    print("out of sample:", oos)

final_output = prediction.assemble_prediction_output(pred_raw, pred_t, pred_out)
print(final_output.head(3))

# explainability - only if best model is tree based
if best_name in ("Random Forest", "XGBoost"):
    shap_values, feat_names, X_trans, X_samp = explainability.explain_tree_model(final_model, X_pred, max_rows=200)
    imp_table = explainability.global_importance_table(shap_values, feat_names)
    print(imp_table.head(10))
    single = explainability.explain_single_prediction(shap_values, feat_names, 0)
    print(single)
else:
    coef_table = explainability.logistic_coefficient_importance(final_model)
    print(coef_table.head(10))

# optimization
scenario_df = optimization.generate_scenarios(train_t, feature_cols, max_combinations=50)
scored = optimization.score_scenarios(final_model, scenario_df)
print(scored.head(5))

# eda
cat_col = mapping["category"]
cat_summary = eda.group_engagement(train_t, cat_col, eng_col)
print(cat_summary.head())
ins = recommendations.category_insight(cat_summary, cat_col)
print(ins)

print("ALL OK in", time.time() - t0, "seconds")
