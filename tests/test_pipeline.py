import sys
import time
import pandas as pd
sys.path.insert(0, ".")
import core

print("=== Running End-to-End Test Pipeline with Strict Data Separation ===")
t0 = time.time()

# 1. Load Training Data Only
print("1. Loading Training Data...")
train_raw = core.load_dataset("data/training_data.csv")
mapping = core.build_column_mapping(train_raw)

# 2. Process Training Data
print("2. Processing Training Data...")
train_clean, rep = core.clean_dataset(train_raw, mapping)
train_fe = core.engineer_features(train_clean, mapping)

eng_col = mapping.get("engagement_rate")
train_t, thresh = core.build_target(train_fe, eng_col)
feature_cols = core.get_feature_columns(train_t, mapping)

# 3. Model Training on Training Data
print("3. Training Models...")
X, y = train_t[feature_cols], train_t["high_engagement"]
X_tr, X_va, y_tr, y_va = core.split_train_validation(X, y)
res = core.train_all_models(X_tr, y_tr, ["Logistic Regression", "Random Forest"])
comp, cache = core.evaluate(res, X_va, y_va)
best_name = comp.iloc[0]["Model"]
best_thresh = core.get_best_threshold(y_va, cache[best_name]["proba"])

print(f"Best model determined strictly from validation split: {best_name}")

# 4. Strict Isolation Over: Evaluate Unseen Prediction Data
print("4. Evaluating Held-Out Prediction Data...")
pred_raw = core.load_dataset("data/prediction_data.csv")
pred_clean, _ = core.clean_dataset(pred_raw, mapping)
pred_fe = core.engineer_features(pred_clean, mapping)

# Refit model and predict
d = core.MODEL_DEFS[best_name]
pipe = core.Pipeline([
    ("prep", core.ColumnTransformer([
        ("num", core.StandardScaler() if d["scale"] else "passthrough", X.select_dtypes(include=[np.number, "bool"]).columns.tolist()),
        ("cat", core.OneHotEncoder(handle_unknown="ignore", sparse_output=False), [c for c in X.columns if c not in X.select_dtypes(include=[np.number]).columns])
    ])),
    ("clf", d["est"])
])
pipe.fit(X, y)

pred_out = core.predict_new(pipe, pred_fe[feature_cols], best_thresh, pred_raw)
print(f"Generated {len(pred_out)} predictions on held-out dataset.")

if eng_col in pred_fe.columns:
    pred_fe["high_engagement"] = (pred_fe[eng_col] > thresh).astype(int)
    oos = core.evaluate_out_of_sample(pred_fe["high_engagement"], pred_out["predicted_high_engagement"], pred_out["predicted_probability_high_engagement"])
    print("Out of Sample Evaluation:", oos)

print(f"Pipeline executed successfully in {time.time() - t0:.2f} seconds.")