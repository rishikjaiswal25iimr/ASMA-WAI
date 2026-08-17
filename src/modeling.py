"""
modeling.py
===========
Layer 5 — Machine Learning: builds a shared preprocessing pipeline and trains
four independent classifiers (Logistic Regression, Random Forest, XGBoost,
SVM) for the High Engagement target, using ONLY the 15,000-record training
dataset (internal train/validation split with stratification).
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

RANDOM_SEED = 42


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipeline = "passthrough"
    if scale_numeric:
        num_pipeline = StandardScaler()

    cat_pipeline = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipeline, categorical_cols))

    return ColumnTransformer(transformers, remainder="drop")


def split_train_validation(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y)


MODEL_DEFINITIONS = {
    "Logistic Regression": {
        "scale_numeric": True,
        "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "param_grid": {"clf__C": [0.1, 1.0, 10.0]},
    },
    "Random Forest": {
        "scale_numeric": False,
        # n_jobs=1 here deliberately: GridSearchCV below already parallelizes across
        # hyperparameter/CV folds with n_jobs=-1. Parallelizing at BOTH levels causes
        # CPU/memory oversubscription (each GridSearchCV worker spawning its own pool
        # of RF worker processes), which can make training extremely slow or exhaust
        # memory on constrained hosts such as Streamlit Community Cloud.
        "estimator": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=1),
        "param_grid": {"clf__n_estimators": [200, 400], "clf__max_depth": [8, 16, None]},
    },
    "XGBoost": {
        "scale_numeric": False,
        # Same reasoning as Random Forest above — avoid nested parallelism.
        "estimator": XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss", n_jobs=1),
        "param_grid": {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5], "clf__learning_rate": [0.05, 0.1]},
    },
    "SVM": {
        # SVM is scaled + trained on a stratified subsample when the training
        # set is large, since SVC training cost grows quickly with n — this
        # keeps the project reproducible without excessive runtime, and is
        # documented in the app/README as an explicit, disclosed adjustment.
        "scale_numeric": True,
        "estimator": SVC(probability=True, random_state=RANDOM_SEED),
        "param_grid": {"clf__C": [1.0, 10.0], "clf__kernel": ["rbf"]},
    },
}


def train_one_model(name: str, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 3,
                     svm_subsample: int = 6000):
    """Train + light hyperparameter-tune a single model; return (fitted_pipeline, train_seconds)."""
    definition = MODEL_DEFINITIONS[name]

    X_fit, y_fit = X_train, y_train
    if name == "SVM" and len(X_train) > svm_subsample:
        X_fit, _, y_fit, _ = train_test_split(
            X_train, y_train, train_size=svm_subsample, stratify=y_train, random_state=RANDOM_SEED
        )

    preprocessor = build_preprocessor(X_fit, scale_numeric=definition["scale_numeric"])
    pipe = Pipeline([("prep", preprocessor), ("clf", definition["estimator"])])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(pipe, definition["param_grid"], cv=cv, scoring="f1", n_jobs=-1)

    start = time.time()
    search.fit(X_fit, y_fit)
    elapsed = time.time() - start

    return search.best_estimator_, elapsed, search.best_params_


def train_all_models(X_train, y_train, model_names=None):
    """Train all requested models; returns dict[name] = {'model', 'train_seconds', 'best_params'}."""
    if model_names is None:
        model_names = list(MODEL_DEFINITIONS.keys())
    results = {}
    for name in model_names:
        model, secs, params = train_one_model(name, X_train, y_train)
        results[name] = {"model": model, "train_seconds": secs, "best_params": params}
    return results
