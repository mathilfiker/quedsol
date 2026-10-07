#!/usr/bin/env python
"""
train.py

Unified training script for ΔG (or any target) prediction models.

Supported models:
    poly : Quadratic Ridge regression, 10-fold ensemble (list of Pipelines).
    svr  : SVR (RBF), GridSearchCV over C / epsilon / gamma.
    xgb  : XGBoost with Optuna hyperparameter optimization.

Usage:
    python train.py --model xgb  --train train.csv --test test.csv --target dG
    python train.py --model svr  --train train.csv --target dG
    python train.py --model poly --train train.csv --target dG --exclude "n_conf,sigma_sasa"
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error

from xgboost import XGBRegressor
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
MODELS_DIR   = Path("trained_models")

# Poly settings
POLY_DEGREE = 2
POLY_ALPHA  = 20.0
POLY_FOLDS  = 10

# SVR settings
SVR_PARAM_GRID = {
    "svr__C":       [0.1, 1, 10, 100, 1000, 1500],
    "svr__epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
    "svr__gamma":   ["scale", 0.001, 0.01, 0.1, 1],
}
SVR_CV_FOLDS = 10

# XGB / Optuna settings
XGB_N_FOLDS  = 5
XGB_N_TRIALS = 1000


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(train_path, test_path, target, exclude):
    def _read(p):
        p = str(p)
        return pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)

    df_tr = _read(train_path)
    if target not in df_tr.columns:
        raise ValueError(f"Target '{target}' not found in training CSV. "
                         f"Available: {list(df_tr.columns)}")

    exclude = set(exclude or [])
    drop_cols = {target, "molecule", "name", "id"} | exclude
    feature_cols = [c for c in df_tr.columns if c not in drop_cols]
    numeric_features = df_tr[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    dropped_nn = set(feature_cols) - set(numeric_features)
    if dropped_nn:
        print(f"  [info] Ignoring non-numeric columns: {sorted(dropped_nn)}")
    unknown = exclude - set(df_tr.columns)
    if unknown:
        print(f"  [warn] --exclude names not found in training CSV: {sorted(unknown)}")
    if exclude:
        actually_excluded = exclude & set(df_tr.columns) - {target}
        if actually_excluded:
            print(f"  [info] Excluded features: {sorted(actually_excluded)}")

    df_tr = df_tr[numeric_features + [target]].dropna()
    X_tr = df_tr[numeric_features].values.astype(float)
    y_tr = df_tr[target].values.astype(float)
    print(f"  Train: {len(df_tr)} samples × {len(numeric_features)} features")

    X_te = y_te = None
    if test_path is not None:
        df_te = _read(test_path)
        missing = [c for c in numeric_features + [target] if c not in df_te.columns]
        if missing:
            raise ValueError(f"Test CSV is missing columns: {missing}")
        df_te = df_te[numeric_features + [target]].dropna()
        X_te = df_te[numeric_features].values.astype(float)
        y_te = df_te[target].values.astype(float)
        print(f"  Test : {len(df_te)} samples")

    return numeric_features, X_tr, y_tr, X_te, y_te


# =============================================================================
# METRICS
# =============================================================================
def print_metrics(label, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"  {label:<8} MAE = {mae:.4f}   RMSE = {rmse:.4f}   R² = {r2:.4f}")


# =============================================================================
# POLYNOMIAL
# =============================================================================
def train_poly(X_tr, y_tr, X_te, y_te):
    """10-fold ensemble of quadratic Ridge pipelines."""
    def make_model():
        return Pipeline([
            ("poly",   PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=POLY_ALPHA)),
        ])

    kf = KFold(n_splits=POLY_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = []
    fold_val_maes = []

    for fold, (tr_idx, held_out_idx) in enumerate(kf.split(X_tr)):
        model = make_model()
        model.fit(X_tr[tr_idx], y_tr[tr_idx])
        val_mae = mean_absolute_error(y_tr[held_out_idx],
                                      model.predict(X_tr[held_out_idx]))
        fold_val_maes.append(val_mae)
        models.append(model)
        print(f"  [poly] Fold {fold+1:02d}/{POLY_FOLDS} "
              f"trained on {len(tr_idx)} molecules "
              f"(held out {len(held_out_idx)})  val MAE={val_mae:.4f}")

    out_path = MODELS_DIR / "polynomial.joblib"
    joblib.dump(models, out_path)
    print(f"\n  Saved → {out_path}")

    def ensemble_predict(X):
        return np.mean([m.predict(X) for m in models], axis=0)

    print(f"\n{'='*60}\nPERFORMANCE\n{'='*60}")
    print(f"  CV MAE (mean of held-out folds) : "
          f"{np.mean(fold_val_maes):.4f}  ±  {np.std(fold_val_maes):.4f}")
    print_metrics("Train", y_tr, ensemble_predict(X_tr))
    if X_te is not None:
        print_metrics("Test", y_te, ensemble_predict(X_te))


# =============================================================================
# SVR
# =============================================================================
def train_svr(X_tr, y_tr, X_te, y_te):
    """SVR (RBF) with GridSearchCV over C/epsilon/gamma."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf")),
    ])
    grid = GridSearchCV(
        pipe,
        param_grid=SVR_PARAM_GRID,
        cv=SVR_CV_FOLDS,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid.fit(X_tr, y_tr)
    best_model = grid.best_estimator_

    print(f"\n  Best params : {grid.best_params_}")
    print(f"  Best score  (neg_MSE) : {grid.best_score_:.4f}")

    out_path = MODELS_DIR / "svr.joblib"
    joblib.dump(best_model, out_path)
    print(f"  Saved → {out_path}")

    print(f"\n{'='*60}\nPERFORMANCE\n{'='*60}")
    print(f"  CV MSE (grid-search best) : {-grid.best_score_:.4f}")
    print_metrics("Train", y_tr, best_model.predict(X_tr))
    if X_te is not None:
        print_metrics("Test", y_te, best_model.predict(X_te))


# =============================================================================
# XGBOOST + OPTUNA
# =============================================================================
def train_xgb(X_tr, y_tr, X_te, y_te):
    kf = KFold(n_splits=XGB_N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 1000),
            learning_rate     = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            max_depth         = trial.suggest_int("max_depth", 2, 8),
            min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            gamma             = trial.suggest_float("gamma", 0.0, 5.0),
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
            eval_metric       = "rmse",
            early_stopping_rounds = 30,
        )

        fold_mses = []
        for tr_idx, val_idx in kf.split(X_tr):
            X_a, X_v = X_tr[tr_idx], X_tr[val_idx]
            y_a, y_v = y_tr[tr_idx], y_tr[val_idx]
            model = XGBRegressor(**params)
            model.fit(X_a, y_a, eval_set=[(X_v, y_v)], verbose=False)
            fold_mses.append(mean_squared_error(y_v, model.predict(X_v)))
        return float(np.mean(fold_mses))

    print(f"  Running Optuna ({XGB_N_TRIALS} trials, {XGB_N_FOLDS}-fold CV)...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
    )
    study.optimize(objective, n_trials=XGB_N_TRIALS, show_progress_bar=True)

    print(f"\n  Best CV MSE : {study.best_value:.4f}")
    print(f"  Best params :")
    for k, v in study.best_params.items():
        print(f"    {k:<20} : {v}")

    # Refit on full training set (no early stopping)
    best_params = study.best_params.copy()
    best_params.update(dict(
        random_state=RANDOM_STATE, n_jobs=-1, eval_metric="rmse",
    ))
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_tr, y_tr, verbose=False)

    out_path = MODELS_DIR / "xgb_optuna.json"
    best_model.save_model(str(out_path))
    print(f"\n  Saved → {out_path}")

    print(f"\n{'='*60}\nPERFORMANCE\n{'='*60}")
    print(f"  CV MSE (Optuna best) : {study.best_value:.4f}")
    print_metrics("Train", y_tr, best_model.predict(X_tr))
    if X_te is not None:
        print_metrics("Test", y_te, best_model.predict(X_te))


# =============================================================================
# CLI
# =============================================================================
TRAINERS = {"poly": train_poly, "svr": train_svr, "xgb": train_xgb}

def parse_args():
    p = argparse.ArgumentParser(description="Train a ΔG prediction model.")
    p.add_argument("--model", choices=list(TRAINERS.keys()), required=True,
                   help="Which model to train.")
    p.add_argument("--train", required=True, help="Training CSV.")
    p.add_argument("--test",  default=None, help="Optional test CSV.")
    p.add_argument("--target", required=True, help="Target column name.")
    p.add_argument("--exclude", default="",
                   help="Comma-separated feature columns to exclude.")
    return p.parse_args()


def main():
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    excluded = [s.strip() for s in args.exclude.split(",") if s.strip()]

    print(f"\nLoading data (target = '{args.target}')...")
    features, X_tr, y_tr, X_te, y_te = load_data(
        args.train, args.test, args.target, exclude=excluded
    )
    print(f"  Using {len(features)} features: {features}")

    print(f"\n{'='*60}\nTRAINING {args.model.upper()}\n{'='*60}")
    TRAINERS[args.model](X_tr, y_tr, X_te, y_te)

    print(f"\n{'='*60}\nDONE\n{'='*60}")


if __name__ == "__main__":
    main()
