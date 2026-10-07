#!/usr/bin/env python
"""
feature_importance.py

Merged feature-importance analysis:
  Part 1 — Pearson correlations, LASSO coefficients, Random Forest
           permutation importance.
  Part 2 — Cross-validated XGBoost + SHAP cumulative-importance elbow plot.

All features in the input CSV are used except the target column
(and 'molecule' / 'name' / 'id' if present).

Usage:
    python feature_importance.py --input train.csv --target dG
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor
import shap

warnings.filterwarnings("ignore")

# Publication-quality font embedding for PDF
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42


# =============================================================================
# CONFIGURATION
# =============================================================================
N_FOLDS      = 5
RANDOM_STATE = 42

LABEL_MAP = {'rotatable_bonds': 'rot_bonds'}

TICK_SIZE  = 20
LABEL_SIZE = 22
TITLE_SIZE = 22

XGB_PARAMS = dict(
    n_estimators          = 600,
    learning_rate         = 0.03,
    max_depth             = 5,
    subsample             = 0.8,
    colsample_bytree      = 0.9,
    min_child_weight      = 3,
    random_state          = RANDOM_STATE,
    n_jobs                = -1,
    early_stopping_rounds = 30,
    eval_metric           = 'mae',
)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(path: str, target: str, exclude=None):
    exclude = set(exclude or [])
    df = pd.read_csv(path) if str(path).endswith('.csv') else pd.read_excel(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. "
                         f"Available: {list(df.columns)}")

    drop_cols = {target, "molecule", "name", "id"} | exclude
    feature_cols = [c for c in df.columns if c not in drop_cols]

    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    dropped = set(feature_cols) - set(numeric_features)
    if dropped:
        print(f"  [info] Ignoring non-numeric columns: {sorted(dropped)}")

    # Warn about excluded names that weren't in the CSV
    unknown = exclude - set(df.columns)
    if unknown:
        print(f"  [warn] --exclude names not found in CSV (ignored): {sorted(unknown)}")

    if exclude:
        actually_excluded = exclude & set(df.columns) - {target}
        if actually_excluded:
            print(f"  [info] Excluded features: {sorted(actually_excluded)}")

    df = df[numeric_features + [target]].dropna()
    print(f"  Loaded {len(df)} samples, {len(numeric_features)} features")
    return df, numeric_features


# =============================================================================
# PART 1 — Correlations + LASSO + RF permutation importance
# =============================================================================
def run_corr_lasso_rf(df, features, target, out_dir):
    X_train = df[features]
    y_train = df[target]

    # ---------- 1. Linear correlations ----------
    correlations = df[features].corrwith(df[target])

    # ---------- 2. LASSO coefficients ----------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lasso_cv = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=10000, n_alphas=100)
    lasso_cv.fit(X_train_scaled, y_train)
    lasso_coefficients = pd.Series(lasso_cv.coef_, index=features)

    # ---------- 3. Permutation importance (Random Forest) ----------
    rf = RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, max_depth=10
    )
    rf.fit(X_train, y_train)

    perm_importance = permutation_importance(
        rf, X_train, y_train, n_repeats=10,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    perm_importances = pd.Series(perm_importance.importances_mean, index=features)
    perm_std         = pd.Series(perm_importance.importances_std,  index=features)

    # ---------- Shared ordering (by perm importance, ascending) ----------
    feature_order = perm_importances.sort_values(ascending=True).index.tolist()

    correlations_ord = correlations.loc[feature_order]
    lasso_ord        = lasso_coefficients.loc[feature_order]
    perm_ord         = perm_importances.loc[feature_order]
    perm_std_ord     = perm_std.loc[feature_order]

    display_labels = [LABEL_MAP.get(f, f) for f in feature_order]

    # ---------- Combined figure ----------
    fig, axes = plt.subplots(
        1, 3, figsize=(22, 10), sharey=True,
        gridspec_kw={'wspace': 0.05}
    )

    # Subplot 1: Correlations
    axes[0].barh(
        feature_order, correlations_ord.values,
        color=['#c0392b' if x < 0 else '#27ae60' for x in correlations_ord.values]
    )
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    axes[0].set_ylabel('Features', fontsize=LABEL_SIZE)
    axes[0].set_title('Correlation coefficient', fontsize=TITLE_SIZE, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].tick_params(axis='both', labelsize=TICK_SIZE)
    axes[0].set_yticks(range(len(feature_order)))
    axes[0].set_yticklabels(display_labels)

    # Subplot 2: LASSO coefficients
    axes[1].barh(
        feature_order, lasso_ord.values,
        color=['#c0392b' if x < 0 else '#27ae60' if x > 0 else 'gray'
               for x in lasso_ord.values]
    )
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_title('LASSO coefficients', fontsize=TITLE_SIZE, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].tick_params(axis='x', labelsize=TICK_SIZE)

    # Subplot 3: Permutation importance
    axes[2].barh(
        feature_order, perm_ord.values,
        xerr=perm_std_ord.values, color='steelblue',
        error_kw=dict(ecolor='black', lw=0.8, capsize=3)
    )
    axes[2].set_title('Permutation importance', fontsize=TITLE_SIZE, fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3)
    axes[2].tick_params(axis='x', labelsize=TICK_SIZE)

    plt.tight_layout()
    png_path = out_dir / "feature_importance_combined.png"
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {png_path}")

# =============================================================================
# PART 2 — CV XGBoost + SHAP elbow
# =============================================================================
def run_cv_shap(df, features, target):
    X = df[features].values.astype(float)
    y = df[target].values.astype(float)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    shap_rows       = np.zeros((len(X), len(features)))
    fold_shap_means = []

    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATED XGBOOST  ({N_FOLDS} folds)")
    print(f"{'='*60}")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        pred_val = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred_val)
        r2  = r2_score(y_val, pred_val)
        print(f"  Fold {fold+1}/{N_FOLDS}  MAE={mae:.4f}  R²={r2:.4f}")

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_val)
        shap_rows[val_idx] = shap_vals
        fold_shap_means.append(np.abs(shap_vals).mean(axis=0))

    return shap_rows, np.array(fold_shap_means)


def compute_importance(shap_matrix, fold_shap_means, features):
    mean_abs  = np.abs(shap_matrix).mean(axis=0)
    std_folds = fold_shap_means.std(axis=0)
    imp = pd.DataFrame({
        'feature':   features,
        'shap_mean': mean_abs,
        'shap_std':  std_folds,
    }).sort_values('shap_mean', ascending=False).reset_index(drop=True)
    imp['rank'] = imp.index + 1
    return imp


def plot_elbow(imp, out_dir):
    cumsum = imp['shap_mean'].cumsum() / imp['shap_mean'].sum() * 100

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(range(1, len(imp)+1), cumsum, 'o-',
            color='steelblue', ms=8, lw=2.2)

    for threshold, color, label in [(80, 'orange', '80%'),
                                    (90, 'crimson', '90%')]:
        idx = int(np.argmax(cumsum >= threshold)) + 1
        ax.axvline(x=idx, color=color, linestyle='--', lw=1.8,
                   label=f'{label} at {idx} features')
        ax.axhline(y=threshold, color=color, linestyle=':', lw=1.2, alpha=0.6)

    ax.set_xlabel("Features", fontsize=LABEL_SIZE)
    ax.set_ylabel("Cumulative importance (%)", fontsize=LABEL_SIZE)

    display_labels = [LABEL_MAP.get(f, f) for f in imp['feature'].tolist()]
    ax.set_xticks(range(1, len(imp)+1))
    ax.set_xticklabels(display_labels, rotation=45, ha='right')

    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    png_path = out_dir / "shap_elbow.png"
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {png_path}")


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Merged feature-importance analysis.")
    p.add_argument("--input",  required=True, help="Input CSV (or Excel) file.")
    p.add_argument("--target", required=True, help="Target column name.")
    p.add_argument("--exclude", default="",
                   help="Comma-separated list of feature columns to exclude "
                        "(e.g. --exclude 'n_conf,sigma_sasa').")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path("feature_importance")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading: {args.input}")
    excluded = [s.strip() for s in args.exclude.split(",") if s.strip()]
    df, features = load_data(args.input, args.target, exclude=excluded)

    print(f"\n{'='*60}")
    print("PART 1 — Correlations + LASSO + RF permutation importance")
    print(f"{'='*60}")
    run_corr_lasso_rf(df, features, args.target, out_dir)

    shap_matrix, fold_shap_means = run_cv_shap(df, features, args.target)
    imp = compute_importance(shap_matrix, fold_shap_means, features)

    print(f"\n{'='*60}")
    print("GENERATING ELBOW PLOT")
    print(f"{'='*60}")
    plot_elbow(imp, out_dir)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()