"""
predict_ahfe.py

Predict absolute hydration free energies (ΔG) from precomputed descriptors
using one of three trained models: XGBoost, SVR, or Polynomial Ridge ensemble.

Models are loaded automatically from ./ahfe_models/ based on --model:
    xgb  -> ahfe_models/xgb_ahfe.json
    svr  -> ahfe_models/svr_ahfe.joblib
    poly -> ahfe_models/polynomial_ahfe.joblib

Feature attribution (available only for --model poly) is on by default and
can be disabled with --no-attribution. For performance reasons, it is advisable
to disable feature attribution when evaluating large datasets.

Usage
-----
    python predict_ahfe.py --model xgb  --input processed_data.csv
    python predict_ahfe.py --model svr  --input processed_data.csv
    python predict_ahfe.py --model poly --input processed_data.csv
    python predict_ahfe.py --model poly --input processed_data.csv --no-attribution
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -------------------------------------------------------------------------
# Model registry: filenames live in ./models/ relative to this script
# -------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent / "ahfe_models"

MODEL_FILES = {
    "xgb":  "xgb_ahfe.json",
    "svr":  "svr_ahfe.joblib",
    "poly": "polynomial_ahfe.joblib",
}

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
SELECTED_FEATURES = [
    "H_donors", "delta_eat", "gap", "alpha", "delta_mu",
    "logP", "rotatable_bonds", "delta_gap", "mol_mass", "mbd",
]

FEATURE_LABELS = {
    "H_donors":        "H donors",
    "delta_eat":       r"$\Delta E_{at}$",
    "gap":             r"$E_{gap}$",
    "alpha":           r"$\alpha$",
    "delta_mu":        r"$\Delta \mu$",
    "logP":            "log P",
    "rotatable_bonds": "rotatable bonds",
    "delta_gap":       r"$\Delta E_{GAP}$",
    "mol_mass":        "mol. mass",
    "mbd":             r"$E_{MBD}$",
}

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

AMBIGUITY_THRESHOLD = 0.9


# -------------------------------------------------------------------------
# Model loading + prediction
# -------------------------------------------------------------------------
def load_model(kind):
    fname = MODEL_FILES[kind]
    path = MODELS_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Expected '{fname}' inside {MODELS_DIR}."
        )
    if kind == "xgb":
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(str(path))
        return model
    return joblib.load(path)

def predict(kind, model, X):
    """Return (y_pred, std_pred_or_None, all_preds_or_None)."""
    if kind == "xgb":
        return model.predict(X), None, None
    if kind == "svr":
        return model.predict(X), None, None
    if kind == "poly":
        all_preds = np.array([m.predict(X) for m in model])
        return all_preds.mean(axis=0), all_preds.std(axis=0), all_preds
    raise ValueError(kind)


# -------------------------------------------------------------------------
# Feature attribution (polynomial ensemble only)
# -------------------------------------------------------------------------
def attribute_sample(x_raw, models):
    """
    Per-feature attribution for one raw sample, averaged across all ensemble
    models. Interaction terms are split proportionally to |x_scaled|.
    """
    x_raw = np.asarray(x_raw).flatten()
    n_feat = len(x_raw)
    model_attrs = []
    model_intercepts = []

    for model in models:
        poly   = model.named_steps["poly"]
        scaler = model.named_steps["scaler"]
        ridge  = model.named_steps["ridge"]

        x_poly   = poly.transform(x_raw.reshape(1, -1)).flatten()
        x_scaled = (x_poly - scaler.mean_) / scaler.scale_
        coefs    = np.atleast_1d(ridge.coef_).flatten()
        contribs = coefs * x_scaled
        powers   = poly.powers_

        feat_attr = np.zeros(n_feat)
        for contrib, power in zip(contribs, powers):
            involved = np.where(power > 0)[0]
            if len(involved) == 0:
                continue  # bias term
            if len(involved) == 1:
                feat_attr[involved[0]] += contrib
            else:
                raw_scaled = np.abs(x_scaled[involved])
                total = raw_scaled.sum()
                weights = (np.ones(len(involved)) / len(involved)
                           if total == 0 else raw_scaled / total)
                for idx, i in enumerate(involved):
                    feat_attr[i] += contrib * weights[idx]

        model_attrs.append(feat_attr)
        model_intercepts.append(float(np.atleast_1d(ridge.intercept_).mean()))

    return np.mean(model_attrs, axis=0), np.mean(model_intercepts)


def verbose_attribution(x_raw, models, feature_names, molecule_name="sample", top_n=15):
    """Print the top_n polynomial terms contributing to the prediction."""
    x_raw = np.asarray(x_raw).flatten()
    first_poly = models[0].named_steps["poly"]
    powers = first_poly.powers_
    n_terms = len(powers)

    avg_contribs = np.zeros(n_terms)
    avg_coefs    = np.zeros(n_terms)
    avg_x_scaled = np.zeros(n_terms)
    avg_x_poly   = np.zeros(n_terms)

    for model in models:
        poly   = model.named_steps["poly"]
        scaler = model.named_steps["scaler"]
        ridge  = model.named_steps["ridge"]
        x_poly   = poly.transform(x_raw.reshape(1, -1)).flatten()
        x_scaled = (x_poly - scaler.mean_) / scaler.scale_
        coefs    = np.atleast_1d(ridge.coef_).flatten()

        avg_x_poly   += x_poly   / len(models)
        avg_x_scaled += x_scaled / len(models)
        avg_coefs    += coefs    / len(models)
        avg_contribs += (coefs * x_scaled) / len(models)

    def term_name(power):
        parts = []
        for i, exp in enumerate(power):
            if exp == 0:
                continue
            elif exp == 1:
                parts.append(feature_names[i])
            else:
                parts.append(f"{feature_names[i]}^{exp}")
        return " x ".join(parts) if parts else "intercept"

    term_names = [term_name(p) for p in powers]
    order = np.argsort(np.abs(avg_contribs))[::-1]

    print(f"\n{'='*80}")
    print(f"  Term-by-term attribution — {molecule_name}")
    print(f"{'='*80}")
    print(f"  {'Term':<28} {'x_raw_val':>10} {'x_scaled':>10} {'coef':>10} {'contrib':>10}  {'credited to'}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*10}  {'-'*20}")
    for t in order[:top_n]:
        power = powers[t]
        involved = np.where(power > 0)[0]
        if len(involved) == 0:
            continue
        credited = " + ".join(feature_names[i] for i in involved)
        print(
            f"  {term_names[t]:<28} "
            f"{avg_x_poly[t]:>10.4f} "
            f"{avg_x_scaled[t]:>10.4f} "
            f"{avg_coefs[t]:>10.4f} "
            f"{avg_contribs[t]:>+10.4f}  "
            f"{credited}"
        )
    non_bias_mask = np.array([len(np.where(p > 0)[0]) > 0 for p in powers])
    print(f"\n  Total (all non-bias terms): {avg_contribs[non_bias_mask].sum():+.4f}")
    print(f"{'='*80}\n")


def check_dominance(attr, feature_names, sample_idx=None):
    abs_attr = np.abs(attr)
    rank = np.argsort(abs_attr)[::-1]
    top, runner_up = rank[0], rank[1]
    ratio = abs_attr[runner_up] / abs_attr[top] if abs_attr[top] > 0 else 0.0
    is_ambiguous = ratio >= AMBIGUITY_THRESHOLD
    if is_ambiguous:
        label = f"sample {sample_idx}" if sample_idx is not None else "sample"
        warnings.warn(
            f"[Attribution] Ambiguous dominant feature for {label}: "
            f"'{feature_names[top]}' ({abs_attr[top]:.4f}) vs "
            f"'{feature_names[runner_up]}' ({abs_attr[runner_up]:.4f}) — "
            f"ratio {ratio:.2f} ≥ {AMBIGUITY_THRESHOLD}.",
            stacklevel=2,
        )
    return is_ambiguous, ratio


# -------------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------------
def plot_mean_abs_attribution(attr_matrix, feature_names, out_path):
    """Bar chart of mean |attribution| across the dataset."""
    mean_abs = np.mean(np.abs(attr_matrix), axis=0)
    order = np.argsort(mean_abs)[::-1]
    labels = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in order]
    colors = [PALETTE[i % len(PALETTE)] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(order)), mean_abs[order], color=colors)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=12)
    ax.set_ylabel("Mean |attribution| (kcal/mol)", fontsize=13)
    ax.set_title("Global feature importance (mean absolute attribution)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_sample_attribution(attr_matrix, intercept, y_pred, feature_names,
                                names, out_dir, max_plots=25):
    """One bar chart per sample showing signed per-feature contribution."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(attr_matrix), max_plots)
    for i in range(n):
        fig, ax = plt.subplots(figsize=(8, 5))
        contribs = attr_matrix[i]
        order = np.argsort(np.abs(contribs))[::-1]
        labels = [FEATURE_LABELS.get(feature_names[j], feature_names[j]) for j in order]
        vals = contribs[order]
        colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in vals]
        ax.barh(range(len(order))[::-1], vals, color=colors)
        ax.set_yticks(range(len(order))[::-1])
        ax.set_yticklabels(labels, fontsize=11)
        ax.axvline(0, color="k", linewidth=0.8)
        title = names[i] if names is not None else f"sample_{i}"
        ax.set_title(
            f"{title}\n"
            f"intercept={intercept:.2f}   ŷ={y_pred[i]:.2f} kcal/mol",
            fontsize=12,
        )
        ax.set_xlabel("Contribution to ΔG (kcal/mol)", fontsize=12)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        safe = str(title).replace("/", "_").replace(" ", "_")
        plt.savefig(out_dir / f"{safe}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    if len(attr_matrix) > max_plots:
        print(f"[attribution] Note: only first {max_plots} per-sample plots saved "
              f"(of {len(attr_matrix)}). Use --max-per-sample-plots to change.")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Predict ΔG using a trained model.")
    p.add_argument("--model", choices=["xgb", "svr", "poly"], default="poly",
                   help="Which trained model to use (loaded from ./models/).")
    p.add_argument("--input", required=True,
                   help="CSV file containing the selected feature columns "
                        "(optionally a 'molecule' column).")
    p.add_argument("--output", default="predictions.csv",
                   help="Output CSV path (default: predictions.csv).")
    p.add_argument("--no-attribution", action="store_true",
                   help="Disable feature attribution (poly model only).")
    p.add_argument("--attribution-dir", default="attribution_results",
                   help="Directory for attribution plots + CSV.")
    p.add_argument("--verbose-molecules", default="",
                   help="Comma-separated molecule names for verbose "
                        "term-by-term breakdown.")
    p.add_argument("--max-per-sample-plots", type=int, default=25,
                   help="Max number of per-sample attribution plots to save.")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- load data ----
    df = pd.read_csv(args.input)
    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required feature columns: {missing}")

    X = df[SELECTED_FEATURES].values
    names = df["molecule"].astype(str).values if "molecule" in df.columns else None

    # ---- load model + predict ----
    print(f"[predict] Loading {args.model} model from {MODELS_DIR / MODEL_FILES[args.model]}")
    model = load_model(args.model)    
    y_pred, std_pred, _ = predict(args.model, model, X)

    out = pd.DataFrame({"y_pred": y_pred})
    if names is not None:
        out.insert(0, "molecule", names)
    if std_pred is not None:
        out["std_pred"] = std_pred

    # ---- attribution (poly only) ----
    do_attr = (args.model == "poly") and (not args.no_attribution)
    if args.no_attribution and args.model != "poly":
        pass  # nothing to do
    if args.model != "poly" and not args.no_attribution:
        print(f"[predict] Attribution not available for --model {args.model}; skipping.")

    if do_attr:
        print(f"[attribution] Running feature attribution for {len(X)} samples …")
        attr_dir = Path(args.attribution_dir)
        attr_dir.mkdir(parents=True, exist_ok=True)

        results = [attribute_sample(X[i], model) for i in range(len(X))]
        attr_matrix = np.vstack([r[0] for r in results])
        intercepts = np.array([r[1] for r in results])
        dominant_idx = np.argmax(np.abs(attr_matrix), axis=1)

        ambiguity_flags = []
        dominance_ratios = []
        for i, attr in enumerate(attr_matrix):
            flag, ratio = check_dominance(
                attr, SELECTED_FEATURES,
                sample_idx=(names[i] if names is not None else i),
            )
            ambiguity_flags.append(flag)
            dominance_ratios.append(ratio)

        # verbose per-molecule breakdown
        verbose_list = [s.strip() for s in args.verbose_molecules.split(",") if s.strip()]
        if verbose_list:
            if names is None:
                print("[attribution] --verbose-molecules ignored (no 'molecule' column).")
            else:
                lower_names = np.array([n.lower() for n in names])
                for mol_name in verbose_list:
                    idxs = np.where(lower_names == mol_name.lower())[0]
                    if len(idxs) == 0:
                        print(f"[attribution] '{mol_name}' not found in input.")
                        continue
                    verbose_attribution(
                        X[idxs[0]], model,
                        feature_names=SELECTED_FEATURES,
                        molecule_name=mol_name, top_n=15,
                    )

        # global importance plot
        plot_mean_abs_attribution(
            attr_matrix, SELECTED_FEATURES,
            out_path=attr_dir / "global_mean_abs_attribution.png",
        )
        # per-sample plots
        plot_per_sample_attribution(
            attr_matrix, float(np.mean(intercepts)), y_pred,
            SELECTED_FEATURES, names, attr_dir / "per_sample",
            max_plots=args.max_per_sample_plots,
        )

        # attribution CSV
        attr_df = df[SELECTED_FEATURES].copy()
        if names is not None:
            attr_df.insert(0, "molecule", names)
        attr_df["y_pred"] = y_pred
        attr_df["std_pred"] = std_pred
        attr_df["intercept"] = intercepts
        attr_df["dom_feature"] = [SELECTED_FEATURES[i] for i in dominant_idx]
        attr_df["dominance_ratio"] = dominance_ratios
        for i, feat in enumerate(SELECTED_FEATURES):
            attr_df[f"attr_{feat}"] = attr_matrix[:, i]
        attr_df["attr_sum"] = attr_matrix.sum(axis=1)
        attr_df["budget_check"] = np.isclose(
            attr_df["attr_sum"] + attr_df["intercept"], attr_df["y_pred"]
        )
        attr_csv = attr_dir / "attribution.csv"
        attr_df.to_csv(attr_csv, index=False)
        print(f"[attribution] Saved → {attr_csv}")
        print(f"[attribution] Plots → {attr_dir}")

        n_amb = sum(ambiguity_flags)
        if n_amb:
            print(f"\n⚠  {n_amb}/{len(X)} samples have an ambiguous dominant feature "
                  f"(ratio ≥ {AMBIGUITY_THRESHOLD}).")
        else:
            print(f"\n✓  All samples have a clearly dominant feature "
                  f"(ratio < {AMBIGUITY_THRESHOLD}).")

        print("\nDominant feature distribution:")
        print(attr_df["dom_feature"].value_counts().to_string())

    # ---- save main predictions ----
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\n[predict] Saved predictions → {args.output}")


if __name__ == "__main__":
    main()
