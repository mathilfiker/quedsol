import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr, spearmanr, kendalltau

# ── Config ───────────────────────────────────────────────────────────────────
selected_features = [
    'H_donors', 'delta_eat', 'gap', 'alpha', 'delta_mu',
    'logP', 'rotatable_bonds', 'delta_gap', 'mol_mass', 'mbd'
]
target = "dG"

# ── Load data & models ────────────────────────────────────────────────────────
df_test = pd.read_csv('freesolv_test.csv')
x_test  = df_test[selected_features].values
y_test  = df_test[target].values

pipe = joblib.load("../../scripts/ahfe_models/polynomial_ahfe.joblib")


# ── Ensemble predict ──────────────────────────────────────────────────────────
def ensemble_predict(models, X_new):
    all_preds = np.array([m.predict(X_new) for m in models])
    return all_preds.mean(axis=0), all_preds.std(axis=0), all_preds

y_pred, std_pred, all_preds = ensemble_predict(pipe, x_test)


# ── Feature attribution (in scaled space) ────────────────────────────────────
#
# Pipeline: x_raw → poly → x_poly → scaler → x_scaled → ridge → ŷ
#
# Ridge computes:  ŷ = coef · x_scaled + intercept
#
# Attribution of poly-term j:  contrib_j = coef_j * x_scaled_j
#
# This is the quantity that actually drives the prediction.
# We then map each poly-term's contribution back to the raw feature(s)
# it was derived from, splitting interaction terms equally.

def attribute_sample(x_raw, models):
    """
    Per-feature attribution for one raw sample, averaged across all ensemble
    models. Interaction terms are split proportionally to |x_scaled|.

    Returns
    -------
    feat_attr : np.ndarray, shape [n_raw_features]  — kcal/mol per feature
    intercept : float  — averaged intercept across models
    """
    x_raw  = np.asarray(x_raw).flatten()
    n_feat = len(x_raw)
    model_attrs  = []
    model_intercepts = []

    for model in models:
        poly   = model.named_steps["poly"]
        scaler = model.named_steps["scaler"]
        ridge  = model.named_steps["ridge"]

        x_poly   = poly.transform(x_raw.reshape(1, -1)).flatten()   # [n_poly_terms]
        x_scaled = (x_poly - scaler.mean_) / scaler.scale_          # [n_poly_terms]
        coefs    = np.atleast_1d(ridge.coef_).flatten()              # [n_poly_terms]
        contribs = coefs * x_scaled                                  # [n_poly_terms]
        powers   = poly.powers_                                      # [n_poly_terms, n_raw_features]

        feat_attr = np.zeros(n_feat)

        for t, (contrib, power) in enumerate(zip(contribs, powers)):
            involved = np.where(power > 0)[0]
            if len(involved) == 0:
                continue  # bias term → goes to intercept

            if len(involved) == 1:
                # Linear or pure quadratic: full credit to the one feature
                feat_attr[involved[0]] += contrib
            else:
                # Interaction term: weight by |x_scaled| of each raw feature.
                raw_scaled = np.abs(x_scaled[involved])   # |x_scaled| for each involved feature's linear term
                total = raw_scaled.sum()
                if total == 0:
                    weights = np.ones(len(involved)) / len(involved)   # fallback: equal split
                else:
                    weights = raw_scaled / total
                for idx, i in enumerate(involved):
                    feat_attr[i] += contrib * weights[idx]

        model_attrs.append(feat_attr)
        model_intercepts.append(float(np.atleast_1d(ridge.intercept_).mean()))

    return np.mean(model_attrs, axis=0), np.mean(model_intercepts)



# ── Verbose term-by-term breakdown for a single molecule ─────────────────────
def verbose_attribution(x_raw, models, feature_names, molecule_name="sample", top_n=15):
    """
    Print the top_n polynomial terms contributing to the attribution of a
    given feature, averaged across all ensemble models.

    For each term shows:
      - the term name  (e.g. "gap", "gap^2", "gap x logP")
      - the raw poly value
      - x_scaled
      - the averaged coefficient
      - the final contribution  coef * x_scaled
      - which raw feature(s) it is credited to
    """
    x_raw = np.asarray(x_raw).flatten()

    # Accumulate per-term contributions averaged over models
    first_poly  = models[0].named_steps["poly"]
    powers      = first_poly.powers_                      # [n_terms, n_raw_feats]
    n_terms     = len(powers)

    avg_contribs  = np.zeros(n_terms)
    avg_coefs     = np.zeros(n_terms)
    avg_x_scaled  = np.zeros(n_terms)
    avg_x_poly    = np.zeros(n_terms)

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

    # Build human-readable term names
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

    # Sort all terms by |contribution|
    order = np.argsort(np.abs(avg_contribs))[::-1]

    print(f"\n{'='*80}")
    print(f"  Term-by-term attribution — {molecule_name}")
    print(f"{'='*80}")
    print(f"  {'Term':<28} {'x_raw_val':>10} {'x_scaled':>10} {'coef':>10} {'contrib':>10}  {'credited to'}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*10}  {'-'*20}")

    for t in order[:top_n]:
        power    = powers[t]
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

    print(f"\n  Total (all terms): {avg_contribs[np.array([len(np.where(p>0)[0])>0 for p in powers])].sum():+.4f}")
    print(f"{'='*80}\n")


# ── Dominance check ───────────────────────────────────────────────────────────
# AMBIGUITY_THRESHOLD: if |attr_2nd| / |attr_1st| exceeds this, the dominant
# feature is not clearly separated from the runner-up. 0.9 means the 2nd
# feature carries at least 90 % of the top feature's absolute attribution.
AMBIGUITY_THRESHOLD = 0.9

def check_dominance(attr, sample_idx=None):
    """
    Warn when the runner-up attribution is within AMBIGUITY_THRESHOLD of the
    top attribution (in absolute terms).

    Parameters
    ----------
    attr       : 1-D array of per-feature attributions for one sample.
    sample_idx : optional sample index, included in the warning message.

    Returns
    -------
    is_ambiguous : bool
    dominance_ratio : float  — |attr_2nd| / |attr_1st|  (0 = clear winner, 1 = tied)
    """
    abs_attr = np.abs(attr)
    rank     = np.argsort(abs_attr)[::-1]          # indices sorted by |attribution|
    top, runner_up = rank[0], rank[1]

    dominance_ratio = abs_attr[runner_up] / abs_attr[top] if abs_attr[top] > 0 else 0.0
    is_ambiguous    = dominance_ratio >= AMBIGUITY_THRESHOLD

    if is_ambiguous:
        label = f"sample {sample_idx}" if sample_idx is not None else "sample"
        warnings.warn(
            f"[Attribution] Ambiguous dominant feature for {label}: "
            f"'{selected_features[top]}' ({abs_attr[top]:.4f}) vs "
            f"'{selected_features[runner_up]}' ({abs_attr[runner_up]:.4f}) — "
            f"ratio {dominance_ratio:.2f} ≥ threshold {AMBIGUITY_THRESHOLD}.",
            stacklevel=2,
        )

    return is_ambiguous, dominance_ratio


# ── Sanity-check on first sample ─────────────────────────────────────────────
def _check_first_sample():
    model  = pipe[0]
    poly   = model.named_steps["poly"]
    scaler = model.named_steps["scaler"]
    ridge  = model.named_steps["ridge"]
    x0     = x_test[0]

    x_scaled  = scaler.transform(poly.transform(x0.reshape(1, -1)))
    y_direct  = ridge.predict(x_scaled)[0]

    attr, intercept = attribute_sample(x0, [model])
    y_reconstr      = attr.sum() + intercept

    print("Sanity check (sample 0, model 0):")
    print(f"  pipeline.predict     : {pipe[0].predict(x0.reshape(1,-1))[0]:.6f}")
    print(f"  attr.sum + intercept : {y_reconstr:.6f}")
    print(f"  match                : {np.isclose(y_direct, y_reconstr)}\n")

_check_first_sample()


# ── Run attribution for all test samples ──────────────────────────────────────
print(f"Running feature attribution for {len(x_test)} samples …")
results      = [attribute_sample(x_test[i], pipe) for i in range(len(x_test))]
attr_matrix  = np.vstack([r[0] for r in results])
intercepts   = np.array([r[1] for r in results])   # same for all samples, but explicit per row
dominant_idx = np.argmax(np.abs(attr_matrix), axis=1)

# Check dominance for every sample and collect summary
ambiguity_flags  = []
dominance_ratios = []
for i, attr in enumerate(attr_matrix):
    flag, ratio = check_dominance(attr, sample_idx=i)
    ambiguity_flags.append(flag)
    dominance_ratios.append(ratio)

# ── Verbose breakdown for molecules of interest ───────────────────────────────
# Add any molecule name here to get the full term-by-term printout.
VERBOSE_MOLECULES = []

if VERBOSE_MOLECULES:
    mol_col = df_test["molecule"] if "molecule" in df_test.columns else None
    for mol_name in VERBOSE_MOLECULES:
        if mol_col is None:
            print(f"Warning: no 'molecule' column in test CSV, skipping verbose for {mol_name}")
            break
        idx = df_test.index[mol_col.str.lower() == mol_name.lower()]
        if len(idx) == 0:
            print(f"Warning: '{mol_name}' not found in test set.")
            continue
        verbose_attribution(
            x_test[idx[0]], pipe,
            feature_names=selected_features,
            molecule_name=mol_name,
            top_n=15,
        )

n_ambiguous = sum(ambiguity_flags)
if n_ambiguous:
    print(f"\n⚠  {n_ambiguous}/{len(x_test)} samples have an ambiguous dominant feature "
          f"(ratio ≥ {AMBIGUITY_THRESHOLD}). See warnings above for details.")
else:
    print(f"\n✓  All samples have a clearly dominant feature (ratio < {AMBIGUITY_THRESHOLD}).")

# Store ratio in output for inspection
dominance_ratios = np.array(dominance_ratios)


# ── Parity plot ───────────────────────────────────────────────────────────────
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

fig, ax = plt.subplots(figsize=(16, 16))

# Axis limits
y_min = min(y_test.min(), y_pred.min())
y_max = max(y_test.max(), y_pred.max())

# Diagonal and shaded bands
x_range = np.linspace(y_min - 1, y_max + 1, 400)
ax.plot(x_range, x_range, 'k-', linewidth=1.5, zorder=1)
ax.fill_between(x_range, x_range - 2, x_range + 2, color="gray", alpha=0.3, zorder=0)
ax.fill_between(x_range, x_range - 1, x_range + 1, color="gray", alpha=0.5, zorder=0)

# Get experimental errors if available
mol_col = df_test["molecule"] if "molecule" in df_test.columns else None
exp_err = 0.6

# Scatter by dominant feature, with error bars
for feat_i, feat_name in enumerate(selected_features):
    mask = dominant_idx == feat_i
    if mask.sum() == 0:
        continue
    ax.errorbar(
        y_test[mask], y_pred[mask],
        xerr=exp_err, yerr=std_pred[mask],
        fmt='o', color=PALETTE[feat_i],
        alpha=0.8, markersize=25,
        label=feat_name, zorder=2,
        elinewidth=1.5, capsize=4,
    )

ax.set_xlim(y_min - 1, y_max + 1)
ax.set_ylim(y_min - 1, y_max + 1)
ax.set_xlabel("Experimental $\\Delta G$ (kcal/mol)", fontsize=45)
ax.set_ylabel("Predicted $\\Delta G$ (kcal/mol)", fontsize=45)
ax.tick_params(axis='x', labelsize=40)
ax.tick_params(axis='y', labelsize=40)

# ── Stats legend  ──────────────────────────────────────────────────
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - y_test.mean()) ** 2)
r2      = 1 - ss_res / ss_tot
pearson_r, _   = pearsonr(y_test, y_pred)
spearman_r, _  = spearmanr(y_test, y_pred)
kendall_tau, _ = kendalltau(y_test, y_pred)
mae_val  = np.mean(np.abs(y_test - y_pred))
rmse_val = np.sqrt(np.mean((y_test - y_pred) ** 2))

stats_text = (
    f"MAE = {mae_val:.2f} kcal/mol\n"
    f"RMSE = {rmse_val:.2f} kcal/mol\n"
    f"τ = {kendall_tau:.2f}\n"
    f"r = {pearson_r:.2f}\n"
    f"R² = {r2:.2f}\n"
    r"$\rho$" + f"  = {spearman_r:.2f}"
)
ax.legend(title=stats_text, loc="upper left",
          title_fontsize=45, fontsize=22, handles=[])

# ── Feature colour legend ──────────────────────────────────
present_features = [
    (i, selected_features[i])
    for i in range(len(selected_features))
    if (dominant_idx == i).sum() > 0
]
FEATURE_LABELS = {
    'H_donors':        'H donors',
    'delta_eat':       r'$\Delta E_{at}$',
    'gap':             r'$E_{gap}$',
    'alpha':           r'$\alpha$',
    'delta_mu':        r'$\Delta \mu$',
    'logP':            'log P',
    'rotatable_bonds': 'rotatable bonds',
    'delta_gap':       r'$\Delta E_{GAP}$',
    'mol_mass':        'mol. mass',
    'mbd':             r'$E_{MBD}$',
}

# Build proxy handles and add as a second legend on the main axis
handles = [
    mpatches.Patch(
        facecolor=PALETTE[feat_i],
        edgecolor="none",
        label=FEATURE_LABELS.get(feat_name, feat_name),
    )
    for feat_i, feat_name in present_features
]

feature_legend = ax.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.12),   # place just below the plot area
    ncol=len(handles),             # horizontal single-row layout
    frameon=False,
    fontsize=16,
    handlelength=1.2,
    handleheight=1.2,
    columnspacing=1.5,
)

# Re-add the stats legend (adding a second legend removes the first)
ax.add_artist(
    ax.legend(title=stats_text, loc="upper left",
              title_fontsize=45, fontsize=22, handles=[])
)
ax.add_artist(feature_legend)

plt.tight_layout()
plt.savefig("results/freesolv_attribution.png", dpi=300,
            bbox_inches="tight", pad_inches=0.1)
plt.show()

# ── Save results ──────────────────────────────────────────────────────────────
df_out = df_test[selected_features + [target]].copy()
if "molecule" in df_test.columns:
    df_out.insert(0, "molecule", df_test["molecule"].values)
df_out["y_pred"]          = y_pred
df_out["std_pred"]        = std_pred
df_out["intercept"]       = intercepts   # baseline: predicted ΔG for average molecule
df_out["dom_feature"]     = [selected_features[i] for i in dominant_idx]
df_out["dominance_ratio"] = dominance_ratios
for i, feat in enumerate(selected_features):
    df_out[f"attr_{feat}"] = attr_matrix[:, i]
# Sanity column: attr_sum + intercept should equal y_pred to floating point
df_out["attr_sum"]        = attr_matrix.sum(axis=1)
df_out["budget_check"]    = np.isclose(df_out["attr_sum"] + df_out["intercept"], df_out["y_pred"])


df_out.to_csv("results/attribution_results_freesolv.csv", index=False)
print("Saved → freesolv/attribution_results.csv")
print("\nDominant feature distribution:")
print(df_out["dom_feature"].value_counts().to_string())