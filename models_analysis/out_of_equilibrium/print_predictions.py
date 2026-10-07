import numpy as np
import pandas as pd
from numpy.linalg import norm

# =========================
# CONSTANTS
# =========================
eV_TO_KCAL = 23.062063375661

# Scaling factors used when printing (mirrors the original script)
SCALES = {
    "eat": eV_TO_KCAL,
    "gap": eV_TO_KCAL,
    "mbd": eV_TO_KCAL,
    "pol": 1.0,
    "dip": 1.0,
}

# Human-readable labels
LABELS = {
    "eat": "Atomization energy",
    "gap": "HOMO-LUMO gap",
    "mbd": "Dispersion energy (MBD)",
    "pol": "Polarizability",
    "dip": "Dipole",
}

# Which properties live in vector space
VECTOR = {"dip"}

SIZES = [30, 40, 50, 70]

# Molecules -> csv filename (must match Script 1)
MOLS = ["small", "small_2", "med", "med_2", "large", "large_2"]

# -----------------------
# Extract gas/solvent references and reconstruct predictions.
# For vector props: reconstruction is |gas_base_vec + pred_delta_vec|
# and direct is |pred_direct_vec|. References from .dat are scalar magnitudes.
# -----------------------
def prop_arrays(df: pd.DataFrame, prop: str):
    gas_ref     = df[f"{prop}_gas_ref"].to_numpy()
    solvent_ref = df[f"{prop}_solvent_ref"].to_numpy()

    delta_recon = {}
    direct_pred = {}

    if prop in VECTOR:
        gas_base = df[[f"{prop}_gas_base_x",
                       f"{prop}_gas_base_y",
                       f"{prop}_gas_base_z"]].to_numpy()
        for s in SIZES:
            d_vec = df[[f"{prop}_pred_delta_{s}_x",
                        f"{prop}_pred_delta_{s}_y",
                        f"{prop}_pred_delta_{s}_z"]].to_numpy()
            i_vec = df[[f"{prop}_pred_direct_{s}_x",
                        f"{prop}_pred_direct_{s}_y",
                        f"{prop}_pred_direct_{s}_z"]].to_numpy()
            delta_recon[s] = norm(gas_base + d_vec, axis=1)
            direct_pred[s] = norm(i_vec, axis=1)
    else:
        gas_base = df[f"{prop}_gas_base"].to_numpy()
        for s in SIZES:
            delta_recon[s] = gas_base + df[f"{prop}_pred_delta_{s}"].to_numpy()
            direct_pred[s] = df[f"{prop}_pred_direct_{s}"].to_numpy()

    return gas_ref, solvent_ref, delta_recon, direct_pred

# -----------------------
# Main
# -----------------------
def main():
    for mol in MOLS:
        df = pd.read_csv(f"evaluations/predictions_{mol}.csv")
        print(f"\n=========== MOLECULE: {mol} ===========")

        for prop, label in LABELS.items():
            scale = SCALES[prop]
            gas_ref, solvent_ref, delta_recon, direct_pred = prop_arrays(df, prop)

            mean_gas     = np.nanmean(gas_ref)
            mean_solvent = np.nanmean(solvent_ref)
            baseline_err = abs(mean_solvent - mean_gas)   # baseline: |<sol> - <gas>|

            print(f"\n--- {label} ---")
            print(f"Average gas phase : {scale * mean_gas:.4f}")
            print(f"Average solvent   : {scale * mean_solvent:.4f}")
            print(f"Baseline (|sol-gas|): {scale * baseline_err:.4f}")

            for s in SIZES:
                m_delta  = np.nanmean(delta_recon[s])
                m_direct = np.nanmean(direct_pred[s])
                err_delta  = abs(m_delta  - mean_solvent)
                err_direct = abs(m_direct - mean_solvent)
                print(f"  size {s}:")
                print(f"    delta  mean = {scale * m_delta:.4f}   error = {scale * err_delta:.4f}")
                print(f"    direct mean = {scale * m_direct:.4f}   error = {scale * err_direct:.4f}")

if __name__ == "__main__":
    main()