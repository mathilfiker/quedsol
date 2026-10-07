import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import mean_absolute_error

# =========================
# CONSTANTS
# =========================
eV_TO_KCAL = 23.062063375661

# Which properties to convert eV -> kcal/mol on print (same as original script)
PROPS = {
    "eat": True,
    "mbd": True,
    "gap": True,
    "pol": False,
    "dip": False,
}
KS = (30, 40, 50, 70)

FILE_ALLEGRO = "evaluations/predictions_allegro.csv"
FILE_MACE    = "evaluations/predictions_mace.csv"

# -----------------------
# Scalar-property MAEs (eat, mbd, gap, pol; also Allegro's dipole scalars)
# Delta reconstruction: gas + predicted_delta
# -----------------------
def mae_scalar(df: pd.DataFrame, prop: str):
    gas  = df[f"{prop}_gas"].to_numpy()
    real = df[f"{prop}_real"].to_numpy()
    mae_delta, mae_direct = [], []
    for k in KS:
        pd_ = df[f"{prop}_pred_delta_{k}"].to_numpy()
        pi_ = df[f"{prop}_pred_direct_{k}"].to_numpy()
        mae_delta.append(mean_absolute_error(real, gas + pd_))   # reconstruction
        mae_direct.append(mean_absolute_error(real, pi_))
    return mae_delta, mae_direct

# -----------------------
# MACE dipole MAEs
# Real target is the magnitude of the real vector: |real_vec|
# Delta reconstruction: |gas_vec + pred_delta_vec|
# Direct prediction:   |pred_direct_vec|
# -----------------------
def mae_dipole_mace(df: pd.DataFrame):
    gas_vec  = df[["dip_gas_x",  "dip_gas_y",  "dip_gas_z" ]].to_numpy()
    real_vec = df[["dip_real_x", "dip_real_y", "dip_real_z"]].to_numpy()
    real_mag = norm(real_vec, axis=1)

    mae_delta, mae_direct = [], []
    for k in KS:
        pd_vec = df[[f"dip_pred_delta_{k}_x",
                     f"dip_pred_delta_{k}_y",
                     f"dip_pred_delta_{k}_z"]].to_numpy()
        pi_vec = df[[f"dip_pred_direct_{k}_x",
                     f"dip_pred_direct_{k}_y",
                     f"dip_pred_direct_{k}_z"]].to_numpy()

        recon_mag = norm(gas_vec + pd_vec, axis=1)   # |gas + pred_delta|
        direct_mag = norm(pi_vec, axis=1)            # |pred_direct|

        mae_delta.append(mean_absolute_error(real_mag, recon_mag))
        mae_direct.append(mean_absolute_error(real_mag, direct_mag))
    return mae_delta, mae_direct

# -----------------------
# Pretty printing
# -----------------------
def fmt(vals, convert):
    factor = eV_TO_KCAL if convert else 1.0
    return [f"{v * factor:.6f}" for v in vals]

# -----------------------
# Main
# -----------------------
def main():
    df_allegro = pd.read_csv(FILE_ALLEGRO)
    df_mace    = pd.read_csv(FILE_MACE)

    for prop, convert in PROPS.items():
        unit = "kcal/mol" if convert else "native"
        print(f"------ {prop.upper()} (units: {unit}, Training sets = {list(KS)}) ------")

        if prop == "dip": 
            gas_base = norm(df_mace[["dip_gas_x", "dip_gas_y", "dip_gas_z" ]].to_numpy(), axis=1) 
            real_base = norm(df_mace[["dip_real_x", "dip_real_y", "dip_real_z"]].to_numpy(), axis=1) 
        else: 
            gas_base = df_mace[f"{prop}_gas"].to_numpy() 
            real_base = df_mace[f"{prop}_real"].to_numpy() 
        print(f"Baseline: {fmt([mean_absolute_error(real_base, gas_base)], convert)[0]}")
        # Allegro (dipole here is scalar in Allegro's CSV → use mae_scalar)
        a_delta, a_direct = mae_scalar(df_allegro, prop)
        print(f"Allegro delta : {fmt(a_delta,  convert)}")
        print(f"Allegro direct: {fmt(a_direct, convert)}")

        # MACE (dipole uses vector reconstruction)
        if prop == "dip":
            m_delta, m_direct = mae_dipole_mace(df_mace)
        else:
            m_delta, m_direct = mae_scalar(df_mace, prop)
        print(f"MACE    delta : {fmt(m_delta,  convert)}")
        print(f"MACE    direct: {fmt(m_direct, convert)}")
        print()

if __name__ == "__main__":
    main()