import pandas as pd
import numpy as np
from numpy.linalg import norm

# =========================
# CONSTANTS
# =========================
eV_TO_KCAL = 23.062063375661

# Which properties to convert eV -> kcal/mol before computing MAPE.
# NOTE: for MAPE, unit conversion cancels out in the ratio (pred/true),
# so the printed values are the same whether we convert or not.
# We keep the flag for consistency with the previous scripts.
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
# MAPE as defined by the user:
#   mean( 100 * |pred - true| / true )
# -----------------------
def mape(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(100.0 * np.abs((pred - true) / true)))

# -----------------------
# Scalar-property MAPEs (eat, mbd, gap, pol; also Allegro's dipole scalars)
# Delta reconstruction: gas + predicted_delta
# -----------------------
def mape_scalar(df: pd.DataFrame, prop: str):
    gas  = df[f"{prop}_gas"].to_numpy()
    real = df[f"{prop}_real"].to_numpy()
    err_delta, err_direct = [], []
    for k in KS:
        pd_ = df[f"{prop}_pred_delta_{k}"].to_numpy()
        pi_ = df[f"{prop}_pred_direct_{k}"].to_numpy()
        err_delta.append(mape(real, gas + pd_))
        err_direct.append(mape(real, pi_))
    return err_delta, err_direct

# -----------------------
# MACE dipole MAPEs
# Target: |real_vec|;  delta reconstruction: |gas_vec + pred_delta_vec|
# Direct: |pred_direct_vec|
# -----------------------
def mape_dipole_mace(df: pd.DataFrame):
    gas_vec  = df[["dip_gas_x",  "dip_gas_y",  "dip_gas_z" ]].to_numpy()
    real_vec = df[["dip_real_x", "dip_real_y", "dip_real_z"]].to_numpy()
    real_mag = norm(real_vec, axis=1)

    err_delta, err_direct = [], []
    for k in KS:
        pd_vec = df[[f"dip_pred_delta_{k}_x",
                     f"dip_pred_delta_{k}_y",
                     f"dip_pred_delta_{k}_z"]].to_numpy()
        pi_vec = df[[f"dip_pred_direct_{k}_x",
                     f"dip_pred_direct_{k}_y",
                     f"dip_pred_direct_{k}_z"]].to_numpy()

        recon_mag  = norm(gas_vec + pd_vec, axis=1)
        direct_mag = norm(pi_vec, axis=1)

        err_delta.append(mape(real_mag, recon_mag))
        err_direct.append(mape(real_mag, direct_mag))
    return err_delta, err_direct

# -----------------------
# Pretty printing (unit conversion is a no-op for MAPE but kept for API parity)
# -----------------------
def fmt(vals):
    return [f"{v:.4f}%" for v in vals]

# -----------------------
# Main
# -----------------------
def main():
    df_allegro = pd.read_csv(FILE_ALLEGRO)
    df_mace    = pd.read_csv(FILE_MACE)

    for prop in PROPS.keys():
        print(f"------ {prop.upper()} (MAPE, Training sets = {list(KS)}) ------")
        if prop == "dip": 
            gas_base = norm(df_mace[["dip_gas_x", "dip_gas_y", "dip_gas_z" ]].to_numpy(), axis=1) 
            real_base = norm(df_mace[["dip_real_x", "dip_real_y", "dip_real_z"]].to_numpy(), axis=1) 
        else: 
            gas_base = df_mace[f"{prop}_gas"].to_numpy() 
            real_base = df_mace[f"{prop}_real"].to_numpy() 
        print(f"Baseline: {mape(real_base, gas_base):.4f}%")
        # Allegro: scalar treatment for all properties (including dipole)
        a_delta, a_direct = mape_scalar(df_allegro, prop)
        print(f"Allegro delta : {fmt(a_delta)}")
        print(f"Allegro direct: {fmt(a_direct)}")

        # MACE: dipole uses vector reconstruction; others scalar
        if prop == "dip":
            m_delta, m_direct = mape_dipole_mace(df_mace)
        else:
            m_delta, m_direct = mape_scalar(df_mace, prop)
        print(f"MACE    delta : {fmt(m_delta)}")
        print(f"MACE    direct: {fmt(m_direct)}")
        print()

if __name__ == "__main__":
    main()