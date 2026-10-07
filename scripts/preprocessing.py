#!/usr/bin/env python
"""
preprocessing.py

Unified workflow:
    1. Parse SMILES from an input CSV (column: 'smiles', optional 'name'),
       generate 3D structures and conformers in-memory.
       -- OR --
       Read pre-computed conformer SDFs from a user-supplied directory.
    2. Compute descriptors (MACE-based + geometric + RDKit) per molecule.
    3. Write a single processed_data CSV.

Usage:
    python preprocessing.py --smiles-csv input.csv --output processed_data.csv
    python preprocessing.py --confs-dir  my_confs/ --output processed_data.csv
    python preprocessing.py --smiles-csv input.csv --save-confs-dir mols_w_confs/
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski, Crippen, rdFreeSASA
from ase import Atoms

import os
import contextlib
import io
import logging
import warnings

# Silence MACE / torch chatter
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")
logging.getLogger("mace").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
SUPPORTED_ELEMENTS = {"H", "C", "N", "O", "F", "S", "Cl"}

CLASSIC_RADII = {
    1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
    15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98,
}

MODEL_DIR = "../enn"

MODEL_FILES = {
    "dip":       ("dip_direct_70.model",  "DipoleMACE"),
    "mbd":       ("mbd_nodelta_70.model", None),
    "pol":       ("pol_nodelta_70.model", None),
    "gap":       ("gap_nodelta_70.model", None),
    "eat":       ("eat_nodelta_70.model", None),
    "dip_delta": ("dip_delta_70.model",   "DipoleMACE"),
    "mbd_delta": ("mbd_delta_70.model",   None),
    "pol_delta": ("pol_delta_70.model",   None),
    "gap_delta": ("gap_delta_70.model",   None),
    "eat_delta": ("eat_delta_70.model",   None),
}

# -------------------------------------------------------------------------
# Helpers: element check, geometric descriptors
# -------------------------------------------------------------------------
def unsupported_elements(mol):
    return {a.GetSymbol() for a in mol.GetAtoms()} - SUPPORTED_ELEMENTS

def radii_list(mol):
    return [CLASSIC_RADII.get(a.GetAtomicNum(), 1.50) for a in mol.GetAtoms()]

def get_gyr(atoms):
    positions = atoms.get_positions()
    masses = atoms.get_masses()
    M_total = masses.sum()
    com = atoms.get_center_of_mass()
    disp = positions - com
    sq = np.sum(disp**2, axis=1)
    return np.sqrt(np.dot(masses, sq) / M_total)

def inertia_tensor(coords, masses):
    M = masses.sum()
    com = np.sum(coords * masses[:, None], axis=0) / M
    x = coords - com
    I = np.zeros((3, 3))
    for i in range(len(masses)):
        r, m = x[i], masses[i]
        I += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return I

def kappa2(coords, masses):
    eigvals = np.linalg.eigvalsh(inertia_tensor(coords, masses))
    l1, l2, l3 = eigvals[::-1]
    return 1.0 - 3.0 * (l1*l2 + l2*l3 + l3*l1) / (l1 + l2 + l3)**2

# -------------------------------------------------------------------------
# Step 1a: SMILES -> RDKit Mol with conformers (in-memory)
# -------------------------------------------------------------------------
def build_mol_with_confs(smiles, name, num_confs=200):
    """Return (rdkit_mol_with_confs, n_rot_bonds, n_confs) or None if failed/unsupported."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[SKIP] {name}: invalid SMILES")
        return None

    bad = unsupported_elements(mol)
    if bad:
        print(f"[SKIP] {name}: contains unsupported element(s) {bad}. "
              f"Supported = {sorted(SUPPORTED_ELEMENTS)}")
        return None

    mol = Chem.AddHs(mol)

    # initial embed
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        print(f"[SKIP] {name}: initial embedding failed")
        return None
    AllChem.UFFOptimizeMolecule(mol)
    mol.SetProp("SMILES", smiles)
    mol.SetProp("_Name", name)

    n_rot = Lipinski.NumRotatableBonds(mol)
    print(f"[{name}] rotatable bonds: {n_rot}")

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.pruneRmsThresh = 0.2
    params.numThreads = 0
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)

    if len(conf_ids) == 0:
        print(f"[SKIP] {name}: multi-conformer embedding failed")
        return None

    AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
    print(f"[{name}] generated {len(conf_ids)} conformers")

    if len(conf_ids) == 1:
        print(f"[{name}] warning: only 1 conformer generated")

    return mol, n_rot, len(conf_ids)

def save_confs_sdf(mol, name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_dir / f"{name}.sdf"))
    for conf in mol.GetConformers():
        writer.write(mol, confId=conf.GetId())
    writer.close()

# -------------------------------------------------------------------------
# Step 1b: read pre-computed conformer SDFs
# -------------------------------------------------------------------------
def load_mols_from_dir(confs_dir):
    """Yield (name, list_of_mols, n_rot_bonds, n_confs) from a directory of SDF files."""
    confs_dir = Path(confs_dir)
    sdf_files = sorted(confs_dir.glob("*.sdf"))
    if not sdf_files:
        raise FileNotFoundError(f"No .sdf files in {confs_dir}")
    for sdf in sdf_files:
        name = sdf.stem
        suppl = [m for m in Chem.SDMolSupplier(str(sdf), removeHs=False) if m is not None]
        if not suppl:
            print(f"[SKIP] {name}: empty/invalid SDF")
            continue
        bad = unsupported_elements(suppl[0])
        if bad:
            print(f"[SKIP] {name}: contains unsupported element(s) {bad}")
            continue
        n_rot = Lipinski.NumRotatableBonds(suppl[0])
        yield name, suppl, n_rot, len(suppl)

# -------------------------------------------------------------------------
# Step 2: MACE calculators (with CUDA fallback)
# -------------------------------------------------------------------------
def build_calculators():
    from mace.calculators import MACECalculator
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    if device == "cpu":
        warnings.warn(
            "CUDA not available - falling back to CPU. "
            "MACE inference will be significantly slower.",
            RuntimeWarning,
        )
    print(f"[MACE] using device: {device}")

    calcs = {}
    buf = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        warnings.simplefilter("ignore")
        for key, (fname, mtype) in MODEL_FILES.items():
            kwargs = dict(
                model_paths=f"{MODEL_DIR}/{fname}",
                device=device,
                default_dtype="float32",
            )
            if mtype is not None:
                kwargs["model_type"] = mtype
            calcs[key] = MACECalculator(**kwargs)
    return calcs

def _energy(atoms, calc):
    atoms.calc = calc
    return atoms.get_total_energy()

def _dipole(atoms, calc):
    atoms.calc = calc
    return atoms.get_dipole_moment()

# -------------------------------------------------------------------------
# Step 3: descriptor computation for a single molecule (all conformers)
# -------------------------------------------------------------------------
def compute_descriptors(name, mols, n_rot, n_conf, calcs):
    """`mols` is either an RDKit Mol with multiple confs OR a list of Mols (each 1 conf)."""
    # Normalize into a per-conformer iterable of (rdkit_mol, conformer)
    if isinstance(mols, Chem.Mol):
        per_conf = [(mols, c) for c in mols.GetConformers()]
    else:
        per_conf = [(m, m.GetConformer(0)) for m in mols]

    ref = per_conf[0][0]
    num_dons = Lipinski.NumHDonors(ref)
    num_acc  = Lipinski.NumHAcceptors(ref)
    logP     = Crippen.MolLogP(ref)

    k2_ens, gyr_ens, sasa_ens = [], [], []
    dip_ens, mbd_ens, pol_ens, gap_ens, eat_ens = [], [], [], [], []
    dip_ens_d, mbd_ens_d, pol_ens_d, gap_ens_d, eat_ens_d = [], [], [], [], []
    massa = None

    for rd_mol, conf in per_conf:
        atomic_nums = [a.GetAtomicNum() for a in rd_mol.GetAtoms()]
        masses = np.array([a.GetMass() for a in rd_mol.GetAtoms()])
        pos = conf.GetPositions()
        atoms = Atoms(symbols=atomic_nums, positions=pos)

        k2_ens.append(kappa2(pos, masses))
        gyr_ens.append(get_gyr(atoms))
        massa = atoms.get_masses().sum()

        dip_ens.append(_dipole(atoms, calcs["dip"]))
        mbd_ens.append(_energy(atoms, calcs["mbd"]))
        pol_ens.append(_energy(atoms, calcs["pol"]))
        gap_ens.append(_energy(atoms, calcs["gap"]))
        eat_ens.append(_energy(atoms, calcs["eat"]))

        dip_ens_d.append(_dipole(atoms, calcs["dip_delta"]))
        mbd_ens_d.append(_energy(atoms, calcs["mbd_delta"]))
        pol_ens_d.append(_energy(atoms, calcs["pol_delta"]))
        gap_ens_d.append(_energy(atoms, calcs["gap_delta"]))
        eat_ens_d.append(_energy(atoms, calcs["eat_delta"]))

        sasa_ens.append(rdFreeSASA.CalcSASA(rd_mol, radii_list(rd_mol)))

    dip_ens    = np.array(dip_ens)
    dip_ens_d  = np.array(dip_ens_d)
    sasa_ens   = np.array(sasa_ens)

    mean_mu       = np.nanmean(np.linalg.norm(dip_ens, axis=1))
    mean_mu_delta = np.nanmean(np.linalg.norm(dip_ens_d, axis=1))
    mean_sasa     = sasa_ens.mean()
    sigma_sasa    = sasa_ens.std()

    return {
        "molecule": name,
        "mu": mean_mu,
        "mbd": np.mean(mbd_ens),
        "alpha": np.mean(pol_ens),
        "gap": np.mean(gap_ens),
        "eat": np.mean(eat_ens),
        "delta_mu": mean_mu_delta,
        "delta_mbd": np.mean(mbd_ens_d),
        "delta_alpha": np.mean(pol_ens_d),
        "delta_gap": np.mean(gap_ens_d),
        "delta_eat": np.mean(eat_ens_d),
        "r_g": np.mean(gyr_ens),
        "mol_mass": massa,
        "kappa2": np.mean(k2_ens),
        "sigma_k2": np.std(k2_ens),
        "logP": logP,
        "sasa": mean_sasa,
        "sigma_sasa": sigma_sasa,
        "rel_sigma_sasa": sigma_sasa / mean_sasa if mean_sasa else np.nan,
        "rotatable_bonds": n_rot,
        "n_conf": n_conf,
        "H_donors": num_dons,
        "H_acceptors": num_acc,
    }

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Descriptor preprocessing pipeline.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--smiles-csv", type=str,
                     help="CSV file with a 'smiles' column (optional 'name' column).")
    src.add_argument("--confs-dir", type=str,
                     help="Directory of pre-computed conformer SDFs (one file per molecule).")
    p.add_argument("--save-confs-dir", type=str, default=None,
                   help="If set (only with --smiles-csv), save generated conformer SDFs here.")
    p.add_argument("--num-confs", type=int, default=200,
                   help="Number of conformers to generate from SMILES (default 200).")
    p.add_argument("--output", type=str, default="processed_data.csv",
                   help="Output CSV path.")
    return p.parse_args()

def iter_molecules(args):
    """Yield (name, mols_or_mol, n_rot, n_conf) tuples."""
    if args.confs_dir:
        yield from load_mols_from_dir(args.confs_dir)
    else:
        df = pd.read_csv(args.smiles_csv)
        if "smiles" not in df.columns:
            raise ValueError("Input CSV must contain a 'smiles' column.")
        names = (df["name"].astype(str).values
                 if "name" in df.columns
                 else [f"mol_{i}" for i in range(len(df))])
        for name, smi in zip(names, df["smiles"].values):
            built = build_mol_with_confs(smi, name, num_confs=args.num_confs)
            if built is None:
                continue
            mol, n_rot, n_conf = built
            if args.save_confs_dir:
                save_confs_sdf(mol, name, args.save_confs_dir)
            yield name, mol, n_rot, n_conf

def main():
    args = parse_args()
    if args.save_confs_dir and args.confs_dir:
        warnings.warn("--save-confs-dir is ignored when --confs-dir is provided.")

    calcs = build_calculators()

    results = []
    for name, mols, n_rot, n_conf in iter_molecules(args):
        print(f"[DESCRIPTORS] {name}")
        sys.stdout.flush()
        try:
            results.append(compute_descriptors(name, mols, n_rot, n_conf, calcs))
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    if not results:
        print("No molecules processed - nothing to write.")
        return
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"Wrote {len(results)} rows to {args.output}")

if __name__ == "__main__":
    main()