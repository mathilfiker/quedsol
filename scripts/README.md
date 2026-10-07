# QUEDSol

Scripts to employ the **QUEDSol** framework.

## Overview

QUEDSol provides a pipeline for generating condensed-phase electronic descriptors from molecular structures and using them to predict solvation-related thermodynamic properties. The framework consists of four main scripts:

- **`preprocessing.py`** — Generates conformer-averaged electronic descriptors from SMILES.
- **`predict_ahfe.py`** — Predicts hydration free energies using pre-trained models.
- **`train.py`** — Trains custom models on QUEDSol descriptors for arbitrary target properties.
- **`feature_importance.py`** — Analyzes feature relevance to guide feature selection prior to training.

---

## `preprocessing.py`

The heart of the framework. It produces condensed-phase electronic descriptors averaged over conformations.

**Input:** A `.csv` file containing SMILES strings.

**Behavior:** The script automatically generates conformers using RDKit's *ETKDGv3* method. Conformers can optionally be saved to a user-specified directory. Alternatively, the user can supply their own conformer set by pointing to a directory containing a single `.sdf` file per molecule (each embedding all conformers for that molecule).

**Output:** A `.csv` file containing averaged global descriptors for each molecule.

### Usage

Run on a set of SMILES without saving the generated conformers:

```bash
python3 preprocessing.py --smiles-csv example_smiles.csv --output processed_data.csv
```

Run on a set of SMILES and save the generated conformers:

```bash
python3 preprocessing.py --smiles-csv example_smiles.csv --output processed_data.csv --save-confs-dir mols_w_confs/
```

Run on a set of pre-generated conformers:

```bash
python3 preprocessing.py --confs-dir mols_w_confs/ --output processed_data.csv
```

---

## `predict_ahfe.py`

The hydration free energy predictor presented in the original publication. It takes as input a `.csv` file produced by `preprocessing.py` and returns a `predictions.csv` file containing the predicted **ΔG** for each molecule.

**Available models:**

- `poly` — Polynomial regression *(default, recommended)*
- `xgb` — XGBoost
- `svr` — Support Vector Regression

The three pre-trained models are stored in **`ahfe_models/`**. The script automatically selects the appropriate subset of features from `processed_data.csv`, as described in the publication.

### Usage

```bash
python3 predict_ahfe.py --model poly --input processed_data.csv
```

When `poly` is selected, the script runs **feature attribution** by default. To disable this (recommended for large datasets), use `--no-attribution`. Attribution analysis is currently not supported for `xgb` or `svr`.

---

## `train.py`

A pre-built utility for training models on QUEDSol descriptors to learn properties beyond hydration free energy.

**Supported models:**

- `xgb` — XGBoost, optimized with *Optuna*.
- `svr` — SVR, optimized with `GridSearchCV`.
- `poly` — Polynomial regression as a 10-fold ensemble of regressors.

Alternatively, `processed_data.csv` can be used to train any custom model with a user-defined script.

**Requirements:** The processed dataset must be enriched with a column containing the target property. Optionally, a test set can be provided, in which case the script reports the *mean absolute error* and *root mean squared error* on it. Trained models are saved to **`trained_models/`**.

> **Recommendation:** Because preprocessing returns many descriptors — not all of which are useful for every application — it is recommended to first run `feature_importance.py` on the training set to guide feature selection.

### Usage

```bash
python3 train.py --model poly --train train.csv --test test.csv --target dG --exclude "sigma_sasa, kappa2"
```

The `--target` name must match the corresponding column in `train.csv`. Both `--exclude` and `--test` are optional.

---

## `feature_importance.py`

Runs four complementary analyses on the training set:

- **Linear correlation** with the target.
- **LASSO coefficients.**
- **Random Forest** permutation importance.
- **XGBoost SHAP** analysis.

Results are summarized in two figures saved to the **`feature_importance/`** folder, which can be used to decide which descriptors to retain.

### Usage

```bash
python3 feature_importance.py --input train.csv --target dG
```
