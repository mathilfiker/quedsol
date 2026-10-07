# QUEDSol: QUantum Electronic Descriptors in SOLution

Repository containing the **QUEDSol** framework as presented in the publication:

> *"From Vacuum Geometry to Solution-Phase Electronic Properties and Thermodynamics via Equivariant Neural Networks."*

QUEDSol allows the user to rapidly generate electronic descriptors representative of the **solvated phase** starting from **gas-phase coordinates**.

---

## Repository Structure

- **`enn/`** — Property prediction models trained with *MACE* on the 70-set. All other models (*Allegro* and smaller training sets) are available for download from the associated **Zenodo** repository.
- **`models_analysis/`** — Scripts to reproduce the results presented in the original publication.
- **`scripts/`** — All scripts necessary to employ the QUEDSol framework, for both hydration free energy predictions and custom studies.

---

## Requirements

- **MACE-torch** — Required to run the scripts. See the [official installation guide](https://mace-docs.readthedocs.io/en/latest/guide/installation.html).
- **SHAP** — Required for the feature importance analysis. See the [SHAP documentation](https://shap.readthedocs.io/en/latest/).
- **SciPy** — We recommend updating to any version **> 1.9**.

---

## Citation

If you use QUEDSol in your research, please cite the original publication:

```bibtex
@article{QUEDSol2026,
  title   = {From Vacuum Geometry to Solution-Phase Electronic Properties and Thermodynamics via Equivariant Neural Networks},
  author  = {Author, A. and Author, B. and Author, C.},
  journal = {Journal Name},
  year    = {2026},
  volume  = {XX},
  pages   = {XXX--XXX},
  doi     = {10.XXXX/XXXXXX}
}
```

Please also cite the associated **Zenodo** repository if you use the downloadable model weights.

