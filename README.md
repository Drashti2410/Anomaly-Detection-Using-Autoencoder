# Anomaly Detection with Autoencoders (KDD Cup 1999)

SEP 740: Deep Learning - Final Project

This repository implements and compares three autoencoder architectures for
unsupervised network-intrusion / anomaly detection on the **KDD Cup 1999**
dataset. All models are trained **only on normal traffic** and use the
**reconstruction error** as an anomaly score: samples whose reconstruction
error exceeds a calibrated threshold are flagged as anomalies.

Three models are provided:


| Model                            | Framework          | Code folder           |
| -------------------------------- | ------------------ | --------------------- |
| Basic Autoencoder (baseline)     | PyTorch            | `src/`                |
| Deep Autoencoder                 | TensorFlow / Keras | `deep_autoencoder/`   |
| Sparse Autoencoder (KL sparsity) | PyTorch            | `sparse_autoencoder/` |


All three share the same preprocessed data, the same evaluation protocol
(percentile thresholds calibrated on normal traffic; primary threshold = **p95**),
and the same evaluation metrics (precision, recall, F1, confusion matrix, and
# SEP740 — Autoencoder-based Anomaly Detection (KDD'99)

Anomaly detection on the KDD Cup 1999 dataset using basic, deep, and sparse
autoencoders — end-to-end preprocessing, training, threshold calibration, and
evaluation with reproducible artifacts and experiments.

---

## Overview

This repository implements three reconstruction-based anomaly detectors:
1) a basic autoencoder, 2) a deep autoencoder (Keras/TensorFlow), and 3) a
sparse autoencoder (PyTorch) with a KL-divergence sparsity penalty. Models are
trained only on normal traffic and use reconstruction error with calibrated
percentile thresholds to flag anomalies. The project emphasizes reproducibility
— preprocessing artifacts, model weights, thresholds, metrics, and figures are
stored under `artifacts/` so results can be inspected or reproduced without
retraining.

---

## Key Features

- End-to-end pipeline: raw KDD'99 → preprocessing → model training →
  threshold calibration → evaluation → visualization
- Three model variants: basic (PyTorch), deep (Keras/TensorFlow), sparse (PyTorch)
- Hyperparameter sweep scripts for sparse-autoencoder sparsity parameters
- Saved artifacts for reproducibility: preprocessor, model weights, thresholds,
  evaluation metrics, reconstruction errors, and figures

---

## Repository Layout (important folders)

- `dataset/` — raw KDD'99 files required by the preprocessing notebook
- `data_preprocessing.ipynb` — preprocessing notebook that produces
  `artifacts/kdd99_preprocessed_data.npz`
- `src/` — basic autoencoder pipeline and utilities
- `deep_autoencoder/` — training, calibration, evaluation for deep AE
- `sparse_autoencoder/` — training, calibration, experiments for sparse AE
- `artifacts/` — preprocessed data, models, thresholds, metrics, figures
- `requirements.txt` — pinned Python dependencies used for development

Refer to the in-repo scripts for exact invocation and config options.

---

## Quickstart (local)

1. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. (Optional) If you need to regenerate preprocessed data, run the preprocessing
   notebook `data_preprocessing.ipynb` (JupyterLab or nbconvert execute).

3. Run one of the model pipelines from the repository root.

- Basic Autoencoder (single-step pipeline):

```bash
cd src
python run_basic_autoencoder_pipeline.py
```

- Deep Autoencoder:

```bash
cd deep_autoencoder
python train_final_deep_autoencoder.py
python calibrate_deep_threshold.py
python evaluate_deep_autoencoder.py
python visualize_deep_results.py
```

- Sparse Autoencoder:

```bash
cd sparse_autoencoder
python train_sparse_autoencoder.py
python calibrate_sparse_threshold.py
python evaluate_sparse_autoencoder.py
python visualize_sparse_results.py
```

Notes:
- Each model writes outputs under `artifacts/` using model-specific prefixes.
- Training scripts use a fixed seed for repeatability. Some non-determinism may
  remain when using GPU acceleration.

---

## Artifacts & Outputs

- `artifacts/kdd99_preprocessed_data.npz` — single preprocessed dataset used by
  all models (contains train/validation/calibration/test splits)
- `artifacts/models/` — saved model weights (PyTorch `.pt`, Keras `.h5`/`.json`)
- `artifacts/thresholds/` — JSON files with calibrated percentile thresholds
- `artifacts/evaluation/` — JSON/CSV metrics and saved reconstruction errors
- `artifacts/figures/` — PNGs used in the project report and analysis

If you only want to reproduce evaluation results, the necessary preprocessed
data and model artifacts are already included so you can skip costly retraining.

---

## Experiments and Hyperparameter Sweeps

The `sparse_autoencoder/` folder includes scripts for sweeping the sparsity
weight (`beta`) and sparsity target (`rho`). Sweep outputs and aggregated
results are placed in `artifacts/experiments/` as CSV/JSON and plotted figures.

---

## Contributing

Contributions are welcome. Suggested ways to help:

- Add unit tests and CI (GitHub Actions)
- Add Dockerfile or reproducible container for experiments
- Expand dataset support or add additional anomaly detection baselines
- Improve documentation and add a short tutorial notebook

Please open issues or PRs; include reproducible steps and small, focused
changes when possible.

---

## Contact / Citation

If you use this code in research, please cite the repository and include a link
to this project. For questions or collaboration, open an issue or contact the
maintainer listed in repository metadata.
