# SEP740 — Autoencoder-based Anomaly Detection (KDD'99)

Anomaly detection on the KDD Cup 1999 dataset using basic, deep, and sparse
autoencoders — end-to-end preprocessing, training, threshold calibration, and
evaluation with reproducible artifacts and experiments. The project now also
ships a **FastAPI inference service**, a **Docker image**, a **GitHub Actions
CI pipeline**, and a **PySpark preprocessing pipeline** for the full dataset.

SEP 740: Deep Learning — Final Project (Group 3)

---

## Overview

This repository implements three reconstruction-based anomaly detectors:
1) a basic autoencoder (PyTorch), 2) a deep autoencoder (Keras/TensorFlow), and
3) a sparse autoencoder (PyTorch) with a KL-divergence sparsity penalty. Models
are trained only on normal traffic and use reconstruction error with calibrated
percentile thresholds (primary threshold = **p95**) to flag anomalies. The
project emphasizes reproducibility — preprocessing artifacts, model weights,
thresholds, metrics, and figures are stored under `artifacts/` so results can be
inspected or reproduced without retraining.

---

## Key Features

- End-to-end pipeline: raw KDD'99 → preprocessing → model training →
  threshold calibration → evaluation → visualization
- Three model variants: basic (PyTorch), deep (Keras/TensorFlow), sparse (PyTorch)
- Hyperparameter sweep scripts for sparse-autoencoder sparsity parameters
- Saved artifacts for reproducibility: preprocessor, model weights, thresholds,
  evaluation metrics, reconstruction errors, and figures
- **REST inference API** (FastAPI) with API-key auth, health/metrics endpoints,
  and per-request latency logging
- **Containerized deployment** via a slim Python Docker image
- **CI pipeline** (GitHub Actions): pytest + Docker build + Trivy image scan
- **PySpark preprocessing pipeline** for the full ~5M-row KDD'99 dataset,
  writing Parquet that can be loaded back into NumPy for model training

---

## Repository Layout

- `dataset/` — raw KDD'99 files required by the preprocessing notebook
- `data_preprocessing.ipynb` — preprocessing notebook that produces
  `artifacts/kdd99_preprocessed_data.npz`
- `src/` — basic autoencoder pipeline and utilities
- `deep_autoencoder/` — training, calibration, evaluation for the deep AE
- `sparse_autoencoder/` — training, calibration, experiments for the sparse AE
- `api/` — FastAPI inference service
  - `main.py` — app, routes (`/health`, `/predict`, `/models`), API-key auth
  - `inference.py` — `AnomalyDetector`: loads model + threshold, scores samples
  - `schemas.py` — Pydantic request/response models
- `spark/` — PySpark preprocessing pipeline
  - `preprocess.py` — Spark ML pipeline (indexing, one-hot, assembling, scaling),
    writes `data/processed_parquet/`
  - `spark_train_prep.py` — converts the Parquet output to dense NumPy arrays
- `tests/` — pytest suite (`test_api.py`, `test_spark_preprocess.py`)
- `artifacts/` — preprocessed data, models, thresholds, metrics, figures
- `Dockerfile` — container image for the inference API
- `.github/workflows/ci.yml` — CI: tests, image build, security scan
- `requirements.txt` — pinned Python dependencies
- `pyproject.toml` / `uv.lock` / `.python-version` — `uv` project metadata (Python 3.13)

---

## Quickstart (local)

1. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

2. (Optional) If you need to regenerate preprocessed data, run the preprocessing
   notebook `data_preprocessing.ipynb` (JupyterLab or `nbconvert --execute`).

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

## Inference API

A FastAPI service wraps the trained autoencoders for online scoring. It loads
model weights from `artifacts/models/` and calibrated thresholds from
`artifacts/thresholds/` on first use, caching each detector per model type.

Run it locally:

```bash
export API_KEY=dev-secret-key          # default if unset
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

| Method | Path       | Auth        | Description                                    |
| ------ | ---------- | ----------- | --------------------------------------------- |
| GET    | `/health`  | none        | Liveness check                                 |
| GET    | `/models`  | none        | Available model types and which are loaded     |
| POST   | `/predict` | `x-api-key` | Score one preprocessed feature vector          |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "x-api-key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model_type": "sparse", "features": [0.1, 0.0, ... ]}'
```

Response:

```json
{
  "reconstruction_error": 0.0213,
  "threshold": 0.0500,
  "is_anomaly": false,
  "confidence": 0.43
}
```

`model_type` is one of `basic`, `deep`, `sparse`. `features` must be an already
preprocessed KDD'99 feature vector (same transformation as training).
`confidence` is a capped `error / threshold` ratio used as a simple proxy.

---

## Docker

Build and run the API in a container:

```bash
docker build -t anomaly-api .
docker run -p 8000:8000 -e API_KEY=your-secret anomaly-api
```

The image is based on `python:3.11-slim` and copies `api/` and `artifacts/`
into the container. It starts `uvicorn api.main:app` on port 8000.

---

## Continuous Integration

`.github/workflows/ci.yml` runs on every push:

1. Install dependencies and run `pytest tests/`
2. Build the Docker image (`anomaly-api`)
3. Scan the image with [Trivy](https://github.com/aquasecurity/trivy-action)

Run the tests locally:

```bash
pytest tests/
```

---

## PySpark Preprocessing (full dataset)

For the full ~5M-row KDD'99 file, `spark/` provides a Spark ML preprocessing
pipeline that mirrors the notebook transformation at scale:

```bash
# 1. Build features and write Parquet (data/processed_parquet/)
python spark/preprocess.py

# 2. Convert Parquet features to dense NumPy arrays for model training
python spark/spark_train_prep.py
#    -> artifacts/X_spark_processed.npy, artifacts/y_spark_processed.npy
```

The pipeline string-indexes and one-hot-encodes the categorical columns
(`protocol_type`, `service`, `flag`), assembles them with the numeric columns,
and standardizes the result. `data/processed_parquet/` is git-ignored.

---

## Artifacts & Outputs

- `artifacts/kdd99_preprocessed_data.npz` — single preprocessed dataset used by
  all models (train/validation/calibration/test splits)
- `artifacts/kdd99_preprocessor.joblib` / `kdd99_preprocessing_metadata.json` —
  fitted preprocessor and its metadata
- `artifacts/models/` — saved weights: `basic_autoencoder.pt`,
  `sparse_autoencoder.pt`, `deep_autoencoder.weights.h5` / `.json`
- `artifacts/thresholds/` — JSON files with calibrated percentile thresholds
- `artifacts/evaluation/` — JSON/CSV metrics and saved reconstruction errors
- `artifacts/figures/` — PNGs used in the project report and analysis
- `artifacts/experiments/` , `artifacts/hyperparameter_search/` ,
  `artifacts/training_history/` — sweep results and training curves

If you only want to reproduce evaluation results, the necessary preprocessed
data and model artifacts are already included so you can skip retraining.

---

## Experiments and Hyperparameter Sweeps

The `sparse_autoencoder/` folder includes scripts for sweeping the sparsity
weight (`beta`) and sparsity target (`rho`). Sweep outputs and aggregated
results are placed in `artifacts/experiments/` as CSV/JSON and plotted figures.

---

## Reports

- `Group3_final_detail_report.pdf` — full project report
- `Group3_IEEE_conference_paper.pdf` — IEEE-format conference paper
- `SEP740_DeepLearning_Group3_PresentationSlides.pdf` — presentation slides

---

## Contact / Citation

If you use this code in research, please cite the repository and include a link
to this project. For questions or collaboration, open an issue.
