import numpy as np
import torch
import json
import pickle
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")

class AnomalyDetector:
    def __init__(self, model_type="sparse"):
        """model_type: 'basic', 'deep', or 'sparse' — matches your 3 trained variants"""
        self.model_type = model_type
        self.model = self._load_model()
        self.threshold = self._load_threshold()
        self.preprocessor = self._load_preprocessor()

    def _load_model(self):
        model_path = ARTIFACTS_DIR / "models" / f"{self.model_type}_autoencoder.pt"
        model = torch.load(model_path, map_location="cpu")
        model.eval()
        return model

    def _load_threshold(self):
        threshold_path = ARTIFACTS_DIR / "thresholds" / f"{self.model_type}_threshold.json"
        with open(threshold_path) as f:
            data = json.load(f)
        return data["p95_threshold"]  # adjust key name to match your actual JSON

    def _load_preprocessor(self):
        prep_path = ARTIFACTS_DIR / "kdd99_preprocessed_data.npz"
        # If you saved a separate scaler/encoder object, load it here instead
        return None

    def predict(self, sample: list[float]) -> dict:
        x = torch.tensor([sample], dtype=torch.float32)
        with torch.no_grad():
            reconstruction = self.model(x)
            error = torch.mean((x - reconstruction) ** 2).item()
        is_anomaly = error > self.threshold
        return {
            "reconstruction_error": error,
            "threshold": self.threshold,
            "is_anomaly": bool(is_anomaly),
            "confidence": min(error / self.threshold, 5.0)  # capped ratio, simple confidence proxy
        }