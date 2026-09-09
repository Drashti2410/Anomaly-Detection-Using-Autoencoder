from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
HEADERS = {"x-api-key": "dev-secret-key"}

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_predict_missing_key():
    r = client.post("/predict", json={"features": [0.1]*41, "model_type": "sparse"})
    assert r.status_code == 422  # missing header

def test_predict_invalid_model_type():
    r = client.post("/predict",
        json={"features": [0.1]*41, "model_type": "invalid"},
        headers=HEADERS)
    assert r.status_code == 422  # pydantic pattern validation catches it

def test_predict_valid(monkeypatch):
    # Mock detector to avoid needing real artifacts in CI
    from api import main
    class FakeDetector:
        def predict(self, features):
            return {"reconstruction_error": 0.02, "threshold": 0.05,
                     "is_anomaly": False, "confidence": 0.4}
    main.detectors["sparse"] = FakeDetector()
    r = client.post("/predict",
        json={"features": [0.1]*41, "model_type": "sparse"},
        headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["is_anomaly"] is False