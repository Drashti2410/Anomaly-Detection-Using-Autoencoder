from fastapi import FastAPI, HTTPException, Header, Depends
from api.schemas import PredictRequest, PredictResponse
from api.inference import AnomalyDetector
import logging, time, os

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Network Anomaly Detection API")

API_KEY = os.getenv("API_KEY", "dev-secret-key")
detectors = {}  # cache loaded models per type

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def get_detector(model_type: str) -> AnomalyDetector:
    if model_type not in detectors:
        try:
            detectors[model_type] = AnomalyDetector(model_type)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=f"Model artifacts not found: {e}")
    return detectors[model_type]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _=Depends(verify_api_key)):
    detector = get_detector(req.model_type)
    start = time.time()
    try:
        result = detector.predict(req.features)
    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
    latency = time.time() - start
    logging.info(f"model={req.model_type} error={result['reconstruction_error']:.4f} "
                 f"anomaly={result['is_anomaly']} latency={latency:.3f}s")
    if latency > 0.5:
        logging.warning(f"High latency: {latency:.3f}s")
    return result

@app.get("/models")
def list_models():
    return {"available": ["basic", "deep", "sparse"], "loaded": list(detectors.keys())}