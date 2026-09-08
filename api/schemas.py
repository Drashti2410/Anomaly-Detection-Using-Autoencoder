from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="Preprocessed KDD'99 feature vector")
    model_type: str = Field(default="sparse", pattern="^(basic|deep|sparse)$")

class PredictResponse(BaseModel):
    reconstruction_error: float
    threshold: float
    is_anomaly: bool
    confidence: float