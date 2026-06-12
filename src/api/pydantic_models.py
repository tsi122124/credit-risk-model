from typing import List

from pydantic import BaseModel, field_validator


class PredictionRequest(BaseModel):
    features: List[float]

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        if len(v) != 56:
            raise ValueError(
                "Exactly 56 features are required"
            )
        return v


class PredictionResponse(BaseModel):
    risk_probability: float
    prediction: int
