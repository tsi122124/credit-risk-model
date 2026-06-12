import mlflow.pyfunc
import numpy as np

from fastapi import FastAPI

from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse,
)

app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0.0"
)

MODEL_URI = "models:/credit_risk_model/1"

model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/")
def root():
    return {
        "message": "Credit Risk Model API Running"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    data = np.array(
        request.features
    ).reshape(1, -1)

    prediction = model.predict(data)

    probability = float(prediction[0])

    predicted_class = int(
        probability >= 0.5
    )

    return PredictionResponse(
        risk_probability=probability,
        prediction=predicted_class
    )
