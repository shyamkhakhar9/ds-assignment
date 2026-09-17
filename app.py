"""FastAPI service for Telco churn prediction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.preprocessing import RAW_INPUT_COLUMNS, decode_prediction

MODEL_PATH = Path(__file__).resolve().parent / "model" / "churn_pipeline.pkl"

app = FastAPI(
    title="Telco Customer Churn API",
    description="Predict whether a telecom customer is likely to churn.",
    version="1.0.0",
)


class CustomerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Union[float, str] = Field(description="Numeric total, or blank for new customers")


class PredictResponse(BaseModel):
    prediction: Literal["Yes", "No"]
    churn_probability: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@lru_cache(maxsize=1)
def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python -m src.train` first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        load_pipeline()
        return HealthResponse(status="ok", model_loaded=True)
    except FileNotFoundError:
        return HealthResponse(status="model_missing", model_loaded=False)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: CustomerPayload):
    try:
        pipeline = load_pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    record = payload.model_dump()
    frame = pd.DataFrame([record], columns=RAW_INPUT_COLUMNS)
    try:
        proba = pipeline.predict_proba(frame)[0]
        classes = list(pipeline.classes_)
        yes_index = classes.index(1) if 1 in classes else 1
        churn_probability = float(proba[yes_index])
        label = int(pipeline.predict(frame)[0])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not score this record: {exc}") from exc

    return PredictResponse(
        prediction=decode_prediction(label),
        churn_probability=round(churn_probability, 4),
    )
