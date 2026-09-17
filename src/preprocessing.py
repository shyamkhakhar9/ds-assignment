"""Leakage-safe cleaning and feature engineering for Telco churn data."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

TARGET_COL = "Churn"
ID_COL = "customerID"

SERVICE_COLS: List[str] = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERIC_FEATURES: List[str] = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "num_services",
    "avg_monthly_spend",
    "fiber_month_to_month",
]

RAW_INPUT_COLUMNS: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Clean types and add engineered features. Stateless — safe to fit on train only."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
        frame["TotalCharges"] = frame["TotalCharges"].fillna(0.0)
        frame["SeniorCitizen"] = pd.to_numeric(frame["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
        frame["tenure"] = pd.to_numeric(frame["tenure"], errors="coerce").fillna(0)
        frame["MonthlyCharges"] = pd.to_numeric(frame["MonthlyCharges"], errors="coerce").fillna(0.0)

        service_flags = []
        for col in SERVICE_COLS:
            if col in frame.columns:
                service_flags.append((frame[col].astype(str) == "Yes").astype(int))
            else:
                service_flags.append(pd.Series(0, index=frame.index))
        frame["num_services"] = sum(service_flags)

        tenure = frame["tenure"].astype(float)
        frame["avg_monthly_spend"] = np.where(
            tenure > 0,
            frame["TotalCharges"] / tenure,
            frame["MonthlyCharges"],
        )
        frame["fiber_month_to_month"] = (
            (frame["InternetService"].astype(str) == "Fiber optic")
            & (frame["Contract"].astype(str) == "Month-to-month")
        ).astype(int)
        return frame


def encode_target(y: pd.Series) -> np.ndarray:
    return (y.astype(str).str.strip() == "Yes").astype(int).to_numpy()


def decode_prediction(label: int) -> str:
    return "Yes" if int(label) == 1 else "No"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_model_pipeline(clf: DecisionTreeClassifier) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", TelcoFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", clf),
        ]
    )


def load_raw_frame(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    frame = df.copy()
    if ID_COL in frame.columns:
        frame = frame.drop(columns=[ID_COL])
    y = frame[TARGET_COL]
    X = frame.drop(columns=[TARGET_COL])
    return X, y
