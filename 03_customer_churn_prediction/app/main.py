from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")


class CustomerFeatures(BaseModel):
    tenure_months: int = Field(ge=0, le=600)
    monthly_charges: float = Field(ge=0, le=100000)
    support_tickets: int = Field(ge=0, le=1000)
    contract_type: str


@app.get("/")
def health():
    return {"status": "ok", "message": "Train the model with python train.py before using /predict."}


@app.post("/predict")
def predict(customer: CustomerFeatures):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not found. Run python train.py first.")
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([customer.model_dump()])
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    return {
        "churn_prediction": "Likely to churn" if prediction == 1 else "Unlikely to churn",
        "churn_probability": round(probability, 3),
        "note": "Educational demo trained on a tiny synthetic dataset."
    }
