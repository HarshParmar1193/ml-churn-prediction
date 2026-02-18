from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(title="Customer Churn Prediction API", version="1.0")

# Load trained model
MODEL_PATH = "models/model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# ==========================
# 1. Define input schema
# ==========================
class ChurnRequest(BaseModel):
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# ==========================
# 2. Endpoints
# ==========================
@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: ChurnRequest):

    if model is None:
        return {"error": "Model file not found. Please train or add model.pkl"}

    df = pd.DataFrame([request.dict()])
    prediction = model.predict(df)[0]

    return {
        "churn_prediction": int(prediction),
        "churn_meaning": "Will churn" if prediction == 1 else "Will NOT churn"
    }
