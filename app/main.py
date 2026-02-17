from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API", version="1.0")

# Load trained model
model = joblib.load("models/model.pkl")

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


@app.post("/predict")
def predict(request: ChurnRequest):
    # Convert Pydantic model to DataFrame
    df = pd.DataFrame([request.dict()])
    
    # Make prediction
    prediction = model.predict(df)[0]
    
    # Return structured response
    return {
        "churn_prediction": int(prediction),
        "churn_meaning": "Will churn" if prediction == 1 else "Will NOT churn"
    }
