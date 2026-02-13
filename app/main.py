from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint, confloat
import joblib
import pandas as pd

# =========================
# Load trained model
# =========================
model = joblib.load("models/model.pkl")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a customer will churn or not",
    version="1.0"
)

# =========================
# Define Pydantic input model
# =========================
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: conint(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: conint(ge=0)
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
    MonthlyCharges: confloat(ge=0)
    TotalCharges: confloat(ge=0)


# =========================
# API Endpoints
# =========================
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        # Convert Pydantic model to DataFrame
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)
        proba = model.predict_proba(df)[:, 1]

        result = {
            "churn_prediction": int(prediction[0]),
            "churn_probability": float(proba[0])
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
