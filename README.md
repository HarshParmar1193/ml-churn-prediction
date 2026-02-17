🚀 Customer Churn Prediction - End-to-End ML System

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/) 
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)](https://fastapi.tiangolo.com/) 
[![Docker](https://img.shields.io/badge/docker-latest-blue)](https://www.docker.com/) 
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange)](https://scikit-learn.org/)

📖 Overview
This project implements a **complete end-to-end ML pipeline** for predicting customer churn. It includes:

- Data preprocessing & feature engineering  
- Model training with Random Forest  
- Model evaluation (Accuracy, ROC-AUC)  
- FastAPI REST API for real-time predictions  
- Dockerized deployment for reproducibility  

🛠 Tech Stack
- Python 3.10 | Pandas | NumPy | Scikit-learn | FastAPI | Pydantic | Docker

✨ Features
- Automated **data preprocessing** with `ColumnTransformer`  
- **Random Forest classifier** for churn prediction  
- Real-time **API for predictions**  
- Dockerized deployment ensures **consistency across environments**  
- Reproducible environment with pinned dependencies

📁 Project Structure
churn-mlops-project/
- src/          # Training & preprocessing pipeline
- app/          # FastAPI application
- Dockerfile    # Container configuration
- requirements.txt  # Pinned dependencies
- README.md     # This file

⚡ How to Run

1️⃣ Clone Repository
- git clone https://github.com/HarshParmar1193/ml-churn-prediction.git
- cd ml-churn-prediction

2️⃣ Build Docker Image
- docker build -t churn-api .

3️⃣ Run Container
- docker run -p 8000:8000 churn-api

4️⃣ Access API
- Open in your browser:http://localhost:8000/docs

🧪 Example API Response
-
{
  "churn_prediction": 0,
  "churn_meaning": "Will NOT churn"
}
