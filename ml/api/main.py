from fastapi import FastAPI
from ml.api.prediction_service import predict_transaction

app = FastAPI(
    title="RecyLink ML API",
    description="Price estimation and transaction anomaly detection",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "RecyLink ML API is running",
        "service": "Price Estimation + Anomaly Detection"
    }


@app.post("/predict")
def predict(transaction: dict):
    return predict_transaction(transaction)