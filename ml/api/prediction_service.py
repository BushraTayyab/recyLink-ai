import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PRICE_MODEL_FILE = BASE_DIR / "models" / "price_model.pkl"
ANOMALY_MODEL_FILE = BASE_DIR / "models" / "anomaly_model.pkl"

HISTORICAL_DATA_FILE = (
    BASE_DIR
    / "data"
    / "modeling_transactions.csv"
)


# =========================================================
# LOAD MODELS
# =========================================================

price_model = joblib.load(PRICE_MODEL_FILE)
anomaly_model = joblib.load(ANOMALY_MODEL_FILE)

historical_data = pd.read_csv(HISTORICAL_DATA_FILE)


# =========================================================
# PRICE ESTIMATION
# =========================================================

def estimate_price(transaction):

    transaction = transaction.copy()

    transaction["transaction_date"] = pd.to_datetime(
        transaction["transaction_date"],
        errors="coerce"
    )

    transaction["transaction_year"] = (
        transaction["transaction_date"].dt.year
    )

    transaction["transaction_month"] = (
        transaction["transaction_date"].dt.month
    )

    transaction["transaction_day"] = (
        transaction["transaction_date"].dt.day
    )

    features = [
        "material",
        "subcategory",
        "weight_kg",
        "condition",
        "city",
        "state",
        "latitude",
        "longitude",
        "quoted_price_per_kg",
        "recycler_id",
        "collector_id",
        "pickup_type",
        "payment_method",
        "transaction_year",
        "transaction_month",
        "transaction_day"
    ]

    X = transaction[features]

    predicted_price = price_model.predict(X)[0]

    weight = transaction["weight_kg"].iloc[0]

    estimated_value = predicted_price * weight

    return predicted_price, estimated_value


# =========================================================
# ANOMALY FEATURE CALCULATION
# =========================================================

def calculate_anomaly_features(transaction):

    price_difference = (
        transaction["final_price_per_kg"]
        - transaction["quoted_price_per_kg"]
    )

    price_difference_percent = (
        price_difference
        / transaction["quoted_price_per_kg"]
    ) * 100

    calculated_final_value = (
        transaction["weight_kg"]
        * transaction["final_price_per_kg"]
    )

    value_difference = (
        calculated_final_value
        - transaction["final_sale_value"]
    )

    value_difference_percent = (
        value_difference
        / transaction["final_sale_value"]
    ) * 100

    material_data = historical_data[
        historical_data["material"]
        == transaction["material"]
    ]

    if len(material_data) > 0:

        material_median_price = (
            material_data["final_price_per_kg"]
            .median()
        )

        material_median_weight = (
            material_data["weight_kg"]
            .median()
        )

    else:

        material_median_price = (
            historical_data["final_price_per_kg"]
            .median()
        )

        material_median_weight = (
            historical_data["weight_kg"]
            .median()
        )

    material_price_deviation = (
        transaction["final_price_per_kg"]
        - material_median_price
    )

    material_price_deviation_percent = (
        material_price_deviation
        / material_median_price
    ) * 100

    weight_deviation_percent = (
        (
            transaction["weight_kg"]
            - material_median_weight
        )
        / material_median_weight
    ) * 100

    features = pd.DataFrame([{

        "weight_kg":
            transaction["weight_kg"],

        "quoted_price_per_kg":
            transaction["quoted_price_per_kg"],

        "final_price_per_kg":
            transaction["final_price_per_kg"],

        "final_sale_value":
            transaction["final_sale_value"],

        "price_difference_percent":
            price_difference_percent,

        "value_difference_percent":
            value_difference_percent,

        "material_price_deviation_percent":
            material_price_deviation_percent,

        "weight_deviation_percent":
            weight_deviation_percent
    }])

    return features


# =========================================================
# ANOMALY DETECTION
# =========================================================

def detect_anomaly(transaction):

    features = calculate_anomaly_features(transaction)

    prediction = anomaly_model.predict(features)[0]

    decision_score = anomaly_model.decision_function(
        features
    )[0]

    anomaly_score = max(
        0.0,
        min(
            1.0,
            0.5 - decision_score
        )
    )

    is_anomaly = prediction == -1

    if is_anomaly:
        message = "Unusual transaction — review"
    else:
        message = "Transaction appears normal"

    return is_anomaly, anomaly_score, message


# =========================================================
# MAIN PREDICTION FUNCTION
# =========================================================

def predict_transaction(transaction_data):

    transaction = pd.DataFrame([
        transaction_data
    ])

    # Price prediction
    predicted_price, estimated_value = estimate_price(
        transaction.copy()
    )

    # Anomaly detection
    is_anomaly, anomaly_score, message = detect_anomaly(
        transaction.iloc[0]
    )

    # Return result
    result = {

        "estimated_price_per_kg":
            round(float(predicted_price), 2),

        "estimated_value":
            round(float(estimated_value), 2),

        "is_anomaly":
            bool(is_anomaly),

        "anomaly_score":
            round(float(anomaly_score), 4),

        "message":
            message
    }

    return result