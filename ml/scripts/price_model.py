import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

input_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "modeling_transactions.csv"
)

models_dir = BASE_DIR / "models"
models_dir.mkdir(parents=True, exist_ok=True)

results_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "price_model_results.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

print("Loading dataset...")

df = pd.read_csv(input_file)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# =========================================================
# DATE FEATURES
# =========================================================

df["transaction_date"] = pd.to_datetime(
    df["transaction_date"],
    errors="coerce"
)

df["transaction_year"] = (
    df["transaction_date"].dt.year
)

df["transaction_month"] = (
    df["transaction_date"].dt.month
)

df["transaction_day"] = (
    df["transaction_date"].dt.day
)


# =========================================================
# TARGET
# =========================================================
#
# We predict the final price received per kg.
#
# final_price_per_kg
#
# Then:
#
# estimated_value =
# predicted_price_per_kg × weight
# =========================================================

target = "final_price_per_kg"


# =========================================================
# FEATURES
# =========================================================

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

X = df[features]
y = df[target]


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# =========================================================
# FEATURE TYPES
# =========================================================

categorical_features = [
    "material",
    "subcategory",
    "condition",
    "city",
    "state",
    "recycler_id",
    "collector_id",
    "pickup_type",
    "payment_method"
]

numeric_features = [
    "weight_kg",
    "latitude",
    "longitude",
    "quoted_price_per_kg",
    "transaction_year",
    "transaction_month",
    "transaction_day"
]


# =========================================================
# PREPROCESSING
# =========================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# =========================================================
# LINEAR REGRESSION
# =========================================================

linear_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)

print("\nTraining Linear Regression...")

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = linear_model.predict(
    X_test
)


# =========================================================
# RANDOM FOREST
# =========================================================

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("Training Random Forest...")

random_forest_model.fit(
    X_train,
    y_train
)

rf_predictions = random_forest_model.predict(
    X_test
)


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return mae, rmse, r2


# =========================================================
# EVALUATE BOTH MODELS
# =========================================================

linear_metrics = evaluate_model(
    "LINEAR REGRESSION",
    y_test,
    linear_predictions
)

rf_metrics = evaluate_model(
    "RANDOM FOREST",
    y_test,
    rf_predictions
)


# =========================================================
# SELECT MODEL
# =========================================================
#
# Lower MAE is better.
# =========================================================

if rf_metrics[0] < linear_metrics[0]:

    best_model = random_forest_model
    best_predictions = rf_predictions
    best_name = "Random Forest"
    best_metrics = rf_metrics

else:

    best_model = linear_model
    best_predictions = linear_predictions
    best_name = "Linear Regression"
    best_metrics = linear_metrics


print("\n" + "=" * 60)
print("SELECTED MODEL")
print("=" * 60)

print("Model:", best_name)
print(f"MAE : {best_metrics[0]:.4f}")
print(f"RMSE: {best_metrics[1]:.4f}")
print(f"R²  : {best_metrics[2]:.4f}")


# =========================================================
# SAVE MODEL
# =========================================================

model_file = (
    models_dir
    / "price_model.pkl"
)

joblib.dump(
    best_model,
    model_file
)

print("\nModel saved to:")
print(model_file)


# =========================================================
# SAVE EVALUATION RESULTS
# =========================================================

results = pd.DataFrame({
    "model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        linear_metrics[0],
        rf_metrics[0]
    ],
    "RMSE": [
        linear_metrics[1],
        rf_metrics[1]
    ],
    "R2": [
        linear_metrics[2],
        rf_metrics[2]
    ]
})

results.to_csv(
    results_file,
    index=False
)

print("\nEvaluation results saved to:")
print(results_file)


# =========================================================
# SAMPLE PRICE PREDICTIONS
# =========================================================

sample = X_test.head(10).copy()

sample["actual_price_per_kg"] = (
    y_test.loc[sample.index]
)

sample["predicted_price_per_kg"] = (
    best_model.predict(sample)
)

sample["weight_kg"] = (
    sample["weight_kg"]
)

sample["estimated_value"] = (
    sample["predicted_price_per_kg"]
    * sample["weight_kg"]
)

print("\n" + "=" * 60)
print("SAMPLE PRICE PREDICTIONS")
print("=" * 60)

print(
    sample[
        [
            "material",
            "weight_kg",
            "actual_price_per_kg",
            "predicted_price_per_kg",
            "estimated_value"
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 60)
print("PRICE MODEL COMPLETE")
print("=" * 60)