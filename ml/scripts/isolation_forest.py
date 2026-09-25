import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

input_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "anomaly_features.csv"
)

models_dir = BASE_DIR / "models"
models_dir.mkdir(parents=True, exist_ok=True)

output_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "isolation_forest_results.csv"
)

model_file = (
    models_dir
    / "anomaly_model.pkl"
)


# =========================================================
# LOAD DATA
# =========================================================

print("Loading anomaly features...")

df = pd.read_csv(input_file)

print("Rows:", len(df))


# =========================================================
# SELECT FEATURES
# =========================================================
#
# IMPORTANT:
# known_anomaly is NOT used as a feature.
#
# It is only used later for evaluation.
# =========================================================

features = [
    "weight_kg",
    "quoted_price_per_kg",
    "final_price_per_kg",
    "final_sale_value",
    "price_difference_percent",
    "value_difference_percent",
    "material_price_deviation_percent",
    "weight_deviation_percent"
]


# =========================================================
# PREPARE DATA
# =========================================================

X = df[features].copy()

# Replace infinite values

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# Fill missing numeric values with median

X = X.fillna(
    X.median()
)


# =========================================================
# TRAIN ISOLATION FOREST
# =========================================================

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=200,
    contamination=0.07,
    random_state=42,
    n_jobs=-1
)

model.fit(X)


# =========================================================
# PREDICTIONS
# =========================================================

# Isolation Forest:
#
#  1 = normal
# -1 = anomaly

raw_prediction = model.predict(X)

df["isolation_prediction"] = raw_prediction


# Convert:
#
# -1 → 1 anomaly
#  1 → 0 normal

df["is_anomaly"] = (
    df["isolation_prediction"] == -1
).astype(int)


# =========================================================
# ANOMALY SCORE
# =========================================================

# decision_function:
# higher = more normal
#
# We reverse it so:
# higher = more unusual

decision_scores = model.decision_function(X)

df["anomaly_score"] = -decision_scores


# Normalize approximately to 0–1

minimum = df["anomaly_score"].min()
maximum = df["anomaly_score"].max()

if maximum != minimum:

    df["anomaly_score"] = (
        (df["anomaly_score"] - minimum)
        / (maximum - minimum)
    )

else:

    df["anomaly_score"] = 0.0


# =========================================================
# DISPLAY DISTRIBUTION
# =========================================================

print("\n" + "=" * 60)
print("ISOLATION FOREST RESULTS")
print("=" * 60)

print(
    df["is_anomaly"]
    .value_counts()
    .rename({
        0: "Normal",
        1: "Anomaly"
    })
)


print(
    "\nTotal anomalies detected:",
    df["is_anomaly"].sum()
)


# =========================================================
# EVALUATE USING GROUND TRUTH
# =========================================================

evaluation_df = df.dropna(
    subset=["known_anomaly"]
).copy()


if len(evaluation_df) > 0:

    y_true = (
        evaluation_df["known_anomaly"]
        .astype(int)
    )

    y_pred = (
        evaluation_df["is_anomaly"]
        .astype(int)
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\n" + "=" * 60)
    print("ISOLATION FOREST EVALUATION")
    print("=" * 60)

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)


# =========================================================
# SHOW MOST UNUSUAL TRANSACTIONS
# =========================================================

print("\n" + "=" * 60)
print("MOST UNUSUAL TRANSACTIONS")
print("=" * 60)

examples = (
    df.sort_values(
        "anomaly_score",
        ascending=False
    )
    [
        [
            "transaction_id",
            "material",
            "weight_kg",
            "quoted_price_per_kg",
            "final_price_per_kg",
            "final_sale_value",
            "is_anomaly",
            "anomaly_score"
        ]
    ]
    .head(10)
)

print(
    examples.to_string(index=False)
)


# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    model_file
)

print("\nIsolation Forest model saved to:")

print(model_file)


# =========================================================
# SAVE RESULTS
# =========================================================

df.to_csv(
    output_file,
    index=False
)

print("\nResults saved to:")

print(output_file)


print("\n" + "=" * 60)
print("ISOLATION FOREST COMPLETE")
print("=" * 60)