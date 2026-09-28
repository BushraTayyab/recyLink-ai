import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

input_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "anomaly_features.csv"
)

output_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "statistical_anomaly_results.csv"
)

print("Loading anomaly features...")

df = pd.read_csv(input_file)

print("Rows:", len(df))


# =========================================================
# STATISTICAL THRESHOLDS
# =========================================================
#
# A transaction is unusual if:
#
# 1. Price differs greatly from normal material price
# 2. Weight differs greatly from normal material weight
# 3. Recorded value differs greatly from calculated value
#
# =========================================================

PRICE_THRESHOLD = 50
WEIGHT_THRESHOLD = 100
VALUE_THRESHOLD = 20


# =========================================================
# CREATE ANOMALY CONDITIONS
# =========================================================

price_anomaly = (
    df["material_price_deviation_percent"].abs()
    > PRICE_THRESHOLD
)

weight_anomaly = (
    df["weight_deviation_percent"].abs()
    > WEIGHT_THRESHOLD
)

value_anomaly = (
    df["value_difference_percent"].abs()
    > VALUE_THRESHOLD
)


# =========================================================
# COMBINE CONDITIONS
# =========================================================

df["statistical_anomaly"] = (
    price_anomaly
    | weight_anomaly
    | value_anomaly
)


# Convert to integer

df["statistical_anomaly"] = (
    df["statistical_anomaly"]
    .astype(int)
)


# =========================================================
# ANOMALY SCORE
# =========================================================

df["anomaly_score"] = (
    (
        df["material_price_deviation_percent"].abs()
        / PRICE_THRESHOLD
    )
    +
    (
        df["weight_deviation_percent"].abs()
        / WEIGHT_THRESHOLD
    )
    +
    (
        df["value_difference_percent"].abs()
        / VALUE_THRESHOLD
    )
) / 3


# =========================================================
# DISPLAY DISTRIBUTION
# =========================================================

print("\n" + "=" * 60)
print("STATISTICAL ANOMALY DISTRIBUTION")
print("=" * 60)

print(
    df["statistical_anomaly"]
    .value_counts()
)

print("\nAnomalies detected:")

print(
    df["statistical_anomaly"].sum()
)


# =========================================================
# EVALUATE AGAINST GROUND TRUTH
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
        evaluation_df["statistical_anomaly"]
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
    print("STATISTICAL BASELINE EVALUATION")
    print("=" * 60)

    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)


# =========================================================
# SHOW EXAMPLES
# =========================================================

print("\n" + "=" * 60)
print("EXAMPLE ANOMALOUS TRANSACTIONS")
print("=" * 60)

examples = df[
    df["statistical_anomaly"] == 1
][
    [
        "transaction_id",
        "material",
        "weight_kg",
        "final_price_per_kg",
        "final_sale_value",
        "material_price_deviation_percent",
        "weight_deviation_percent",
        "value_difference_percent",
        "anomaly_score"
    ]
].head(10)

print(
    examples.to_string(index=False)
)


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
print("STATISTICAL ANOMALY DETECTION COMPLETE")
print("=" * 60)