import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

input_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "modeling_transactions.csv"
)

output_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "price_baseline_results.csv"
)

print("Loading cleaned dataset...")

df = pd.read_csv(input_file)

print("Rows loaded:", len(df))


# ---------------------------------------------------------
# PRICE BASELINE
# ---------------------------------------------------------
# Simple baseline:
#
# estimated value =
# weight × quoted price per kg
# ---------------------------------------------------------

df["baseline_price_per_kg"] = df["quoted_price_per_kg"]

df["baseline_estimated_value"] = (
    df["weight_kg"]
    * df["baseline_price_per_kg"]
)


# ---------------------------------------------------------
# CALCULATE ERROR
# ---------------------------------------------------------

df["baseline_error"] = (
    df["baseline_estimated_value"]
    - df["final_sale_value"]
)

df["absolute_error"] = (
    df["baseline_error"].abs()
)


# ---------------------------------------------------------
# EVALUATION METRICS
# ---------------------------------------------------------

actual = df["final_sale_value"]

predicted = df["baseline_estimated_value"]

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


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PRICE BASELINE RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ---------------------------------------------------------
# SAMPLE PREDICTIONS
# ---------------------------------------------------------

print("\nSample predictions:")

sample = df[
    [
        "transaction_id",
        "material",
        "weight_kg",
        "quoted_price_per_kg",
        "final_sale_value",
        "baseline_estimated_value"
    ]
].head(10)

print(
    sample.to_string(index=False)
)


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

df.to_csv(
    output_file,
    index=False
)

print("\nResults saved to:")

print(output_file)

print("\n" + "=" * 60)
print("BASELINE COMPLETE")
print("=" * 60)