import pandas as pd
from pathlib import Path


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

output_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "anomaly_features.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

print("Loading transaction dataset...")

df = pd.read_csv(input_file)

print("Rows:", len(df))


# =========================================================
# BASIC PRICE FEATURES
# =========================================================

# Difference between quoted and final price

df["price_difference"] = (
    df["final_price_per_kg"]
    - df["quoted_price_per_kg"]
)


# Percentage difference

df["price_difference_percent"] = (
    df["price_difference"]
    / df["quoted_price_per_kg"].replace(0, pd.NA)
) * 100


# =========================================================
# VALUE CONSISTENCY FEATURE
# =========================================================

# Expected value based on final price

df["calculated_final_value"] = (
    df["weight_kg"]
    * df["final_price_per_kg"]
)


# Difference between calculated value
# and recorded final sale value

df["value_difference"] = (
    df["calculated_final_value"]
    - df["final_sale_value"]
)


# Percentage value difference

df["value_difference_percent"] = (
    df["value_difference"]
    / df["final_sale_value"].replace(0, pd.NA)
) * 100


# =========================================================
# MATERIAL PRICE BASELINE
# =========================================================

# Calculate typical final price for each material

material_median_price = (
    df.groupby("material")["final_price_per_kg"]
    .transform("median")
)


df["material_median_price"] = (
    material_median_price
)


# Difference from normal material price

df["material_price_deviation"] = (
    df["final_price_per_kg"]
    - df["material_median_price"]
)


# Percentage deviation from material median

df["material_price_deviation_percent"] = (
    df["material_price_deviation"]
    / df["material_median_price"].replace(0, pd.NA)
) * 100


# =========================================================
# WEIGHT BASELINE
# =========================================================

# Typical transaction weight for each material

material_median_weight = (
    df.groupby("material")["weight_kg"]
    .transform("median")
)


df["material_median_weight"] = (
    material_median_weight
)


df["weight_deviation_percent"] = (
    (
        df["weight_kg"]
        - df["material_median_weight"]
    )
    / df["material_median_weight"].replace(0, pd.NA)
) * 100


# =========================================================
# DISPLAY FEATURE INFORMATION
# =========================================================

print("\n" + "=" * 60)
print("ANOMALY FEATURES CREATED")
print("=" * 60)

feature_columns = [
    "price_difference",
    "price_difference_percent",
    "calculated_final_value",
    "value_difference",
    "value_difference_percent",
    "material_median_price",
    "material_price_deviation",
    "material_price_deviation_percent",
    "material_median_weight",
    "weight_deviation_percent"
]

for column in feature_columns:
    print("-", column)


# =========================================================
# SAMPLE
# =========================================================

print("\nSample anomaly features:")

print(
    df[
        [
            "transaction_id",
            "material",
            "weight_kg",
            "quoted_price_per_kg",
            "final_price_per_kg",
            "final_sale_value",
            "price_difference_percent",
            "value_difference_percent",
            "material_price_deviation_percent",
            "weight_deviation_percent"
        ]
    ].head(10).to_string(index=False)
)


# =========================================================
# SAVE
# =========================================================

df.to_csv(
    output_file,
    index=False
)

print("\nFeatures saved to:")
print(output_file)

print("\n" + "=" * 60)
print("ANOMALY FEATURE CREATION COMPLETE")
print("=" * 60)