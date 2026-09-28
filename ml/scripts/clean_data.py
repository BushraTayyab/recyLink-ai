import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

input_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "combined_transactions.csv"
)

output_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "modeling_transactions.csv"
)

print("Loading dataset...")

df = pd.read_csv(input_file)

print("Original rows:", len(df))


# ---------------------------------------------------------
# 1. REMOVE DUPLICATE TRANSACTIONS
# ---------------------------------------------------------

print("\nChecking duplicate transaction IDs...")

duplicates = df["transaction_id"].duplicated().sum()

print("Duplicate IDs:", duplicates)

df = df.drop_duplicates(
    subset="transaction_id",
    keep="first"
)

print("Rows after duplicate removal:", len(df))


# ---------------------------------------------------------
# 2. CONVERT NUMERIC COLUMNS
# ---------------------------------------------------------

numeric_columns = [
    "weight_kg",
    "latitude",
    "longitude",
    "quoted_price_per_kg",
    "final_price_per_kg",
    "estimated_value",
    "final_sale_value"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ---------------------------------------------------------
# 3. CONVERT DATE
# ---------------------------------------------------------

df["transaction_date"] = pd.to_datetime(
    df["transaction_date"],
    errors="coerce"
)


# ---------------------------------------------------------
# 4. CHECK MISSING VALUES
# ---------------------------------------------------------

print("\nMissing values before cleaning:")

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values.")
else:
    print(missing)


# ---------------------------------------------------------
# 5. REMOVE INVALID WEIGHTS
# ---------------------------------------------------------

invalid_weight = (
    df["weight_kg"].isna()
    | (df["weight_kg"] <= 0)
)

print(
    "\nInvalid weights:",
    invalid_weight.sum()
)

df = df[~invalid_weight]


# ---------------------------------------------------------
# 6. REMOVE INVALID PRICES
# ---------------------------------------------------------

invalid_price = (
    df["final_price_per_kg"].isna()
    | (df["final_price_per_kg"] <= 0)
)

print(
    "Invalid final prices:",
    invalid_price.sum()
)

df = df[~invalid_price]


# ---------------------------------------------------------
# 7. REMOVE INVALID FINAL SALE VALUES
# ---------------------------------------------------------

invalid_sale_value = (
    df["final_sale_value"].isna()
    | (df["final_sale_value"] <= 0)
)

print(
    "Invalid final sale values:",
    invalid_sale_value.sum()
)

df = df[~invalid_sale_value]


# ---------------------------------------------------------
# 8. CHECK ESTIMATED VALUE
# ---------------------------------------------------------

df["calculated_value"] = (
    df["weight_kg"]
    * df["quoted_price_per_kg"]
)

print("\nCalculated value column created.")


# ---------------------------------------------------------
# 9. CREATE PRICE DEVIATION FEATURE
# ---------------------------------------------------------

df["price_difference"] = (
    df["final_price_per_kg"]
    - df["quoted_price_per_kg"]
)

df["price_difference_percent"] = (
    df["price_difference"]
    / df["quoted_price_per_kg"].replace(0, pd.NA)
) * 100


# ---------------------------------------------------------
# 10. REMOVE ROWS WITHOUT REQUIRED ML DATA
# ---------------------------------------------------------

required_columns = [
    "material",
    "subcategory",
    "weight_kg",
    "condition",
    "final_price_per_kg",
    "final_sale_value"
]

before = len(df)

df = df.dropna(
    subset=required_columns
)

print(
    "\nRows removed because of missing ML data:",
    before - len(df)
)


# ---------------------------------------------------------
# 11. ANOMALY LABEL INFORMATION
# ---------------------------------------------------------

if "known_anomaly" in df.columns:

    print("\nAnomaly labels:")

    print(
        df["known_anomaly"]
        .value_counts(dropna=False)
    )

    print(
        "\nRows without anomaly labels:",
        df["known_anomaly"].isna().sum()
    )


# ---------------------------------------------------------
# 12. SAVE CLEAN DATASET
# ---------------------------------------------------------

df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)

print("Final rows:", len(df))
print("Final columns:", len(df.columns))

print("\nSaved to:")
print(output_file)