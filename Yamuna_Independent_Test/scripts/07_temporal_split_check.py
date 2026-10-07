from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 7 — FORWARD TEMPORAL SPLIT CHECK
# 2021–2023 TRAIN -> 2024 TEST
# ============================================================

BASE = Path(r"D:\Etawah_Water_Quality")
TEST_BASE = BASE / "Yamuna_Independent_Test"

DATA = BASE / "data" / "ml_ready_turbidity.csv"
RESULTS = TEST_BASE / "results"

RESULTS.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "MNDWI_median",
    "NDWI_median",
    "MNDWI_mean",
    "NDWI_mean",
    "water_fraction",
    "B11_mean",
    "B12_mean",
    "B8_mean",
]

TARGET = "turbidity_NTU"


print("=" * 75)
print("STEP 7 — FORWARD TEMPORAL SPLIT CHECK")
print("=" * 75)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA)

print(f"\nTotal rows: {len(df)}")

if "cwc_datetime" not in df.columns:
    raise ValueError(
        "Missing cwc_datetime column."
    )

df["cwc_datetime"] = pd.to_datetime(
    df["cwc_datetime"],
    errors="coerce"
)


# ------------------------------------------------------------
# REQUIRED COLUMNS
# ------------------------------------------------------------

required = FEATURES + [
    TARGET,
    "cwc_datetime",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n" +
        "\n".join(missing)
    )


# ------------------------------------------------------------
# YEAR
# ------------------------------------------------------------

df["year"] = df["cwc_datetime"].dt.year


print("\nObservations by year:")
print(
    df.groupby("year")
      .size()
      .reset_index(name="n")
      .to_string(index=False)
)


# ------------------------------------------------------------
# CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMissing values by feature:")

for col in FEATURES + [TARGET]:
    print(
        f"{col:20s} : {df[col].isna().sum()}"
    )


# ------------------------------------------------------------
# FORWARD SPLIT
# ------------------------------------------------------------

train = df[
    df["year"] <= 2023
].copy()

test = df[
    df["year"] == 2024
].copy()


print("\n" + "=" * 75)
print("FORWARD TEMPORAL SPLIT")
print("=" * 75)

print(
    f"Training period: 2021–2023"
)

print(
    f"Testing period : 2024"
)

print(
    f"Training rows  : {len(train)}"
)

print(
    f"Testing rows   : {len(test)}"
)


# ------------------------------------------------------------
# COMPLETE CASES
# ------------------------------------------------------------

train_complete = train.dropna(
    subset=FEATURES + [TARGET]
).copy()

test_complete = test.dropna(
    subset=FEATURES + [TARGET]
).copy()


print("\nComplete observations:")

print(
    f"Training complete: {len(train_complete)}"
)

print(
    f"Testing complete : {len(test_complete)}"
)


# ------------------------------------------------------------
# TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("TARGET DISTRIBUTION")
print("=" * 75)

for name, data in [
    ("TRAIN 2021–2023", train_complete),
    ("TEST 2024", test_complete),
]:

    print(f"\n{name}")

    if len(data) > 0:

        print(
            f"N      : {len(data)}"
        )

        print(
            f"Mean   : {data[TARGET].mean():.3f}"
        )

        print(
            f"Median : {data[TARGET].median():.3f}"
        )

        print(
            f"Min    : {data[TARGET].min():.3f}"
        )

        print(
            f"Max    : {data[TARGET].max():.3f}"
        )

        print(
            f"Std    : {data[TARGET].std():.3f}"
        )


# ------------------------------------------------------------
# TEST OBSERVATIONS
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("2024 TEST OBSERVATIONS")
print("=" * 75)

test_columns = [
    "station",
    "cwc_datetime",
    TARGET,
    "sentinel2_id",
    "satellite_date",
    "days_difference",
    "cloud_percentage",
]

test_columns = [
    c for c in test_columns
    if c in test_complete.columns
]

print(
    test_complete[
        test_columns
    ]
    .sort_values("cwc_datetime")
    .to_string(index=False)
)


# ------------------------------------------------------------
# SAVE SPLITS
# ------------------------------------------------------------

train_complete.to_csv(
    RESULTS / "temporal_train_2021_2023.csv",
    index=False
)

test_complete.to_csv(
    RESULTS / "temporal_test_2024.csv",
    index=False
)


# ------------------------------------------------------------
# FEATURE SCALE CHECK
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("FEATURE SCALE CHECK")
print("=" * 75)

for col in FEATURES:

    print(f"\n{col}")

    print(
        f"  Train median: "
        f"{train_complete[col].median():.6f}"
    )

    print(
        f"  Test median : "
        f"{test_complete[col].median():.6f}"
    )

    print(
        f"  Train mean  : "
        f"{train_complete[col].mean():.6f}"
    )

    print(
        f"  Test mean   : "
        f"{test_complete[col].mean():.6f}"
    )


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("STEP 7 COMPLETE")
print("=" * 75)

print(
    "\nSaved:"
)

print(
    RESULTS / "temporal_train_2021_2023.csv"
)

print(
    RESULTS / "temporal_test_2024.csv"
)

print(
    "\nNo model was trained."
)

print(
    "This step only checks whether the temporal split "
    "is suitable for independent evaluation."
)