from pathlib import Path
import pandas as pd


# ============================================================
# STEP 3B — INDEPENDENT SPATIAL TEST
# Dataset verification
# ============================================================

# Project folders
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Input file
# ------------------------------------------------------------

INPUT_FILE = DATA_DIR / "Yamuna_Independent_Spatial_Test_Features.csv"

print("=" * 70)
print("STEP 3B — INDEPENDENT SPATIAL TEST")
print("=" * 70)

print("\nInput file:")
print(INPUT_FILE)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nCSV file not found:\n{INPUT_FILE}\n\n"
        "Please place Yamuna_Independent_Spatial_Test_Features.csv "
        "inside the data folder."
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")


# ------------------------------------------------------------
# Basic information
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("DATASET INFORMATION")
print("-" * 70)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


print("\nColumns:")
for column in df.columns:
    print(f"  - {column}")


# ------------------------------------------------------------
# Required Compact-8 features
# ------------------------------------------------------------

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

REQUIRED_COLUMNS = FEATURES + [
    TARGET,
    "station",
    "satellite_id",
    "satellite_date",
    "cwc_datetime",
    "days_difference",
    "cloud_percentage",
]


# ------------------------------------------------------------
# Check required columns
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("REQUIRED COLUMN CHECK")
print("-" * 70)

missing_columns = [
    column for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    print("\nERROR — Missing columns:")
    for column in missing_columns:
        print(f"  ✗ {column}")

    raise ValueError(
        "\nThe exported GEE dataset is missing required columns."
    )

else:
    print("\nAll required columns are present.")
    print("✓ Compact-8 features")
    print("✓ turbidity target")
    print("✓ station information")
    print("✓ Sentinel-2 information")


# ------------------------------------------------------------
# Station check
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("STATION CHECK")
print("-" * 70)

station_counts = (
    df["station"]
    .value_counts()
    .sort_index()
)

print(station_counts.to_string())

print(f"\nNumber of stations: {df['station'].nunique()}")


# ------------------------------------------------------------
# Expected station count
# ------------------------------------------------------------

EXPECTED_STATIONS = {
    "AURAIYA",
    "KALPI",
    "HAMIRPUR",
    "RAJAPUR",
    "Mathura (Gokul Barrage)",
    "AGRA (J.B.)",
}

actual_stations = set(df["station"].dropna().unique())

missing_stations = EXPECTED_STATIONS - actual_stations
unexpected_stations = actual_stations - EXPECTED_STATIONS

if missing_stations:
    print("\nWARNING — Missing expected stations:")
    for station in sorted(missing_stations):
        print(f"  ✗ {station}")
else:
    print("\n✓ All six expected stations are present.")

if unexpected_stations:
    print("\nWARNING — Unexpected stations:")
    for station in sorted(unexpected_stations):
        print(f"  ? {station}")


# ------------------------------------------------------------
# Row count check
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("ROW COUNT CHECK")
print("-" * 70)

EXPECTED_ROWS = 666

print(f"Expected rows : {EXPECTED_ROWS}")
print(f"Actual rows   : {len(df)}")

if len(df) == EXPECTED_ROWS:
    print("✓ Row count matches the GEE result.")
else:
    print("⚠ Row count does NOT match the expected 666 records.")


# ------------------------------------------------------------
# Missing-value check
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("MISSING VALUE CHECK")
print("-" * 70)

missing = df[REQUIRED_COLUMNS].isna().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("✓ No missing values in required columns.")
else:
    print("Missing values:")
    print(missing.to_string())


# ------------------------------------------------------------
# Duplicate check
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("DUPLICATE CHECK")
print("-" * 70)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count}")

if duplicate_count == 0:
    print("✓ No completely duplicated rows.")
else:
    print("⚠ Duplicate rows detected.")


# ------------------------------------------------------------
# Target summary
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("TURBIDITY SUMMARY")
print("-" * 70)

print(df[TARGET].describe().to_string())


# ------------------------------------------------------------
# Feature summary
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("COMPACT-8 FEATURE SUMMARY")
print("-" * 70)

print(
    df[FEATURES]
    .describe()
    .T
    .to_string()
)


# ------------------------------------------------------------
# Sentinel-2 grouping
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("SENTINEL-2 GROUPING")
print("-" * 70)

unique_s2 = df["satellite_id"].nunique()

print(f"Observations              : {len(df)}")
print(f"Unique Sentinel-2 images  : {unique_s2}")


# ------------------------------------------------------------
# Temporal matching summary
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("TEMPORAL MATCHING SUMMARY")
print("-" * 70)

print(
    df["days_difference"]
    .describe()
    .to_string()
)

print(
    f"\nMean absolute date difference: "
    f"{df['days_difference'].abs().mean():.4f} days"
)


# ------------------------------------------------------------
# Cloud summary
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("CLOUD SUMMARY")
print("-" * 70)

print(
    df["cloud_percentage"]
    .describe()
    .to_string()
)


# ------------------------------------------------------------
# Save verified dataset
# ------------------------------------------------------------

OUTPUT_FILE = RESULTS_DIR / "independent_test_verified.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("STEP 3B DATASET VERIFICATION COMPLETE")
print("=" * 70)

print(f"\nVerified dataset saved to:")
print(OUTPUT_FILE)

print("\nNext step:")
print("Run this script and send me the complete console output.")
print("=" * 70)