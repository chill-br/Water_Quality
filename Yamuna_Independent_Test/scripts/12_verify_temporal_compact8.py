from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# STEP 12 VERIFICATION
# Verify correct temporal Compact-8 Sentinel-2 features
# ============================================================

ROOT = Path(r"D:\Etawah_Water_Quality\Yamuna_Independent_Test")

INPUT_FILE = (
    ROOT
    / "data"
    / "ETAWAH_CORRECT_TEMPORAL_COMPACT8_2021_2024.csv"
)

OUTPUT_FILE = (
    ROOT
    / "results"
    / "correct_temporal_compact8_verified.csv"
)

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

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

print("=" * 70)
print("STEP 12 — VERIFY CORRECT TEMPORAL COMPACT-8 DATASET")
print("=" * 70)

print(f"\nInput file:\n{INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nCSV not found:\n{INPUT_FILE}\n\n"
        "Download the GEE export and place it in the data folder."
    )

df = pd.read_csv(INPUT_FILE)

print("\nRows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
for col in df.columns:
    print("  ", col)

# ------------------------------------------------------------
# 1. Required columns
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("1. REQUIRED COLUMN CHECK")
print("-" * 70)

required = [
    "station",
    "latitude",
    "longitude",
    "cwc_datetime",
    "turbidity_NTU",
    "satellite_id",
    "satellite_date",
    "satellite_datetime",
    "days_difference",
    "cloud_percentage",
    "valid_B8_pixels",
    "match_status",
] + FEATURES

missing = [c for c in required if c not in df.columns]

if missing:
    print("MISSING COLUMNS:")
    for c in missing:
        print("  ", c)
    raise ValueError("Required columns are missing.")

print("All required columns present: YES")

# ------------------------------------------------------------
# 2. Match status
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("2. MATCH STATUS")
print("-" * 70)

print(df["match_status"].value_counts(dropna=False))

valid_status = (df["match_status"] == "VALID_MATCH").sum()

print("\nVALID_MATCH rows:", valid_status)

# ------------------------------------------------------------
# 3. Missing values
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("3. MISSING VALUE CHECK")
print("-" * 70)

missing_counts = df[required].isna().sum()

print(missing_counts[missing_counts > 0])

if missing_counts.sum() == 0:
    print("No missing values in required columns: YES")
else:
    print("\nWARNING: Missing values detected.")

# ------------------------------------------------------------
# 4. Numeric feature statistics
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("4. COMPACT-8 FEATURE STATISTICS")
print("-" * 70)

print(
    df[FEATURES]
    .describe()
    .T[
        [
            "count",
            "mean",
            "std",
            "min",
            "50%",
            "max",
        ]
    ]
)

# ------------------------------------------------------------
# 5. NaN / Inf check
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("5. NaN / INF CHECK")
print("-" * 70)

numeric_features = df[FEATURES].apply(pd.to_numeric, errors="coerce")

nan_count = int(numeric_features.isna().sum().sum())
inf_count = int(
    np.isinf(numeric_features.to_numpy(dtype=float)).sum()
)

print("Feature NaN count:", nan_count)
print("Feature Inf count:", inf_count)

if nan_count == 0 and inf_count == 0:
    print("Feature numerical integrity: PASS")
else:
    print("Feature numerical integrity: FAIL")

# ------------------------------------------------------------
# 6. Reflectance scale check
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("6. REFLECTANCE SCALE CHECK")
print("-" * 70)

reflectance = ["B8_mean", "B11_mean", "B12_mean"]

for col in reflectance:
    values = pd.to_numeric(df[col], errors="coerce")

    print(
        f"{col}: "
        f"min={values.min():.6f}, "
        f"median={values.median():.6f}, "
        f"mean={values.mean():.6f}, "
        f"max={values.max():.6f}"
    )

    above_one = int((values > 1).sum())
    above_10000 = int((values > 10000).sum())

    print(f"  values > 1: {above_one}")
    print(f"  values > 10000: {above_10000}")

# Expected Sentinel-2 surface reflectance after /10000:
# generally around 0–1, although small values outside this range
# can occur depending on processing/masking.

# ------------------------------------------------------------
# 7. Water fraction check
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("7. WATER FRACTION CHECK")
print("-" * 70)

wf = pd.to_numeric(df["water_fraction"], errors="coerce")

print("Minimum:", wf.min())
print("Maximum:", wf.max())
print("Mean:", wf.mean())
print("Median:", wf.median())

outside_water = ((wf < 0) | (wf > 1)).sum()

print("Outside [0,1]:", int(outside_water))

if outside_water == 0:
    print("Water fraction range: PASS")
else:
    print("Water fraction range: FAIL")

# ------------------------------------------------------------
# 8. Valid B8 pixel check
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("8. VALID B8 PIXEL CHECK")
print("-" * 70)

b8_pixels = pd.to_numeric(
    df["valid_B8_pixels"],
    errors="coerce"
)

print("Minimum:", b8_pixels.min())
print("Maximum:", b8_pixels.max())
print("Mean:", b8_pixels.mean())

zero_pixels = int((b8_pixels <= 0).sum())

print("Rows with <=0 valid B8 pixels:", zero_pixels)

if zero_pixels == 0:
    print("Pixel availability: PASS")
else:
    print("WARNING: zero-pixel rows remain.")

# ------------------------------------------------------------
# 9. Temporal difference verification
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("9. TEMPORAL MATCH VERIFICATION")
print("-" * 70)

df["cwc_datetime_parsed"] = pd.to_datetime(
    df["cwc_datetime"],
    dayfirst=True,
    errors="coerce",
)

df["satellite_datetime_parsed"] = pd.to_datetime(
    df["satellite_datetime"],
    errors="coerce",
)

actual_difference = (
    df["satellite_datetime_parsed"]
    - df["cwc_datetime_parsed"]
).abs().dt.total_seconds() / 86400.0

stored_difference = pd.to_numeric(
    df["days_difference"],
    errors="coerce",
)

difference_error = (
    actual_difference - stored_difference
).abs()

print(
    "Stored days_difference — mean:",
    stored_difference.mean()
)

print(
    "Actual date difference — mean:",
    actual_difference.mean()
)

print(
    "Maximum discrepancy:",
    difference_error.max()
)

mismatch_count = int(
    (difference_error > 0.01).sum()
)

print(
    "Mismatch > 0.01 days:",
    mismatch_count
)

if mismatch_count == 0:
    print("Temporal consistency: PASS")
else:
    print("Temporal consistency: FAIL")

# ------------------------------------------------------------
# 10. Temporal windows
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("10. TEMPORAL MATCH WINDOWS")
print("-" * 70)

for threshold in [0.5, 1.0, 1.5, 2.0]:
    count = int((stored_difference <= threshold).sum())
    print(f"<= {threshold:.1f} day: {count}")

print(
    "Maximum stored difference:",
    stored_difference.max()
)

# ------------------------------------------------------------
# 11. Year distribution
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("11. CWC YEAR DISTRIBUTION")
print("-" * 70)

df["cwc_year"] = df["cwc_datetime_parsed"].dt.year

print(df["cwc_year"].value_counts().sort_index())

print("\n2021–2023 training candidates:")
print(
    int(
        df["cwc_year"]
        .isin([2021, 2022, 2023])
        .sum()
    )
)

print("\n2024 independent-test candidates:")
print(
    int(
        (df["cwc_year"] == 2024).sum()
    )
)

# ------------------------------------------------------------
# 12. Duplicate checks
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("12. DUPLICATE CHECKS")
print("-" * 70)

# Exact duplicate rows
exact_duplicates = int(df.duplicated().sum())

print("Exact duplicate rows:", exact_duplicates)

# Same CWC observation
cwc_key = [
    "station",
    "cwc_datetime",
    "turbidity_NTU",
]

cwc_duplicates = int(
    df.duplicated(
        subset=cwc_key,
        keep=False
    ).sum()
)

print(
    "Rows involved in duplicate "
    "station + CWC datetime + turbidity:",
    cwc_duplicates
)

# Same Sentinel image
sentinel_counts = (
    df["satellite_id"]
    .value_counts()
)

repeated_sentinel_ids = int(
    (sentinel_counts > 1).sum()
)

rows_in_repeated_sentinel_groups = int(
    sentinel_counts[sentinel_counts > 1].sum()
)

print(
    "Unique Sentinel-2 IDs:",
    df["satellite_id"].nunique()
)

print(
    "Repeated Sentinel-2 IDs:",
    repeated_sentinel_ids
)

print(
    "Rows belonging to repeated Sentinel groups:",
    rows_in_repeated_sentinel_groups
)

# ------------------------------------------------------------
# 13. Turbidity distribution
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("13. TURBIDITY DISTRIBUTION")
print("-" * 70)

turbidity = pd.to_numeric(
    df["turbidity_NTU"],
    errors="coerce"
)

print(
    turbidity.describe()[
        [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ]
    ]
)

print("\nTurbidity categories:")

categories = {
    "Low (<10)": turbidity < 10,
    "Moderate (10–20)": (turbidity >= 10) & (turbidity < 20),
    "Elevated (20–50)": (turbidity >= 20) & (turbidity < 50),
    "High (50–100)": (turbidity >= 50) & (turbidity < 100),
    "Very high (>=100)": turbidity >= 100,
}

for name, mask in categories.items():
    print(f"  {name}: {int(mask.sum())}")

# ------------------------------------------------------------
# 14. Extreme turbidity observations
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("14. EXTREME TURBIDITY OBSERVATIONS")
print("-" * 70)

extreme = (
    df.loc[
        turbidity >= 100,
        [
            "station",
            "cwc_datetime",
            "turbidity_NTU",
            "satellite_id",
            "satellite_date",
            "days_difference",
            "cloud_percentage",
        ],
    ]
    .sort_values("turbidity_NTU", ascending=False)
)

print(extreme.to_string(index=False))

# ------------------------------------------------------------
# 15. Final verification summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VERIFICATION SUMMARY")
print("=" * 70)

checks = {
    "Expected 104 rows": len(df) == 104,
    "All required columns present": len(missing) == 0,
    "No missing required values": missing_counts.sum() == 0,
    "No feature NaN": nan_count == 0,
    "No feature Inf": inf_count == 0,
    "Water fraction within [0,1]": outside_water == 0,
    "No zero-pixel rows": zero_pixels == 0,
    "Temporal difference consistent": mismatch_count == 0,
    "All matches <=2 days": stored_difference.max() <= 2.0,
}

for name, passed in checks.items():
    print(
        f"{'PASS' if passed else 'FAIL'}  {name}"
    )

overall = all(checks.values())

print("\nOVERALL:", "PASS" if overall else "REVIEW REQUIRED")

# ------------------------------------------------------------
# 16. Save verified copy
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

# Remove helper columns before saving
save_df = df.drop(
    columns=[
        "cwc_datetime_parsed",
        "satellite_datetime_parsed",
        "cwc_year",
    ],
    errors="ignore",
)

save_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nVerified file saved to:")
print(OUTPUT_FILE)

print("\nSTEP 12 VERIFICATION COMPLETE")