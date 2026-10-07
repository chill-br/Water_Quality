from pathlib import Path
import pandas as pd
import numpy as np


BASE = Path(r"D:\Etawah_Water_Quality\Yamuna_Independent_Test")

FILE = (
    BASE
    / "data"
    / "ETAWAH_CWC_Sentinel2_CORRECT_TEMPORAL_MATCH_2021_2024.csv"
)

print("=" * 75)
print("STEP 11 — VERIFY CORRECT TEMPORAL MATCH")
print("=" * 75)

df = pd.read_csv(FILE)

print("\nRows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
for c in df.columns:
    print(" ", c)


# ------------------------------------------------------------
# DATETIME CONVERSION
# ------------------------------------------------------------

df["cwc_datetime"] = pd.to_datetime(
    df["cwc_datetime_iso"],
    errors="coerce"
)

df["sentinel2_datetime"] = pd.to_datetime(
    df["sentinel2_datetime"],
    errors="coerce"
)

df["sentinel2_date"] = pd.to_datetime(
    df["sentinel2_date"],
    errors="coerce"
)


# ------------------------------------------------------------
# ACTUAL TIME DIFFERENCE
# ------------------------------------------------------------

df["actual_difference_days"] = (
    (
        df["sentinel2_datetime"]
        - df["cwc_datetime"]
    )
    .abs()
    .dt.total_seconds()
    / 86400
)


# ------------------------------------------------------------
# CHECK STORED VS ACTUAL
# ------------------------------------------------------------

df["difference_error"] = (
    df["actual_difference_days"]
    - df["days_difference"]
).abs()


print("\n" + "=" * 75)
print("TEMPORAL CONSISTENCY")
print("=" * 75)

print(
    "\nStored days_difference:"
)

print(
    df["days_difference"]
    .describe()
    .to_string()
)

print(
    "\nActual CWC–Sentinel difference:"
)

print(
    df["actual_difference_days"]
    .describe()
    .to_string()
)

print(
    "\nMaximum difference between stored and actual:",
    df["difference_error"].max()
)

print(
    "\nRecords with mismatch > 0.01 days:",
    (df["difference_error"] > 0.01).sum()
)


# ------------------------------------------------------------
# THRESHOLDS
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("TEMPORAL THRESHOLDS")
print("=" * 75)

for threshold in [0.5, 1.0, 1.5, 2.0]:

    n = (
        df["actual_difference_days"]
        <= threshold
    ).sum()

    print(
        f"<= {threshold} day: {n}"
    )


# ------------------------------------------------------------
# DATE SAMPLE
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("FIRST 20 MATCHES")
print("=" * 75)

cols = [
    "cwc_datetime",
    "sentinel2_datetime",
    "sentinel2_id",
    "days_difference",
    "actual_difference_days",
    "cloud_percentage",
    "Turbidity (NTU)",
]

cols = [
    c for c in cols
    if c in df.columns
]

print(
    df[
        cols
    ]
    .sort_values("cwc_datetime")
    .head(20)
    .to_string(index=False)
)


# ------------------------------------------------------------
# YEAR COUNTS
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("MATCHED OBSERVATIONS BY YEAR")
print("=" * 75)

print(
    df["cwc_datetime"]
    .dt.year
    .value_counts()
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# DUPLICATES
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("DUPLICATE CHECK")
print("=" * 75)

key = [
    c for c in [
        "Station",
        "cwc_datetime_iso",
        "Turbidity (NTU)",
        "sentinel2_id",
    ]
    if c in df.columns
]

print("Duplicate key:", key)

if key:
    print(
        "Duplicate rows:",
        df.duplicated(
            subset=key,
            keep=False
        ).sum()
    )


# ------------------------------------------------------------
# SAVE VERIFIED FILE
# ------------------------------------------------------------

OUT = (
    BASE
    / "results"
    / "correct_temporal_match_verified.csv"
)

df.to_csv(
    OUT,
    index=False
)

print("\nSaved:")
print(OUT)

print("\n" + "=" * 75)
print("STEP 11 COMPLETE")
print("=" * 75)