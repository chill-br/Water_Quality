from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 9 — TEMPORAL DATA INTEGRITY CHECK
# ============================================================

BASE = Path(r"D:\Etawah_Water_Quality")
TEST_BASE = BASE / "Yamuna_Independent_Test"

TRAIN_FILE = (
    TEST_BASE
    / "results"
    / "temporal_train_2021_2023.csv"
)

TEST_FILE = (
    TEST_BASE
    / "results"
    / "temporal_test_2024.csv"
)

print("=" * 75)
print("STEP 9 — TEMPORAL DATA INTEGRITY CHECK")
print("=" * 75)


train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

train["cwc_datetime"] = pd.to_datetime(
    train["cwc_datetime"],
    errors="coerce"
)

test["cwc_datetime"] = pd.to_datetime(
    test["cwc_datetime"],
    errors="coerce"
)

if "satellite_date" in train.columns:
    train["satellite_date"] = pd.to_datetime(
        train["satellite_date"],
        errors="coerce"
    )

if "satellite_date" in test.columns:
    test["satellite_date"] = pd.to_datetime(
        test["satellite_date"],
        errors="coerce"
    )


# ============================================================
# RAW DUPLICATES
# ============================================================

print("\n" + "=" * 75)
print("DUPLICATE ANALYSIS")
print("=" * 75)

print(
    f"Training rows: {len(train)}"
)

print(
    f"Testing rows : {len(test)}"
)

print(
    "\nExact duplicate training rows:",
    train.duplicated().sum()
)

print(
    "Exact duplicate testing rows:",
    test.duplicated().sum()
)


# ============================================================
# UNIQUE OBSERVATION DEFINITION
# ============================================================

key = [
    c for c in [
        "station",
        "cwc_datetime",
        "turbidity_NTU",
    ]
    if c in train.columns
]

print("\nObservation key:")
print(key)


print(
    "\nDuplicate observations using station + "
    "CWC datetime + turbidity:"
)

print(
    "Training:",
    train.duplicated(
        subset=key,
        keep=False
    ).sum()
)

print(
    "Testing:",
    test.duplicated(
        subset=key,
        keep=False
    ).sum()
)


# ============================================================
# TEMPORAL CONSISTENCY
# ============================================================

print("\n" + "=" * 75)
print("CWC–SENTINEL DATE CONSISTENCY")
print("=" * 75)

for name, df in [
    ("TRAIN", train),
    ("TEST", test),
]:

    if "satellite_date" not in df.columns:
        print(f"\n{name}: satellite_date missing")
        continue

    df["actual_date_difference_days"] = (
        (
            df["satellite_date"]
            - df["cwc_datetime"].dt.normalize()
        )
        .abs()
        .dt.total_seconds()
        / 86400
    )

    print(f"\n{name}")

    print(
        "Stored days_difference:"
    )

    print(
        df["days_difference"]
        .describe()
        .to_string()
    )

    print(
        "\nActual date difference:"
    )

    print(
        df["actual_date_difference_days"]
        .describe()
        .to_string()
    )

    inconsistent = df[
        np.abs(
            df["actual_date_difference_days"]
            - df["days_difference"]
        ) > 0.01
    ].copy()

    print(
        f"\nInconsistent records: {len(inconsistent)}"
    )

    if len(inconsistent) > 0:

        cols = [
            "station",
            "cwc_datetime",
            "satellite_date",
            "days_difference",
            "actual_date_difference_days",
            "turbidity_NTU",
        ]

        cols = [
            c for c in cols
            if c in inconsistent.columns
        ]

        print(
            inconsistent[cols]
            .sort_values("cwc_datetime")
            .head(30)
            .to_string(index=False)
        )


# ============================================================
# SENTINEL ID CONSISTENCY
# ============================================================

print("\n" + "=" * 75)
print("SENTINEL ID / DATE CONSISTENCY")
print("=" * 75)

for name, df in [
    ("TRAIN", train),
    ("TEST", test),
]:

    if "sentinel2_id" not in df.columns:
        print(f"\n{name}: sentinel2_id missing")
        continue

    print(f"\n{name}")

    print(
        "Unique Sentinel-2 IDs:",
        df["sentinel2_id"].nunique()
    )

    repeated = (
        df.groupby("sentinel2_id")
        .size()
        .reset_index(name="n")
    )

    repeated = repeated[
        repeated["n"] > 1
    ]

    print(
        "Repeated Sentinel-2 IDs:",
        len(repeated)
    )


# ============================================================
# PRINT TEST DATA
# ============================================================

print("\n" + "=" * 75)
print("2024 TEST DATA")
print("=" * 75)

cols = [
    "station",
    "cwc_datetime",
    "turbidity_NTU",
    "sentinel2_id",
    "satellite_date",
    "days_difference",
]

cols = [
    c for c in cols
    if c in test.columns
]

print(
    test[cols]
    .sort_values("cwc_datetime")
    .to_string(index=False)
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("STEP 9 COMPLETE")
print("=" * 75)

print(
    "\nDO NOT report the R² = 0.770 temporal result "
    "as final until the date-linkage integrity is resolved."
)

print(
    "\nNo model training performed."
)