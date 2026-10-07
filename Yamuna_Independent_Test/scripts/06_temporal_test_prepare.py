from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 6 — TEMPORAL TEST PREPARATION
# Determine the available Etawah temporal holdout
# ============================================================

BASE = Path(r"D:\Etawah_Water_Quality")
TEST_BASE = BASE / "Yamuna_Independent_Test"

DEVELOPMENT_FILE = BASE / "data" / "ml_ready_turbidity.csv"
SPECTRAL_FILE = BASE / "data" / "Etawah_CWC_Sentinel2_Spectral_Indices_2021_2025 (1).csv"

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
print("STEP 6 — TEMPORAL TEST PREPARATION")
print("=" * 75)


# ============================================================
# LOAD DEVELOPMENT DATA
# ============================================================

print("\nLoading development dataset:")
print(DEVELOPMENT_FILE)

dev = pd.read_csv(DEVELOPMENT_FILE)

print(f"Development rows: {len(dev)}")

if "cwc_datetime" not in dev.columns:
    raise ValueError(
        "Development dataset does not contain 'cwc_datetime'."
    )

dev["cwc_datetime"] = pd.to_datetime(
    dev["cwc_datetime"],
    errors="coerce"
)

print("\nDevelopment date range:")

print(
    "Minimum:",
    dev["cwc_datetime"].min()
)

print(
    "Maximum:",
    dev["cwc_datetime"].max()
)


# ============================================================
# LOAD ORIGINAL ETAWAH SPECTRAL DATA
# ============================================================

print("\nLoading original Etawah CWC–Sentinel spectral dataset:")
print(SPECTRAL_FILE)

spectral = pd.read_csv(SPECTRAL_FILE)

print(f"Spectral rows: {len(spectral)}")

if "cwc_datetime" not in spectral.columns:
    raise ValueError(
        "Spectral dataset does not contain 'cwc_datetime'."
    )

spectral["cwc_datetime"] = pd.to_datetime(
    spectral["cwc_datetime"],
    errors="coerce"
)


# ============================================================
# FILTER ETAWAH
# ============================================================

if "station" not in spectral.columns:
    raise ValueError(
        "Spectral dataset does not contain 'station'."
    )

etawah = spectral[
    spectral["station"].astype(str).str.upper() == "ETAWAH"
].copy()

print(f"\nETAWAH spectral rows: {len(etawah)}")


# ============================================================
# DATE RANGE
# ============================================================

print("\nETAWAH spectral date range:")

print(
    "Minimum:",
    etawah["cwc_datetime"].min()
)

print(
    "Maximum:",
    etawah["cwc_datetime"].max()
)


# ============================================================
# DEVELOPMENT DATE RANGE
# ============================================================

dev_min = dev["cwc_datetime"].min()
dev_max = dev["cwc_datetime"].max()

print("\nDevelopment period:")
print(f"{dev_min} → {dev_max}")


# ============================================================
# FIND OBSERVATIONS AFTER DEVELOPMENT PERIOD
# ============================================================

later = etawah[
    etawah["cwc_datetime"] > dev_max
].copy()

print("\n" + "=" * 75)
print("TEMPORAL HOLDOUT CANDIDATES")
print("=" * 75)

print(
    f"Observations after development maximum date: {len(later)}"
)


if len(later) > 0:

    print("\nCandidate date range:")
    print(
        "Minimum:",
        later["cwc_datetime"].min()
    )
    print(
        "Maximum:",
        later["cwc_datetime"].max()
    )

    print("\nCandidate observations:")

    columns = [
        "station",
        "cwc_datetime",
        TARGET,
        "satellite_id",
        "satellite_date",
        "days_difference",
    ]

    columns = [
        c for c in columns
        if c in later.columns
    ]

    print(
        later[columns]
        .sort_values("cwc_datetime")
        .to_string(index=False)
    )

    later.to_csv(
        RESULTS / "temporal_holdout_candidates.csv",
        index=False
    )

    print(
        "\nSaved:"
    )
    print(
        RESULTS / "temporal_holdout_candidates.csv"
    )

else:

    print(
        "\nNo observations occur after the development "
        "dataset's maximum CWC date."
    )


# ============================================================
# CHECK FOR YEAR DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("ETAWAH OBSERVATIONS BY YEAR")
print("=" * 75)

year_counts = (
    etawah
    .dropna(subset=["cwc_datetime"])
    .assign(
        year=lambda x: x["cwc_datetime"].dt.year
    )
    .groupby("year")
    .size()
    .reset_index(name="n")
)

print(year_counts.to_string(index=False))


# ============================================================
# CHECK DEVELOPMENT DATA BY YEAR
# ============================================================

print("\n" + "=" * 75)
print("DEVELOPMENT DATA BY YEAR")
print("=" * 75)

dev_year_counts = (
    dev
    .dropna(subset=["cwc_datetime"])
    .assign(
        year=lambda x: x["cwc_datetime"].dt.year
    )
    .groupby("year")
    .size()
    .reset_index(name="n")
)

print(dev_year_counts.to_string(index=False))


# ============================================================
# IMPORTANT CHECK
# ============================================================

print("\n" + "=" * 75)
print("STEP 6 PREPARATION COMPLETE")
print("=" * 75)

print(
    "\nIMPORTANT:"
)

print(
    "Do NOT retrain the Etawah model."
)

print(
    "Do NOT use temporal holdout observations for model fitting."
)

print(
    "The next step depends on the available post-development "
    "Etawah observations."
)

print(
    "\nCandidate file:"
)

print(
    RESULTS / "temporal_holdout_candidates.csv"
)