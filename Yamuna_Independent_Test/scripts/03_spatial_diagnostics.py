from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STEP 3C — SPATIAL TRANSFER DIAGNOSTICS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# INPUT FILES
# ------------------------------------------------------------

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "independent_spatial_predictions.csv"
)

ETAWAH_FILE = (
    Path(r"D:\Etawah_Water_Quality\data")
    / "ml_ready_turbidity.csv"
)


# ------------------------------------------------------------
# OUTPUT FILES
# ------------------------------------------------------------

CLOUD_RESULTS_FILE = (
    RESULTS_DIR
    / "diagnostic_cloud_performance.csv"
)

TIME_RESULTS_FILE = (
    RESULTS_DIR
    / "diagnostic_temporal_performance.csv"
)

STATION_RESULTS_FILE = (
    RESULTS_DIR
    / "diagnostic_station_summary.csv"
)

CATEGORY_RESULTS_FILE = (
    RESULTS_DIR
    / "diagnostic_turbidity_category.csv"
)

FEATURE_SHIFT_FILE = (
    RESULTS_DIR
    / "diagnostic_feature_distribution_shift.csv"
)

EXTREME_RESULTS_FILE = (
    RESULTS_DIR
    / "diagnostic_extreme_events.csv"
)

CORRELATION_FILE = (
    RESULTS_DIR
    / "diagnostic_error_correlations.csv"
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

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

PREDICTED = "predicted_turbidity_NTU"

RESIDUAL = "residual_NTU"

ABS_ERROR = "absolute_error_NTU"


# ============================================================
# START
# ============================================================

print("=" * 75)
print("STEP 3C — SPATIAL TRANSFER DIAGNOSTICS")
print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading independent prediction dataset:")
print(PREDICTIONS_FILE)

if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Prediction file not found:\n{PREDICTIONS_FILE}"
    )

df = pd.read_csv(PREDICTIONS_FILE)

print(f"\nRows loaded: {len(df)}")


# ------------------------------------------------------------
# Remove invalid rows if necessary
# ------------------------------------------------------------

required_columns = (
    FEATURES
    + [
        TARGET,
        PREDICTED,
        RESIDUAL,
        ABS_ERROR,
        "station",
        "satellite_id",
        "cloud_percentage",
        "days_difference",
    ]
)

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


df = df.dropna(
    subset=[
        TARGET,
        PREDICTED,
        RESIDUAL,
        ABS_ERROR,
    ]
).copy()


print(
    f"Rows used for diagnostics: {len(df)}"
)


# ============================================================
# BASIC DIAGNOSTIC SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("BASIC DIAGNOSTIC SUMMARY")
print("=" * 75)

print(
    f"\nObserved turbidity mean : "
    f"{df[TARGET].mean():.3f}"
)

print(
    f"Predicted turbidity mean: "
    f"{df[PREDICTED].mean():.3f}"
)

print(
    f"Observed turbidity median : "
    f"{df[TARGET].median():.3f}"
)

print(
    f"Predicted turbidity median: "
    f"{df[PREDICTED].median():.3f}"
)

print(
    f"\nObserved minimum : "
    f"{df[TARGET].min():.3f}"
)

print(
    f"Observed maximum : "
    f"{df[TARGET].max():.3f}"
)

print(
    f"Predicted minimum : "
    f"{df[PREDICTED].min():.3f}"
)

print(
    f"Predicted maximum : "
    f"{df[PREDICTED].max():.3f}"
)


# ============================================================
# 1. STATION DIAGNOSTICS
# ============================================================

print("\n" + "=" * 75)
print("1. STATION-LEVEL DIAGNOSTICS")
print("=" * 75)

station_rows = []

for station, group in df.groupby(
    "station",
    sort=True
):

    station_rows.append(
        {
            "station": station,
            "n": len(group),
            "observed_mean_NTU": group[TARGET].mean(),
            "predicted_mean_NTU": group[PREDICTED].mean(),
            "observed_median_NTU": group[TARGET].median(),
            "predicted_median_NTU": group[PREDICTED].median(),
            "observed_max_NTU": group[TARGET].max(),
            "predicted_max_NTU": group[PREDICTED].max(),
            "MAE_NTU": group[ABS_ERROR].mean(),
            "mean_residual_NTU": group[RESIDUAL].mean(),
            "median_absolute_error_NTU": group[ABS_ERROR].median(),
            "underprediction_percent": (
                np.mean(
                    group[PREDICTED]
                    < group[TARGET]
                )
                * 100
            ),
        }
    )

station_df = pd.DataFrame(
    station_rows
)

print(
    station_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

station_df.to_csv(
    STATION_RESULTS_FILE,
    index=False
)


# ============================================================
# 2. CLOUD DIAGNOSTICS
# ============================================================

print("\n" + "=" * 75)
print("2. CLOUD-CONTAMINATION DIAGNOSTICS")
print("=" * 75)


def cloud_category(value):

    if value <= 10:
        return "0–10%"

    elif value <= 20:
        return "10–20%"

    elif value <= 40:
        return "20–40%"

    elif value <= 60:
        return "40–60%"

    elif value <= 80:
        return "60–80%"

    else:
        return "80–100%"


df["cloud_category"] = (
    df["cloud_percentage"]
    .apply(cloud_category)
)


cloud_order = [
    "0–10%",
    "10–20%",
    "20–40%",
    "40–60%",
    "60–80%",
    "80–100%",
]

cloud_rows = []

for category in cloud_order:

    group = df[
        df["cloud_category"]
        == category
    ]

    if len(group) == 0:
        continue

    cloud_rows.append(
        {
            "cloud_category": category,
            "n": len(group),
            "observed_mean_NTU": group[TARGET].mean(),
            "predicted_mean_NTU": group[PREDICTED].mean(),
            "MAE_NTU": group[ABS_ERROR].mean(),
            "mean_residual_NTU": group[RESIDUAL].mean(),
            "underprediction_percent": (
                np.mean(
                    group[PREDICTED]
                    < group[TARGET]
                )
                * 100
            ),
        }
    )

cloud_df = pd.DataFrame(
    cloud_rows
)

print(
    cloud_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

cloud_df.to_csv(
    CLOUD_RESULTS_FILE,
    index=False
)


# ============================================================
# 3. TEMPORAL MATCHING DIAGNOSTICS
# ============================================================

print("\n" + "=" * 75)
print("3. TEMPORAL-MISMATCH DIAGNOSTICS")
print("=" * 75)


def time_category(value):

    value = abs(value)

    if value <= 0.5:
        return "≤0.5 day"

    elif value <= 1.0:
        return "0.5–1 day"

    elif value <= 1.5:
        return "1–1.5 days"

    else:
        return ">1.5 days"


df["time_category"] = (
    df["days_difference"]
    .apply(time_category)
)


time_order = [
    "≤0.5 day",
    "0.5–1 day",
    "1–1.5 days",
    ">1.5 days",
]

time_rows = []

for category in time_order:

    group = df[
        df["time_category"]
        == category
    ]

    if len(group) == 0:
        continue

    time_rows.append(
        {
            "time_category": category,
            "n": len(group),
            "observed_mean_NTU": group[TARGET].mean(),
            "predicted_mean_NTU": group[PREDICTED].mean(),
            "MAE_NTU": group[ABS_ERROR].mean(),
            "mean_residual_NTU": group[RESIDUAL].mean(),
            "underprediction_percent": (
                np.mean(
                    group[PREDICTED]
                    < group[TARGET]
                )
                * 100
            ),
        }
    )

time_df = pd.DataFrame(
    time_rows
)

print(
    time_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

time_df.to_csv(
    TIME_RESULTS_FILE,
    index=False
)


# ============================================================
# 4. TURBIDITY CATEGORY DIAGNOSTICS
# ============================================================

print("\n" + "=" * 75)
print("4. TURBIDITY CATEGORY DIAGNOSTICS")
print("=" * 75)


def turbidity_category(value):

    if value < 10:
        return "Low (<10)"

    elif value < 20:
        return "Moderate (10–20)"

    elif value < 50:
        return "Elevated (20–50)"

    elif value < 100:
        return "High (50–100)"

    else:
        return "Very high (>=100)"


df["turbidity_category"] = (
    df[TARGET]
    .apply(turbidity_category)
)


category_order = [
    "Low (<10)",
    "Moderate (10–20)",
    "Elevated (20–50)",
    "High (50–100)",
    "Very high (>=100)",
]

category_rows = []

for category in category_order:

    group = df[
        df["turbidity_category"]
        == category
    ]

    if len(group) == 0:
        continue

    category_rows.append(
        {
            "category": category,
            "n": len(group),
            "observed_mean_NTU": group[TARGET].mean(),
            "predicted_mean_NTU": group[PREDICTED].mean(),
            "observed_median_NTU": group[TARGET].median(),
            "predicted_median_NTU": group[PREDICTED].median(),
            "MAE_NTU": group[ABS_ERROR].mean(),
            "mean_residual_NTU": group[RESIDUAL].mean(),
            "underprediction_percent": (
                np.mean(
                    group[PREDICTED]
                    < group[TARGET]
                )
                * 100
            ),
        }
    )

category_df = pd.DataFrame(
    category_rows
)

print(
    category_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

category_df.to_csv(
    CATEGORY_RESULTS_FILE,
    index=False
)


# ============================================================
# 5. EXTREME-EVENT DIAGNOSTICS
# ============================================================

print("\n" + "=" * 75)
print("5. EXTREME-EVENT DIAGNOSTICS")
print("=" * 75)

extreme = df[
    df[TARGET] >= 100
].copy()

print(
    f"\nExtreme observations (>=100 NTU): "
    f"{len(extreme)}"
)

if len(extreme) > 0:

    print(
        f"Observed mean : "
        f"{extreme[TARGET].mean():.3f}"
    )

    print(
        f"Predicted mean: "
        f"{extreme[PREDICTED].mean():.3f}"
    )

    print(
        f"MAE           : "
        f"{extreme[ABS_ERROR].mean():.3f}"
    )

    print(
        f"Mean residual : "
        f"{extreme[RESIDUAL].mean():.3f}"
    )

    print(
        f"Underprediction: "
        f"{np.mean(extreme[PREDICTED] < extreme[TARGET]) * 100:.2f}%"
    )

    extreme_output = extreme[
        [
            "station",
            "cwc_datetime",
            TARGET,
            PREDICTED,
            RESIDUAL,
            ABS_ERROR,
            "cloud_percentage",
            "days_difference",
        ]
    ].sort_values(
        TARGET,
        ascending=False
    )

    extreme_output.to_csv(
        EXTREME_RESULTS_FILE,
        index=False
    )


# ============================================================
# 6. ERROR CORRELATIONS
# ============================================================

print("\n" + "=" * 75)
print("6. ERROR / FEATURE CORRELATIONS")
print("=" * 75)

diagnostic_variables = (
    FEATURES
    + [
        TARGET,
        PREDICTED,
        ABS_ERROR,
        RESIDUAL,
        "cloud_percentage",
        "days_difference",
    ]
)

correlation_rows = []

for variable in diagnostic_variables:

    # Make sure duplicate CSV column names do not
    # produce a DataFrame instead of a Series.
    variable_data = df.loc[:, variable]

    residual_data = df.loc[:, RESIDUAL]

    # If duplicate column names exist, use the first occurrence.
    if isinstance(variable_data, pd.DataFrame):
        variable_data = variable_data.iloc[:, 0]

    if isinstance(residual_data, pd.DataFrame):
        residual_data = residual_data.iloc[:, 0]

    valid = pd.DataFrame(
        {
            "variable": pd.to_numeric(
                variable_data,
                errors="coerce"
            ),
            "residual": pd.to_numeric(
                residual_data,
                errors="coerce"
            ),
        }
    ).dropna()

    if len(valid) < 3:
        correlation = np.nan

    else:
        correlation = valid[
            "variable"
        ].corr(
            valid["residual"]
        )

    correlation_rows.append(
        {
            "variable": variable,
            "correlation_with_residual": correlation,
        }
    )

correlation_df = pd.DataFrame(
    correlation_rows
)

correlation_df[
    "absolute_correlation"
] = correlation_df[
    "correlation_with_residual"
].abs()

correlation_df = correlation_df.sort_values(
    "absolute_correlation",
    ascending=False
)

print(
    correlation_df[
        [
            "variable",
            "correlation_with_residual",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

correlation_df.to_csv(
    CORRELATION_FILE,
    index=False
)


# ============================================================
# 7. FEATURE DISTRIBUTION SHIFT
# ============================================================

print("\n" + "=" * 75)
print("7. FEATURE DISTRIBUTION SHIFT")
print("=" * 75)

if not ETAWAH_FILE.exists():

    print(
        "\nEtawah development dataset not found."
    )

else:

    etawah = pd.read_csv(
        ETAWAH_FILE
    )

    shift_rows = []

    for feature in FEATURES:

        if feature not in etawah.columns:
            continue

        etawah_values = pd.to_numeric(
            etawah[feature],
            errors="coerce"
        ).dropna()

        test_values = pd.to_numeric(
            df[feature],
            errors="coerce"
        ).dropna()

        shift_rows.append(
            {
                "feature": feature,

                "etawah_n": len(
                    etawah_values
                ),

                "independent_n": len(
                    test_values
                ),

                "etawah_mean": (
                    etawah_values.mean()
                ),

                "independent_mean": (
                    test_values.mean()
                ),

                "etawah_median": (
                    etawah_values.median()
                ),

                "independent_median": (
                    test_values.median()
                ),

                "etawah_std": (
                    etawah_values.std()
                ),

                "independent_std": (
                    test_values.std()
                ),

                "mean_difference": (
                    test_values.mean()
                    - etawah_values.mean()
                ),

                "median_difference": (
                    test_values.median()
                    - etawah_values.median()
                ),
            }
        )

    shift_df = pd.DataFrame(
        shift_rows
    )

    print(
        shift_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    shift_df.to_csv(
        FEATURE_SHIFT_FILE,
        index=False
    )


# ============================================================
# 8. REFLECTANCE RANGE CHECK
# ============================================================

print("\n" + "=" * 75)
print("8. REFLECTANCE / FEATURE RANGE CHECK")
print("=" * 75)

reflectance_features = [
    "B11_mean",
    "B12_mean",
    "B8_mean",
]

for feature in reflectance_features:

    values = df[feature]

    print(
        f"\n{feature}:"
    )

    print(
        f"  min = {values.min():.3f}"
    )

    print(
        f"  max = {values.max():.3f}"
    )

    print(
        f"  >1 count = "
        f"{(values > 1).sum()}"
    )


# ============================================================
# 9. PREDICTION COMPRESSION
# ============================================================

print("\n" + "=" * 75)
print("9. PREDICTION COMPRESSION")
print("=" * 75)

observed_range = (
    df[TARGET].max()
    - df[TARGET].min()
)

predicted_range = (
    df[PREDICTED].max()
    - df[PREDICTED].min()
)

print(
    f"\nObserved range  : "
    f"{observed_range:.3f} NTU"
)

print(
    f"Predicted range : "
    f"{predicted_range:.3f} NTU"
)

print(
    f"Observed std    : "
    f"{df[TARGET].std():.3f}"
)

print(
    f"Predicted std   : "
    f"{df[PREDICTED].std():.3f}"
)

if observed_range != 0:

    compression_ratio = (
        predicted_range
        / observed_range
    )

    print(
        f"\nPrediction range / observed range: "
        f"{compression_ratio:.4f}"
    )


# ============================================================
# 10. HIGH-CLOUD EXTREME EVENTS
# ============================================================

print("\n" + "=" * 75)
print("10. EXTREME EVENTS BY CLOUD CONDITION")
print("=" * 75)

extreme = df[
    df[TARGET] >= 100
].copy()

if len(extreme) > 0:

    extreme["cloud_group"] = np.where(
        extreme["cloud_percentage"] <= 20,
        "Cloud <=20%",
        "Cloud >20%"
    )

    extreme_cloud = (
        extreme
        .groupby("cloud_group")
        .agg(
            n=(TARGET, "size"),
            observed_mean_NTU=(
                TARGET,
                "mean"
            ),
            predicted_mean_NTU=(
                PREDICTED,
                "mean"
            ),
            MAE_NTU=(
                ABS_ERROR,
                "mean"
            ),
            mean_residual_NTU=(
                RESIDUAL,
                "mean"
            ),
        )
        .reset_index()
    )

    extreme_cloud[
        "underprediction_percent"
    ] = (
        extreme
        .groupby("cloud_group")
        .apply(
            lambda x:
            np.mean(
                x[PREDICTED]
                < x[TARGET]
            ) * 100
        )
        .values
    )

    print(
        extreme_cloud.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )


# ============================================================
# 11. LOW-CLOUD HIGH-TURBIDITY EVENTS
# ============================================================

print("\n" + "=" * 75)
print("11. HIGH TURBIDITY WITH LOW CLOUD")
print("=" * 75)

low_cloud_high_turbidity = df[
    (df[TARGET] >= 100)
    & (df["cloud_percentage"] <= 20)
].copy()

print(
    f"\nHigh turbidity (>=100 NTU) "
    f"with cloud <=20%: "
    f"{len(low_cloud_high_turbidity)}"
)

if len(low_cloud_high_turbidity) > 0:

    print(
        low_cloud_high_turbidity[
            [
                "station",
                "cwc_datetime",
                TARGET,
                PREDICTED,
                RESIDUAL,
                "cloud_percentage",
                "days_difference",
            ]
        ]
        .sort_values(
            TARGET,
            ascending=False
        )
        .head(30)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("STEP 3C COMPLETE")
print("=" * 75)

print("\nDiagnostic files saved:")

print(
    f"  Station:\n  {STATION_RESULTS_FILE}"
)

print(
    f"\n  Cloud:\n  {CLOUD_RESULTS_FILE}"
)

print(
    f"\n  Temporal:\n  {TIME_RESULTS_FILE}"
)

print(
    f"\n  Turbidity category:\n  {CATEGORY_RESULTS_FILE}"
)

print(
    f"\n  Feature shift:\n  {FEATURE_SHIFT_FILE}"
)

print(
    f"\n  Extreme events:\n  {EXTREME_RESULTS_FILE}"
)

print(
    f"\n  Error correlations:\n  {CORRELATION_FILE}"
)

print("\nNo model was retrained or modified.")
print("=" * 75)