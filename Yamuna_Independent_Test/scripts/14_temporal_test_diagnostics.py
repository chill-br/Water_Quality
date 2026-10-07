from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# STEP 14 — 2024 INDEPENDENT TEMPORAL TEST DIAGNOSTICS
# ============================================================

ROOT = Path(
    r"D:\Etawah_Water_Quality\Yamuna_Independent_Test"
)

INPUT_FILE = (
    ROOT
    / "results"
    / "temporal_2024_frozen_model_predictions.csv"
)

RESULTS_DIR = ROOT / "results"

CLOUD_FILE = (
    RESULTS_DIR
    / "temporal_2024_cloud_performance.csv"
)

TIME_FILE = (
    RESULTS_DIR
    / "temporal_2024_timegap_performance.csv"
)

ERROR_FILE = (
    RESULTS_DIR
    / "temporal_2024_error_ranking.csv"
)


print("=" * 75)
print("STEP 14 — 2024 TEMPORAL TEST DIAGNOSTICS")
print("=" * 75)


# ============================================================
# LOAD
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nFile not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nRows:", len(df))


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "cwc_datetime",
    "turbidity_NTU",
    "predicted_turbidity_NTU",
    "residual_observed_minus_predicted",
    "days_difference",
    "cloud_percentage",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# NUMERIC FIELDS
# ============================================================

for col in [
    "turbidity_NTU",
    "predicted_turbidity_NTU",
    "residual_observed_minus_predicted",
    "days_difference",
    "cloud_percentage",
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )


df["absolute_error"] = (
    df["residual_observed_minus_predicted"]
    .abs()
)


# ============================================================
# HELPER
# ============================================================

def calculate_metrics(subset):

    n = len(subset)

    if n == 0:
        return {
            "n": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "bias": np.nan,
            "underprediction_percent": np.nan,
        }

    y = subset["turbidity_NTU"].to_numpy(
        dtype=float
    )

    p = subset[
        "predicted_turbidity_NTU"
    ].to_numpy(dtype=float)

    mae = mean_absolute_error(y, p)

    rmse = np.sqrt(
        mean_squared_error(y, p)
    )

    if n >= 2:
        r2 = r2_score(y, p)
    else:
        r2 = np.nan

    bias = np.mean(y - p)

    underprediction = (
        p < y
    ).mean() * 100

    return {
        "n": n,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "bias_observed_minus_predicted": bias,
        "underprediction_percent": underprediction,
    }


# ============================================================
# 1. CLOUD DIAGNOSTICS
# ============================================================

print("\n" + "-" * 75)
print("1. CLOUD COVER DIAGNOSTICS")
print("-" * 75)

print(
    "\nCloud statistics:"
)

print(
    df["cloud_percentage"].describe()
)


cloud_groups = [
    ("<=20%", df["cloud_percentage"] <= 20),
    ("<=40%", df["cloud_percentage"] <= 40),
    ("<=60%", df["cloud_percentage"] <= 60),
    (">60%", df["cloud_percentage"] > 60),
]


cloud_rows = []

for name, mask in cloud_groups:

    subset = df[mask].copy()

    metrics = calculate_metrics(subset)

    row = {
        "cloud_group": name,
        **metrics,
    }

    cloud_rows.append(row)

    print(f"\n{name}")

    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            print(
                f"  {key}: {value:.3f}"
            )
        else:
            print(
                f"  {key}: {value}"
            )


cloud_df = pd.DataFrame(
    cloud_rows
)


# ============================================================
# 2. TEMPORAL GAP DIAGNOSTICS
# ============================================================

print("\n" + "-" * 75)
print("2. TEMPORAL GAP DIAGNOSTICS")
print("-" * 75)

time_groups = [
    ("<=0.5 day", df["days_difference"] <= 0.5),
    ("<=1.0 day", df["days_difference"] <= 1.0),
    ("<=1.5 days", df["days_difference"] <= 1.5),
    (">1.5 days", df["days_difference"] > 1.5),
]


time_rows = []

for name, mask in time_groups:

    subset = df[mask].copy()

    metrics = calculate_metrics(subset)

    row = {
        "time_group": name,
        **metrics,
    }

    time_rows.append(row)

    print(f"\n{name}")

    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            print(
                f"  {key}: {value:.3f}"
            )
        else:
            print(
                f"  {key}: {value}"
            )


time_df = pd.DataFrame(
    time_rows
)


# ============================================================
# 3. TURBIDITY CATEGORIES
# ============================================================

print("\n" + "-" * 75)
print("3. TURBIDITY CATEGORY DIAGNOSTICS")
print("-" * 75)

category_definitions = [
    (
        "Low (<10)",
        df["turbidity_NTU"] < 10,
    ),
    (
        "Moderate (10–20)",
        (
            (df["turbidity_NTU"] >= 10)
            & (df["turbidity_NTU"] < 20)
        ),
    ),
    (
        "Elevated (20–50)",
        (
            (df["turbidity_NTU"] >= 20)
            & (df["turbidity_NTU"] < 50)
        ),
    ),
    (
        "High (50–100)",
        (
            (df["turbidity_NTU"] >= 50)
            & (df["turbidity_NTU"] < 100)
        ),
    ),
    (
        "Very high (>=100)",
        df["turbidity_NTU"] >= 100,
    ),
]


for name, mask in category_definitions:

    subset = df[mask].copy()

    metrics = calculate_metrics(subset)

    print(f"\n{name}")

    print(
        "  n:",
        metrics["n"]
    )

    if metrics["n"] > 0:
        print(
            "  MAE:",
            f"{metrics['MAE']:.3f}"
        )
        print(
            "  RMSE:",
            f"{metrics['RMSE']:.3f}"
        )
        print(
            "  bias:",
            f"{metrics['bias_observed_minus_predicted']:.3f}"
        )
        print(
            "  underprediction:",
            f"{metrics['underprediction_percent']:.1f}%"
        )


# ============================================================
# 4. ERROR RANKING
# ============================================================

print("\n" + "-" * 75)
print("4. INDIVIDUAL ERROR RANKING")
print("-" * 75)

error_columns = [
    "cwc_datetime",
    "turbidity_NTU",
    "predicted_turbidity_NTU",
    "residual_observed_minus_predicted",
    "absolute_error",
    "days_difference",
    "cloud_percentage",
    "satellite_date",
    "satellite_id",
]

available_error_columns = [
    c for c in error_columns
    if c in df.columns
]

error_df = (
    df[available_error_columns]
    .sort_values(
        "absolute_error",
        ascending=False,
    )
    .reset_index(drop=True)
)

print(
    error_df.to_string(index=False)
)


# ============================================================
# 5. WORST ERRORS
# ============================================================

print("\n" + "-" * 75)
print("5. TOP 5 LARGEST ABSOLUTE ERRORS")
print("-" * 75)

print(
    error_df.head(5).to_string(
        index=False
    )
)


# ============================================================
# 6. CLOUD + HIGH TURBIDITY CHECK
# ============================================================

print("\n" + "-" * 75)
print("6. HIGH TURBIDITY UNDER LOW CLOUD")
print("-" * 75)

high_turbidity = df[
    df["turbidity_NTU"] >= 50
].copy()

low_cloud_high_turbidity = high_turbidity[
    high_turbidity["cloud_percentage"] <= 20
].copy()

print(
    "High turbidity observations:",
    len(high_turbidity)
)

print(
    "High turbidity with cloud <=20%:",
    len(low_cloud_high_turbidity)
)

if len(low_cloud_high_turbidity) > 0:

    print(
        low_cloud_high_turbidity[
            [
                "cwc_datetime",
                "turbidity_NTU",
                "predicted_turbidity_NTU",
                "absolute_error",
                "days_difference",
                "cloud_percentage",
            ]
        ].to_string(index=False)
    )


# ============================================================
# 7. LOW CLOUD OVERALL PERFORMANCE
# ============================================================

print("\n" + "-" * 75)
print("7. LOW-CLOUD 2024 TEST")
print("-" * 75)

low_cloud = df[
    df["cloud_percentage"] <= 20
].copy()

metrics_low_cloud = calculate_metrics(
    low_cloud
)

print(
    "\nObservations:",
    len(low_cloud)
)

for key, value in metrics_low_cloud.items():

    if isinstance(value, (float, np.floating)):
        print(
            f"{key}: {value:.3f}"
        )
    else:
        print(
            f"{key}: {value}"
        )


# ============================================================
# 8. VERY CLOUDY TEST
# ============================================================

print("\n" + "-" * 75)
print("8. VERY CLOUDY 2024 TEST")
print("-" * 75)

very_cloudy = df[
    df["cloud_percentage"] > 60
].copy()

metrics_very_cloudy = calculate_metrics(
    very_cloudy
)

print(
    "\nObservations:",
    len(very_cloudy)
)

for key, value in metrics_very_cloudy.items():

    if isinstance(value, (float, np.floating)):
        print(
            f"{key}: {value:.3f}"
        )
    else:
        print(
            f"{key}: {value}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

cloud_df.to_csv(
    CLOUD_FILE,
    index=False,
)

time_df.to_csv(
    TIME_FILE,
    index=False,
)

error_df.to_csv(
    ERROR_FILE,
    index=False,
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("STEP 14 COMPLETE")
print("=" * 75)

print("\nSaved:")

print(
    "\nCloud performance:"
)
print(CLOUD_FILE)

print(
    "\nTemporal-gap performance:"
)
print(TIME_FILE)

print(
    "\nError ranking:"
)
print(ERROR_FILE)

print(
    "\nIMPORTANT:"
)
print(
    "No model retraining or parameter tuning was performed."
)
print(
    "The frozen 2024 predictions were used unchanged."
)