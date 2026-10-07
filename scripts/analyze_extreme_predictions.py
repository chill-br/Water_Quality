from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\Etawah_Water_Quality")

DATA_FILE = PROJECT_ROOT / "data" / "ml_ready_turbidity.csv"

PREDICTIONS_FILE = (
    PROJECT_ROOT / "data" / "extreme_prediction_details.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT / "data" / "extreme_prediction_summary.csv"
)

CATEGORY_FILE = (
    PROJECT_ROOT / "data" / "turbidity_category_errors.csv"
)

EXTREME_FILE = (
    PROJECT_ROOT / "data" / "extreme_turbidity_predictions.csv"
)


TARGET = "turbidity_NTU"
GROUP = "sentinel2_id"

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# COMPACT 8 FEATURES
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


# ============================================================
# MODEL
# ============================================================

def create_model():

    return GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        loss="huber",
        random_state=RANDOM_STATE,
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ETAWAH EXTREME TURBIDITY PREDICTION ANALYSIS")
print("=" * 70)

print(f"Reading: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print(f"Rows loaded: {len(df)}")


# ============================================================
# FILTER LABELLED DATA
# ============================================================

df = df[df[TARGET].notna()].copy()

print(f"Rows with turbidity labels: {len(df)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURES + [
    TARGET,
    GROUP,
    "patch_number",
    "patch_filename",
    "satellite_date",
    "cwc_datetime",
    "latitude",
    "longitude",
    "days_difference",
    "cloud_percentage",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR: Missing required columns:")

    for column in missing_columns:
        print(f"  - {column}")

    raise SystemExit(1)


# ============================================================
# CLEAN DATA
# ============================================================

df = df.replace([np.inf, -np.inf], np.nan)

before = len(df)

df = df.dropna(
    subset=FEATURES + [
        TARGET,
        GROUP,
    ]
).copy()

after = len(df)

print(f"Rows after feature cleaning: {after}")

if before != after:
    print(f"Rows removed: {before - after}")


# ============================================================
# BASIC DATA INFORMATION
# ============================================================

print(f"Unique Sentinel groups: {df[GROUP].nunique()}")

print(
    f"Turbidity range: "
    f"{df[TARGET].min():.3f} - "
    f"{df[TARGET].max():.3f} NTU"
)


# ============================================================
# CREATE OUT-OF-FOLD PREDICTIONS
# ============================================================

X = df[FEATURES]
y = df[TARGET]
groups = df[GROUP]

cv = GroupKFold(n_splits=N_SPLITS)

oof_predictions = np.full(len(df), np.nan)

fold_numbers = np.full(len(df), -1)

print("\n" + "=" * 70)
print("GENERATING GROUPED OUT-OF-FOLD PREDICTIONS")
print("=" * 70)

for fold_number, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups=groups),
    start=1,
):

    print(f"\nFold {fold_number}")

    print(f"  Training rows: {len(train_idx)}")
    print(f"  Validation rows: {len(test_idx)}")

    train_groups = set(
        groups.iloc[train_idx]
    )

    test_groups = set(
        groups.iloc[test_idx]
    )

    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise RuntimeError(
            "GROUP LEAKAGE DETECTED!"
        )

    print(
        f"  Training Sentinel groups: "
        f"{len(train_groups)}"
    )

    print(
        f"  Validation Sentinel groups: "
        f"{len(test_groups)}"
    )

    model = create_model()

    model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx],
    )

    predictions = model.predict(
        X.iloc[test_idx]
    )

    oof_predictions[test_idx] = predictions

    fold_numbers[test_idx] = fold_number

    fold_mae = mean_absolute_error(
        y.iloc[test_idx],
        predictions,
    )

    fold_rmse = np.sqrt(
        mean_squared_error(
            y.iloc[test_idx],
            predictions,
        )
    )

    fold_r2 = r2_score(
        y.iloc[test_idx],
        predictions,
    )

    print(
        f"  MAE={fold_mae:.3f} | "
        f"RMSE={fold_rmse:.3f} | "
        f"R²={fold_r2:.3f}"
    )


# ============================================================
# VERIFY ALL ROWS HAVE PREDICTIONS
# ============================================================

if np.isnan(oof_predictions).any():

    missing_predictions = np.isnan(
        oof_predictions
    ).sum()

    raise RuntimeError(
        f"{missing_predictions} rows have no "
        "out-of-fold prediction."
    )


# ============================================================
# BUILD PREDICTION DATAFRAME
# ============================================================

prediction_df = df.copy()

prediction_df["fold"] = fold_numbers

prediction_df["observed_turbidity"] = prediction_df[
    TARGET
]

prediction_df["predicted_turbidity"] = oof_predictions


# Residual:
# positive = model underpredicts
# negative = model overpredicts
prediction_df["residual"] = (
    prediction_df["observed_turbidity"]
    - prediction_df["predicted_turbidity"]
)


prediction_df["absolute_error"] = np.abs(
    prediction_df["residual"]
)


# Relative error is not meaningful when observed
# turbidity is zero. There are no zero values expected,
# but this protects the calculation.
prediction_df["relative_error_percent"] = np.where(
    prediction_df["observed_turbidity"] != 0,
    (
        prediction_df["residual"]
        / prediction_df["observed_turbidity"]
        * 100
    ),
    np.nan,
)


# Absolute relative error
prediction_df["absolute_relative_error_percent"] = (
    np.abs(
        prediction_df[
            "relative_error_percent"
        ]
    )
)


# Underprediction flag
prediction_df["underpredicted"] = (
    prediction_df["residual"] > 0
)


# ============================================================
# TURBIDITY CATEGORIES
# ============================================================

def turbidity_category(value):

    if value < 20:
        return "<20 NTU"

    elif value < 50:
        return "20–50 NTU"

    elif value < 100:
        return "50–100 NTU"

    else:
        return "≥100 NTU"


prediction_df["turbidity_category"] = (
    prediction_df[
        "observed_turbidity"
    ].apply(turbidity_category)
)


# ============================================================
# OVERALL MODEL PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("OVERALL OUT-OF-FOLD PERFORMANCE")
print("=" * 70)

overall_mae = mean_absolute_error(
    prediction_df["observed_turbidity"],
    prediction_df["predicted_turbidity"],
)

overall_rmse = np.sqrt(
    mean_squared_error(
        prediction_df["observed_turbidity"],
        prediction_df["predicted_turbidity"],
    )
)

overall_r2 = r2_score(
    prediction_df["observed_turbidity"],
    prediction_df["predicted_turbidity"],
)

print(f"MAE : {overall_mae:.3f} NTU")
print(f"RMSE: {overall_rmse:.3f} NTU")
print(f"R²  : {overall_r2:.3f}")


# ============================================================
# ERROR BY TURBIDITY CATEGORY
# ============================================================

print("\n" + "=" * 70)
print("ERROR BY TURBIDITY CATEGORY")
print("=" * 70)

category_order = [
    "<20 NTU",
    "20–50 NTU",
    "50–100 NTU",
    "≥100 NTU",
]

category_results = []

for category in category_order:

    subset = prediction_df[
        prediction_df["turbidity_category"]
        == category
    ].copy()

    if len(subset) == 0:
        continue

    observed = subset[
        "observed_turbidity"
    ]

    predicted = subset[
        "predicted_turbidity"
    ]

    mae = mean_absolute_error(
        observed,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            observed,
            predicted,
        )
    )

    if len(subset) >= 2:

        r2 = r2_score(
            observed,
            predicted,
        )

    else:

        r2 = np.nan

    mean_residual = subset[
        "residual"
    ].mean()

    median_absolute_error = subset[
        "absolute_error"
    ].median()

    underprediction_rate = (
        subset["underpredicted"].mean()
        * 100
    )

    mean_cloud = subset[
        "cloud_percentage"
    ].mean()

    median_cloud = subset[
        "cloud_percentage"
    ].median()

    mean_days_difference = subset[
        "days_difference"
    ].mean()

    median_days_difference = subset[
        "days_difference"
    ].median()

    category_results.append(
        {
            "turbidity_category": category,
            "n": len(subset),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "mean_residual": mean_residual,
            "median_absolute_error": median_absolute_error,
            "underprediction_rate_percent":
                underprediction_rate,
            "mean_cloud_percentage":
                mean_cloud,
            "median_cloud_percentage":
                median_cloud,
            "mean_days_difference":
                mean_days_difference,
            "median_days_difference":
                median_days_difference,
        }
    )

    print(
        f"\n{category}"
    )

    print(
        f"  n = {len(subset)}"
    )

    print(
        f"  MAE = {mae:.3f} NTU"
    )

    print(
        f"  RMSE = {rmse:.3f} NTU"
    )

    print(
        f"  R² = {r2:.3f}"
    )

    print(
        f"  Mean residual = "
        f"{mean_residual:.3f} NTU"
    )

    print(
        f"  Median absolute error = "
        f"{median_absolute_error:.3f} NTU"
    )

    print(
        f"  Underprediction rate = "
        f"{underprediction_rate:.1f}%"
    )

    print(
        f"  Mean cloud = "
        f"{mean_cloud:.3f}%"
    )

    print(
        f"  Mean date difference = "
        f"{mean_days_difference:.3f} days"
    )


category_df = pd.DataFrame(
    category_results
)


# ============================================================
# EXTREME TURBIDITY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("EXTREME TURBIDITY ANALYSIS (>=100 NTU)")
print("=" * 70)

extreme_df = prediction_df[
    prediction_df["observed_turbidity"] >= 100
].copy()

print(
    f"Extreme observations: "
    f"{len(extreme_df)}"
)


if len(extreme_df) > 0:

    extreme_mae = mean_absolute_error(
        extreme_df["observed_turbidity"],
        extreme_df["predicted_turbidity"],
    )

    extreme_rmse = np.sqrt(
        mean_squared_error(
            extreme_df["observed_turbidity"],
            extreme_df["predicted_turbidity"],
        )
    )

    extreme_mean_residual = (
        extreme_df["residual"].mean()
    )

    extreme_median_residual = (
        extreme_df["residual"].median()
    )

    extreme_underprediction_rate = (
        extreme_df["underpredicted"].mean()
        * 100
    )

    extreme_mean_cloud = (
        extreme_df["cloud_percentage"].mean()
    )

    extreme_mean_days_difference = (
        extreme_df["days_difference"].mean()
    )

    print(
        f"Extreme MAE: "
        f"{extreme_mae:.3f} NTU"
    )

    print(
        f"Extreme RMSE: "
        f"{extreme_rmse:.3f} NTU"
    )

    print(
        f"Mean residual: "
        f"{extreme_mean_residual:.3f} NTU"
    )

    print(
        f"Median residual: "
        f"{extreme_median_residual:.3f} NTU"
    )

    print(
        f"Underprediction rate: "
        f"{extreme_underprediction_rate:.1f}%"
    )

    print(
        f"Mean cloud percentage: "
        f"{extreme_mean_cloud:.3f}%"
    )

    print(
        f"Mean CWC-Sentinel difference: "
        f"{extreme_mean_days_difference:.3f} days"
    )

    # --------------------------------------------------------
    # SORT BY OBSERVED TURBIDITY
    # --------------------------------------------------------

    print("\nExtreme observations:")
    print("-" * 70)

    extreme_display = extreme_df[
        [
            "patch_number",
            GROUP,
            "satellite_date",
            "cwc_datetime",
            "observed_turbidity",
            "predicted_turbidity",
            "residual",
            "absolute_error",
            "relative_error_percent",
            "cloud_percentage",
            "days_difference",
            "water_fraction",
            "MNDWI_mean",
            "MNDWI_median",
            "NDWI_mean",
            "NDWI_median",
            "fold",
        ]
    ].sort_values(
        by="observed_turbidity",
        ascending=False,
    )

    print(
        extreme_display.to_string(
            index=False
        )
    )


# ============================================================
# WORST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("LARGEST ABSOLUTE ERRORS")
print("=" * 70)

worst_predictions = prediction_df[
    [
        "patch_number",
        GROUP,
        "satellite_date",
        "cwc_datetime",
        "observed_turbidity",
        "predicted_turbidity",
        "residual",
        "absolute_error",
        "relative_error_percent",
        "cloud_percentage",
        "days_difference",
        "water_fraction",
        "MNDWI_mean",
        "MNDWI_median",
        "NDWI_mean",
        "NDWI_median",
        "fold",
    ]
].sort_values(
    by="absolute_error",
    ascending=False,
).head(20)

print(
    worst_predictions.to_string(
        index=False
    )
)


# ============================================================
# EXTREME CLOUD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CLOUDINESS AND PREDICTION ERROR")
print("=" * 70)

cloud_bins = [
    (-np.inf, 5, "<5%"),
    (5, 20, "5–20%"),
    (20, 40, "20–40%"),
    (40, 60, "40–60%"),
    (60, np.inf, ">=60%"),
]

cloud_results = []

for lower, upper, label in cloud_bins:

    subset = prediction_df[
        (prediction_df["cloud_percentage"] > lower)
        & (prediction_df["cloud_percentage"] <= upper)
    ].copy()

    if len(subset) == 0:
        continue

    cloud_results.append(
        {
            "cloud_category": label,
            "n": len(subset),
            "mean_absolute_error":
                subset["absolute_error"].mean(),
            "median_absolute_error":
                subset["absolute_error"].median(),
            "mean_residual":
                subset["residual"].mean(),
            "mean_turbidity":
                subset["observed_turbidity"].mean(),
        }
    )

    print(
        f"{label}: "
        f"n={len(subset)} | "
        f"MAE={subset['absolute_error'].mean():.3f} | "
        f"mean residual={subset['residual'].mean():.3f}"
    )


cloud_df = pd.DataFrame(
    cloud_results
)


# ============================================================
# TEMPORAL MISMATCH ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CWC-SENTINEL TEMPORAL DIFFERENCE AND ERROR")
print("=" * 70)

time_bins = [
    (-np.inf, 0.5, "<=0.5 days"),
    (0.5, 1.0, "0.5–1.0 days"),
    (1.0, 1.5, "1.0–1.5 days"),
    (1.5, 2.0, "1.5–2.0 days"),
    (2.0, np.inf, ">2.0 days"),
]

time_results = []

for lower, upper, label in time_bins:

    subset = prediction_df[
        (prediction_df["days_difference"] > lower)
        & (prediction_df["days_difference"] <= upper)
    ].copy()

    if len(subset) == 0:
        continue

    time_results.append(
        {
            "days_difference_category": label,
            "n": len(subset),
            "mean_absolute_error":
                subset["absolute_error"].mean(),
            "median_absolute_error":
                subset["absolute_error"].median(),
            "mean_residual":
                subset["residual"].mean(),
            "mean_turbidity":
                subset["observed_turbidity"].mean(),
        }
    )

    print(
        f"{label}: "
        f"n={len(subset)} | "
        f"MAE={subset['absolute_error'].mean():.3f} | "
        f"mean residual={subset['residual'].mean():.3f}"
    )


time_df = pd.DataFrame(
    time_results
)


# ============================================================
# EXTREME OBSERVATIONS: CLOUD + TEMPORAL REVIEW
# ============================================================

if len(extreme_df) > 0:

    print("\n" + "=" * 70)
    print("EXTREME OBSERVATIONS: DATA QUALITY REVIEW")
    print("=" * 70)

    print(
        "\nExtreme observations with cloud >20%:"
    )

    high_cloud_extreme = extreme_df[
        extreme_df["cloud_percentage"] > 20
    ]

    if len(high_cloud_extreme) == 0:

        print("  None")

    else:

        print(
            high_cloud_extreme[
                [
                    "patch_number",
                    GROUP,
                    "observed_turbidity",
                    "predicted_turbidity",
                    "absolute_error",
                    "cloud_percentage",
                    "days_difference",
                ]
            ].sort_values(
                "cloud_percentage",
                ascending=False,
            ).to_string(index=False)
        )

    print(
        "\nExtreme observations with "
        "CWC-Sentinel difference >1.5 days:"
    )

    high_time_extreme = extreme_df[
        extreme_df["days_difference"] > 1.5
    ]

    if len(high_time_extreme) == 0:

        print("  None")

    else:

        print(
            high_time_extreme[
                [
                    "patch_number",
                    GROUP,
                    "observed_turbidity",
                    "predicted_turbidity",
                    "absolute_error",
                    "cloud_percentage",
                    "days_difference",
                ]
            ].sort_values(
                "days_difference",
                ascending=False,
            ).to_string(index=False)
        )


# ============================================================
# CORRELATION OF ERROR WITH CLOUD/TEMPORAL VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("ERROR CORRELATIONS")
print("=" * 70)

correlation_columns = [
    "observed_turbidity",
    "predicted_turbidity",
    "residual",
    "absolute_error",
    "cloud_percentage",
    "days_difference",
    "water_fraction",
    "MNDWI_mean",
    "MNDWI_median",
    "NDWI_mean",
    "NDWI_median",
    "B11_mean",
    "B12_mean",
    "B8_mean",
]

correlation_table = prediction_df[
    correlation_columns
].corr()

print(
    "\nCorrelation with absolute error:"
)

absolute_error_correlations = (
    correlation_table["absolute_error"]
    .sort_values(
        ascending=False
    )
)

print(
    absolute_error_correlations.to_string()
)


print(
    "\nCorrelation with residual:"
)

residual_correlations = (
    correlation_table["residual"]
    .sort_values(
        ascending=False
    )
)

print(
    residual_correlations.to_string()
)


# ============================================================
# SAVE DETAILED PREDICTIONS
# ============================================================

prediction_df = prediction_df.sort_values(
    by=[
        "observed_turbidity",
        "absolute_error",
    ],
    ascending=[
        False,
        False,
    ],
)

prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False,
)

print("\nSaved:")
print(PREDICTIONS_FILE)


# ============================================================
# SAVE CATEGORY SUMMARY
# ============================================================

category_df.to_csv(
    CATEGORY_FILE,
    index=False,
)

print("Saved:")
print(CATEGORY_FILE)


# ============================================================
# SAVE EXTREME OBSERVATIONS
# ============================================================

if len(extreme_df) > 0:

    extreme_df = extreme_df.sort_values(
        by="observed_turbidity",
        ascending=False,
    )

    extreme_df.to_csv(
        EXTREME_FILE,
        index=False,
    )

else:

    pd.DataFrame().to_csv(
        EXTREME_FILE,
        index=False,
    )

print("Saved:")
print(EXTREME_FILE)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_rows = []

summary_rows.append(
    {
        "analysis": "Overall",
        "n": len(prediction_df),
        "MAE": overall_mae,
        "RMSE": overall_rmse,
        "R2": overall_r2,
        "mean_residual":
            prediction_df["residual"].mean(),
        "underprediction_rate_percent":
            prediction_df["underpredicted"].mean() * 100,
        "mean_cloud_percentage":
            prediction_df["cloud_percentage"].mean(),
        "mean_days_difference":
            prediction_df["days_difference"].mean(),
    }
)


if len(extreme_df) > 0:

    summary_rows.append(
        {
            "analysis": "Extreme >=100 NTU",
            "n": len(extreme_df),
            "MAE": extreme_mae,
            "RMSE": extreme_rmse,
            "R2": np.nan,
            "mean_residual":
                extreme_mean_residual,
            "underprediction_rate_percent":
                extreme_underprediction_rate,
            "mean_cloud_percentage":
                extreme_mean_cloud,
            "mean_days_difference":
                extreme_mean_days_difference,
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)

print("Saved:")
print(SUMMARY_FILE)


# ============================================================
# SAVE CLOUD AND TEMPORAL ANALYSES
# ============================================================

CLOUD_FILE = (
    PROJECT_ROOT
    / "data"
    / "prediction_error_by_cloud.csv"
)

TIME_FILE = (
    PROJECT_ROOT
    / "data"
    / "prediction_error_by_date_difference.csv"
)

cloud_df.to_csv(
    CLOUD_FILE,
    index=False,
)

time_df.to_csv(
    TIME_FILE,
    index=False,
)

print("Saved:")
print(CLOUD_FILE)
print(TIME_FILE)


# ============================================================
# FINAL INTERPRETATION FLAGS
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION CHECKS")
print("=" * 70)

if len(extreme_df) > 0:

    if extreme_mean_residual > 0:

        print(
            "Extreme turbidity observations are, on average, "
            "UNDERPREDICTED."
        )

    else:

        print(
            "Extreme turbidity observations are, on average, "
            "OVERPREDICTED."
        )

    if extreme_underprediction_rate >= 70:

        print(
            "WARNING: >=70% of extreme observations are "
            "underpredicted."
        )

    if extreme_mean_cloud > 20:

        print(
            "NOTE: Extreme observations have relatively high "
            "mean cloud percentage (>20%)."
        )

    if extreme_mean_days_difference > 1.5:

        print(
            "NOTE: Extreme observations have relatively large "
            "mean CWC-Sentinel temporal difference (>1.5 days)."
        )


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)