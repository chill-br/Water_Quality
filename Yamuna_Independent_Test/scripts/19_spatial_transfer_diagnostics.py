
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# STEP 19
# DIAGNOSTIC ANALYSIS OF FROZEN EXPANDED SPATIAL TEST
#
# IMPORTANT:
# - No model retraining
# - No modification of predictions
# - No test data added to training
# - Diagnostic analysis only
# ============================================================

BASE = Path(r"D:\Etawah_Water_Quality\Yamuna_Independent_Test")
RESULTS = BASE / "results"

PRED_FILE = RESULTS / "expanded_spatial_frozen_model_predictions.csv"
FEATURE_FILE = BASE / "data" / "Yamuna_Expanded_Compact8_Features_820.csv"
TRAIN_FILE = Path(r"D:\Etawah_Water_Quality\data\ml_ready_turbidity.csv")

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
# HELPER FUNCTIONS
# ============================================================

def metrics_table(df, observed_col, predicted_col):

    df = df[[observed_col, predicted_col]].dropna()

    if len(df) == 0:
        return {
            "n": 0,
            "MAE_NTU": np.nan,
            "RMSE_NTU": np.nan,
            "R2": np.nan,
            "bias_observed_minus_predicted_NTU": np.nan,
            "Pearson_r": np.nan,
            "Pearson_p": np.nan,
            "observed_mean_NTU": np.nan,
            "predicted_mean_NTU": np.nan,
        }

    y = df[observed_col].astype(float)
    p = df[predicted_col].astype(float)

    if len(df) >= 2 and y.nunique() > 1 and p.nunique() > 1:
        r, pval = pearsonr(y, p)
    else:
        r, pval = np.nan, np.nan

    return {
        "n": len(df),
        "MAE_NTU": mean_absolute_error(y, p),
        "RMSE_NTU": np.sqrt(mean_squared_error(y, p)),
        "R2": (
            r2_score(y, p)
            if len(df) >= 2 and y.nunique() > 1
            else np.nan
        ),
        "bias_observed_minus_predicted_NTU": (y - p).mean(),
        "Pearson_r": r,
        "Pearson_p": pval,
        "observed_mean_NTU": y.mean(),
        "predicted_mean_NTU": p.mean(),
    }


def add_metrics_row(output, name, df):

    row = metrics_table(
        df,
        "Turbidity (NTU)",
        "predicted_turbidity_NTU"
    )

    row["subset"] = name
    output.append(row)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 19 — SPATIAL TRANSFER DIAGNOSTICS")
print("=" * 70)

print("\nLoading frozen predictions...")
pred = pd.read_csv(PRED_FILE)

print("Prediction rows:", len(pred))

print("\nPrediction columns:")
print(pred.columns.tolist())


print("\nLoading Compact-8 feature table...")
features = pd.read_csv(FEATURE_FILE)

print("Feature rows:", len(features))


print("\nLoading Etawah training data...")
train = pd.read_csv(TRAIN_FILE)

print("Training rows:", len(train))


# ============================================================
# BASIC VALIDATION
# ============================================================

required_pred = [
    "Station",
    "Data Acquisition Time",
    "Turbidity (NTU)",
    "predicted_turbidity_NTU",
    "cloud_percentage",
    "days_difference",
    "sentinel2_id",
]

missing_pred = [c for c in required_pred if c not in pred.columns]

if missing_pred:
    raise ValueError(
        f"Missing required prediction columns: {missing_pred}"
    )

missing_features = [
    c for c in FEATURES
    if c not in features.columns
]

if missing_features:
    raise ValueError(
        f"Missing required Compact-8 features: {missing_features}"
    )


# ============================================================
# NORMALIZE DUPLICATE WATER FRACTION COLUMN
# ============================================================

# The prediction CSV contains both water_fraction and water_fraction.1.
# Use the original water_fraction as the canonical field.
if "water_fraction" in pred.columns:
    pred["water_fraction_canonical"] = pd.to_numeric(
        pred["water_fraction"],
        errors="coerce"
    )
elif "water_fraction.1" in pred.columns:
    pred["water_fraction_canonical"] = pd.to_numeric(
        pred["water_fraction.1"],
        errors="coerce"
    )
else:
    pred["water_fraction_canonical"] = np.nan


# ============================================================
# CREATE DIAGNOSTIC ERROR COLUMNS
# ============================================================

pred["Turbidity (NTU)"] = pd.to_numeric(
    pred["Turbidity (NTU)"],
    errors="coerce"
)

pred["predicted_turbidity_NTU"] = pd.to_numeric(
    pred["predicted_turbidity_NTU"],
    errors="coerce"
)

pred["cloud_percentage"] = pd.to_numeric(
    pred["cloud_percentage"],
    errors="coerce"
)

pred["days_difference"] = pd.to_numeric(
    pred["days_difference"],
    errors="coerce"
)

pred["B8_mean"] = pd.to_numeric(
    pred["B8_mean"],
    errors="coerce"
)

pred["B11_mean"] = pd.to_numeric(
    pred["B11_mean"],
    errors="coerce"
)

pred["B12_mean"] = pd.to_numeric(
    pred["B12_mean"],
    errors="coerce"
)

pred["residual_observed_minus_predicted_NTU"] = (
    pred["Turbidity (NTU)"]
    - pred["predicted_turbidity_NTU"]
)

pred["absolute_error_NTU"] = (
    pred["residual_observed_minus_predicted_NTU"].abs()
)

pred["underpredicted"] = (
    pred["predicted_turbidity_NTU"]
    < pred["Turbidity (NTU)"]
)


# ============================================================
# 1. TURBIDITY-REGIME DIAGNOSTICS
# ============================================================

print("\n" + "=" * 70)
print("1. TURBIDITY-REGIME DIAGNOSTICS")
print("=" * 70)

diagnostic_rows = []

add_metrics_row(
    diagnostic_rows,
    "All labelled spatial test",
    pred
)

below_100 = pred[
    pred["Turbidity (NTU)"] < 100
].copy()

add_metrics_row(
    diagnostic_rows,
    "Observed turbidity <100 NTU",
    below_100
)

above_100 = pred[
    pred["Turbidity (NTU)"] >= 100
].copy()

add_metrics_row(
    diagnostic_rows,
    "Observed turbidity >=100 NTU",
    above_100
)

below_50 = pred[
    pred["Turbidity (NTU)"] < 50
].copy()

add_metrics_row(
    diagnostic_rows,
    "Observed turbidity <50 NTU",
    below_50
)

mid = pred[
    (pred["Turbidity (NTU)"] >= 50) &
    (pred["Turbidity (NTU)"] < 100)
].copy()

add_metrics_row(
    diagnostic_rows,
    "Observed turbidity 50–<100 NTU",
    mid
)

regime_df = pd.DataFrame(diagnostic_rows)

print(
    regime_df[
        [
            "subset",
            "n",
            "MAE_NTU",
            "RMSE_NTU",
            "R2",
            "bias_observed_minus_predicted_NTU",
            "Pearson_r",
        ]
    ].round(3).to_string(index=False)
)

regime_df.to_csv(
    RESULTS / "spatial_transfer_turbidity_regime_diagnostics.csv",
    index=False
)


# ============================================================
# 2. UNDERPREDICTION BY TURBIDITY
# ============================================================

print("\n" + "=" * 70)
print("2. UNDERPREDICTION DIAGNOSTICS")
print("=" * 70)

under_rows = []

bins = [
    ("<10", -np.inf, 10),
    ("10–<20", 10, 20),
    ("20–<50", 20, 50),
    ("50–<100", 50, 100),
    ("100–<200", 100, 200),
    ("200–<500", 200, 500),
    (">=500", 500, np.inf),
]

for label, lo, hi in bins:

    subset = pred[
        (pred["Turbidity (NTU)"] >= lo) &
        (pred["Turbidity (NTU)"] < hi)
    ].copy()

    under_rows.append({
        "turbidity_range": label,
        "n": len(subset),
        "observed_mean_NTU":
            subset["Turbidity (NTU)"].mean(),
        "predicted_mean_NTU":
            subset["predicted_turbidity_NTU"].mean(),
        "MAE_NTU":
            subset["absolute_error_NTU"].mean(),
        "mean_residual_observed_minus_predicted_NTU":
            subset["residual_observed_minus_predicted_NTU"].mean(),
        "underpredicted_percent":
            subset["underpredicted"].mean() * 100
            if len(subset)
            else np.nan,
    })

under_df = pd.DataFrame(under_rows)

print(
    under_df.round(3).to_string(index=False)
)

under_df.to_csv(
    RESULTS / "spatial_transfer_underprediction_by_turbidity.csv",
    index=False
)


# ============================================================
# 3. B8 > 1 DIAGNOSTIC
# ============================================================

print("\n" + "=" * 70)
print("3. B8 > 1 DIAGNOSTIC")
print("=" * 70)

b8_rows = pred[
    pred["B8_mean"] > 1
].copy()

print("Usable labelled rows:", len(pred))
print("B8 > 1 rows:", len(b8_rows))
print(
    "B8 > 1 percentage:",
    round(len(b8_rows) / len(pred) * 100, 3)
)

if len(b8_rows):

    b8_summary = pd.DataFrame({
        "metric": [
            "count",
            "mean_B8",
            "median_B8",
            "max_B8",
            "mean_turbidity",
            "mean_prediction",
            "MAE",
            "mean_cloud",
        ],
        "value": [
            len(b8_rows),
            b8_rows["B8_mean"].mean(),
            b8_rows["B8_mean"].median(),
            b8_rows["B8_mean"].max(),
            b8_rows["Turbidity (NTU)"].mean(),
            b8_rows["predicted_turbidity_NTU"].mean(),
            b8_rows["absolute_error_NTU"].mean(),
            b8_rows["cloud_percentage"].mean(),
        ]
    })

    print("\nB8 > 1 summary:")
    print(
        b8_summary.round(3).to_string(index=False)
    )

    b8_export_cols = [
        "Station",
        "Data Acquisition Time",
        "Turbidity (NTU)",
        "predicted_turbidity_NTU",
        "absolute_error_NTU",
        "B8_mean",
        "B11_mean",
        "B12_mean",
        "MNDWI_mean",
        "NDWI_mean",
        "water_fraction_canonical",
        "cloud_percentage",
        "days_difference",
        "sentinel2_id",
    ]

    b8_rows[
        b8_export_cols
    ].sort_values(
        "absolute_error_NTU",
        ascending=False
    ).to_csv(
        RESULTS / "spatial_transfer_B8_gt1_diagnostics.csv",
        index=False
    )

else:
    print("No B8 > 1 observations.")


# ============================================================
# 4. B8 <= 1 VS B8 > 1 MODEL ERROR
# ============================================================

print("\n" + "=" * 70)
print("4. B8 GROUP ERROR COMPARISON")
print("=" * 70)

b8_group_rows = []

for label, subset in [
    ("B8 <= 1", pred[pred["B8_mean"] <= 1]),
    ("B8 > 1", pred[pred["B8_mean"] > 1]),
]:

    row = metrics_table(
        subset,
        "Turbidity (NTU)",
        "predicted_turbidity_NTU"
    )

    row["B8_group"] = label
    row["mean_B8"] = subset["B8_mean"].mean()
    row["mean_cloud_percent"] = subset["cloud_percentage"].mean()

    b8_group_rows.append(row)

b8_group_df = pd.DataFrame(b8_group_rows)

print(
    b8_group_df.round(3).to_string(index=False)
)

b8_group_df.to_csv(
    RESULTS / "spatial_transfer_B8_group_performance.csv",
    index=False
)


# ============================================================
# 5. FEATURE DISTRIBUTION SHIFT
# ============================================================

print("\n" + "=" * 70)
print("5. FEATURE DISTRIBUTION SHIFT")
print("=" * 70)

train_features = train.copy()

missing_train_features = [
    c for c in FEATURES
    if c not in train_features.columns
]

if missing_train_features:
    raise ValueError(
        "Training dataset does not contain the frozen Compact-8 "
        f"feature columns: {missing_train_features}"
    )

shift_rows = []

for feature in FEATURES:

    train_values = pd.to_numeric(
        train_features[feature],
        errors="coerce"
    ).dropna()

    test_values = pd.to_numeric(
        pred[feature],
        errors="coerce"
    ).dropna()

    train_mean = train_values.mean()
    test_mean = test_values.mean()

    train_median = train_values.median()
    test_median = test_values.median()

    train_std = train_values.std()
    test_std = test_values.std()

    mean_shift = test_mean - train_mean

    if abs(train_mean) > 1e-12:
        mean_shift_percent = (
            mean_shift / abs(train_mean) * 100
        )
    else:
        mean_shift_percent = np.nan

    pooled_std = np.sqrt(
        (
            (len(train_values) - 1) * train_std**2
            + (len(test_values) - 1) * test_std**2
        )
        /
        (
            len(train_values)
            + len(test_values)
            - 2
        )
    )

    if pooled_std > 0:
        standardized_shift = (
            test_mean - train_mean
        ) / pooled_std
    else:
        standardized_shift = np.nan

    shift_rows.append({
        "feature": feature,
        "training_n": len(train_values),
        "test_n": len(test_values),
        "training_mean": train_mean,
        "test_mean": test_mean,
        "mean_shift": mean_shift,
        "mean_shift_percent": mean_shift_percent,
        "training_median": train_median,
        "test_median": test_median,
        "training_std": train_std,
        "test_std": test_std,
        "standardized_mean_shift": standardized_shift,
    })

shift_df = pd.DataFrame(shift_rows)

print(
    shift_df.round(4).to_string(index=False)
)

shift_df.to_csv(
    RESULTS / "spatial_feature_distribution_shift.csv",
    index=False
)


# ============================================================
# 6. STATION-LEVEL FEATURE SHIFT
# ============================================================

print("\n" + "=" * 70)
print("6. STATION-LEVEL FEATURE SHIFT")
print("=" * 70)

station_feature_rows = []

for station, group in pred.groupby("Station"):

    for feature in FEATURES:

        train_values = pd.to_numeric(
            train_features[feature],
            errors="coerce"
        ).dropna()

        test_values = pd.to_numeric(
            group[feature],
            errors="coerce"
        ).dropna()

        if len(test_values) == 0:
            continue

        train_mean = train_values.mean()
        test_mean = test_values.mean()

        station_feature_rows.append({
            "Station": station,
            "feature": feature,
            "training_mean": train_mean,
            "station_test_mean": test_mean,
            "mean_difference":
                test_mean - train_mean,
            "station_test_median":
                test_values.median(),
            "station_test_std":
                test_values.std(),
            "n": len(test_values),
        })

station_feature_df = pd.DataFrame(
    station_feature_rows
)

print(
    station_feature_df.round(4).to_string(index=False)
)

station_feature_df.to_csv(
    RESULTS / "spatial_station_feature_shift.csv",
    index=False
)


# ============================================================
# 7. STATION ERROR VS EXTREME-TURBIDITY BURDEN
# ============================================================

print("\n" + "=" * 70)
print("7. STATION ERROR VS EXTREME-TURBIDITY BURDEN")
print("=" * 70)

station_rows = []

for station, group in pred.groupby("Station"):

    total = len(group)

    extreme = group[
        group["Turbidity (NTU)"] >= 100
    ]

    very_extreme = group[
        group["Turbidity (NTU)"] >= 500
    ]

    metrics = metrics_table(
        group,
        "Turbidity (NTU)",
        "predicted_turbidity_NTU"
    )

    station_rows.append({
        "Station": station,
        "n": total,
        "MAE_NTU": metrics["MAE_NTU"],
        "RMSE_NTU": metrics["RMSE_NTU"],
        "R2": metrics["R2"],
        "bias_NTU":
            metrics["bias_observed_minus_predicted_NTU"],
        "observed_mean_NTU":
            group["Turbidity (NTU)"].mean(),
        "predicted_mean_NTU":
            group["predicted_turbidity_NTU"].mean(),
        "percent_ge_100_NTU":
            len(extreme) / total * 100,
        "percent_ge_500_NTU":
            len(very_extreme) / total * 100,
        "n_ge_100_NTU":
            len(extreme),
        "n_ge_500_NTU":
            len(very_extreme),
    })

station_extreme_df = pd.DataFrame(
    station_rows
)

print(
    station_extreme_df.round(3).to_string(index=False)
)

station_extreme_df.to_csv(
    RESULTS / "spatial_station_extreme_burden.csv",
    index=False
)


# ============================================================
# 8. HIGH-TURBIDITY PERFORMANCE AT DIFFERENT CLOUD LEVELS
# ============================================================

print("\n" + "=" * 70)
print("8. HIGH-TURBIDITY CLOUD DIAGNOSTIC")
print("=" * 70)

high_turbidity = pred[
    pred["Turbidity (NTU)"] >= 100
].copy()

cloud_rows = []

for label, subset in [
    ("All high turbidity >=100", high_turbidity),

    (
        "High turbidity, cloud <=20%",
        high_turbidity[
            high_turbidity["cloud_percentage"] <= 20
        ]
    ),

    (
        "High turbidity, cloud <=40%",
        high_turbidity[
            high_turbidity["cloud_percentage"] <= 40
        ]
    ),

    (
        "High turbidity, cloud <=60%",
        high_turbidity[
            high_turbidity["cloud_percentage"] <= 60
        ]
    ),

    (
        "High turbidity, cloud >60%",
        high_turbidity[
            high_turbidity["cloud_percentage"] > 60
        ]
    ),
]:

    row = metrics_table(
        subset,
        "Turbidity (NTU)",
        "predicted_turbidity_NTU"
    )

    row["subset"] = label
    cloud_rows.append(row)

high_cloud_df = pd.DataFrame(
    cloud_rows
)

print(
    high_cloud_df.round(3).to_string(index=False)
)

high_cloud_df.to_csv(
    RESULTS / "spatial_high_turbidity_cloud_diagnostics.csv",
    index=False
)


# ============================================================
# 9. TOP 50 ERRORS
# ============================================================

print("\n" + "=" * 70)
print("9. TOP 50 ABSOLUTE ERRORS")
print("=" * 70)

top_errors = pred.sort_values(
    "absolute_error_NTU",
    ascending=False
).head(50)

print(
    top_errors[
        [
            "Station",
            "Data Acquisition Time",
            "Turbidity (NTU)",
            "predicted_turbidity_NTU",
            "absolute_error_NTU",
            "cloud_percentage",
            "days_difference",
            "B8_mean",
            "B11_mean",
            "B12_mean",
            "sentinel2_id",
        ]
    ].to_string(index=False)
)

top_errors.to_csv(
    RESULTS / "spatial_transfer_top50_errors_diagnostic.csv",
    index=False
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DIAGNOSTIC SUMMARY")
print("=" * 70)

print(
    "Frozen spatial test predictions:",
    len(pred)
)

print(
    "Observed turbidity <100 NTU:",
    len(below_100)
)

print(
    "Observed turbidity >=100 NTU:",
    len(above_100)
)

print(
    "Observed turbidity >=500 NTU:",
    len(
        pred[
            pred["Turbidity (NTU)"] >= 500
        ]
    )
)

print(
    "B8 > 1:",
    len(b8_rows)
)

print(
    "B8 > 1 percentage:",
    round(
        len(b8_rows) / len(pred) * 100,
        3
    )
)

print(
    "Mean absolute error <100 NTU:",
    round(
        below_100["absolute_error_NTU"].mean(),
        3
    )
)

print(
    "Mean absolute error >=100 NTU:",
    round(
        above_100["absolute_error_NTU"].mean(),
        3
    )
)

print("\nNo model retraining performed.")
print("No predictions modified.")
print("No test observations added to training.")

print("\nDiagnostic outputs written to:")
print(RESULTS)

print("\nSTEP 19 COMPLETE.")

