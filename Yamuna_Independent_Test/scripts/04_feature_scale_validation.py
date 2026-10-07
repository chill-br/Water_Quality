from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# STEP 3D — FEATURE SCALE VALIDATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results"

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "independent_spatial_predictions.csv"
)

ETAWAH_FILE = (
    Path(r"D:\Etawah_Water_Quality\data")
    / "ml_ready_turbidity.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "independent_spatial_scale_corrected_predictions.csv"
)


# ============================================================
# FROZEN MODEL DEFINITION
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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("STEP 3D — FEATURE SCALE VALIDATION")
print("=" * 75)

print("\nLoading Etawah development dataset:")
print(ETAWAH_FILE)

etawah = pd.read_csv(
    ETAWAH_FILE
)

print(
    f"Etawah rows: {len(etawah)}"
)


print("\nLoading independent dataset:")
print(PREDICTIONS_FILE)

independent = pd.read_csv(
    PREDICTIONS_FILE
)

print(
    f"Independent rows: {len(independent)}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_etawah = (
    FEATURES
    + [TARGET, "sentinel2_id"]
)

required_independent = (
    FEATURES
    + [TARGET, "station"]
)

missing_etawah = [
    c for c in required_etawah
    if c not in etawah.columns
]

missing_independent = [
    c for c in required_independent
    if c not in independent.columns
]

if missing_etawah:

    raise ValueError(
        "Missing Etawah columns:\n"
        + "\n".join(missing_etawah)
    )

if missing_independent:

    raise ValueError(
        "Missing independent columns:\n"
        + "\n".join(missing_independent)
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for column in FEATURES + [TARGET]:

    etawah[column] = pd.to_numeric(
        etawah[column],
        errors="coerce"
    )

    independent[column] = pd.to_numeric(
        independent[column],
        errors="coerce"
    )


# ============================================================
# REMOVE UNLABELLED INDEPENDENT OBSERVATIONS
# ============================================================

independent = independent.dropna(
    subset=[TARGET]
).copy()

print(
    f"\nIndependent labelled observations: "
    f"{len(independent)}"
)


# ============================================================
# VERIFY TRAINING FEATURE SCALE
# ============================================================

print("\n" + "=" * 75)
print("TRAINING FEATURE SCALE")
print("=" * 75)

for feature in FEATURES:

    values = etawah[feature].dropna()

    print(
        f"{feature:20s} "
        f"min={values.min():.6f} "
        f"median={values.median():.6f} "
        f"mean={values.mean():.6f} "
        f"max={values.max():.6f}"
    )


# ============================================================
# VERIFY INDEPENDENT FEATURE SCALE BEFORE CORRECTION
# ============================================================

print("\n" + "=" * 75)
print("INDEPENDENT FEATURE SCALE — BEFORE CORRECTION")
print("=" * 75)

for feature in FEATURES:

    values = independent[feature].dropna()

    print(
        f"{feature:20s} "
        f"min={values.min():.6f} "
        f"median={values.median():.6f} "
        f"mean={values.mean():.6f} "
        f"max={values.max():.6f}"
    )


# ============================================================
# COPY INDEPENDENT DATA
# ============================================================

corrected = independent.copy()


# ============================================================
# SCALE CORRECTION
#
# Sentinel-2 optical bands in the independent dataset
# are on approximately 0–10000 integer reflectance scale.
#
# Etawah training data use approximately 0–1 reflectance scale.
#
# Therefore:
#
# B8_mean  / 10000
# B11_mean / 10000
# B12_mean / 10000
# ============================================================

REFLECTANCE_FEATURES = [
    "B8_mean",
    "B11_mean",
    "B12_mean",
]

for feature in REFLECTANCE_FEATURES:

    corrected[feature] = (
        corrected[feature] / 10000.0
    )


# ============================================================
# VERIFY CORRECTED SCALE
# ============================================================

print("\n" + "=" * 75)
print("INDEPENDENT FEATURE SCALE — AFTER CORRECTION")
print("=" * 75)

for feature in FEATURES:

    values = corrected[feature].dropna()

    print(
        f"{feature:20s} "
        f"min={values.min():.6f} "
        f"median={values.median():.6f} "
        f"mean={values.mean():.6f} "
        f"max={values.max():.6f}"
    )


# ============================================================
# TRAIN FROZEN ETawah MODEL
# ============================================================

print("\n" + "=" * 75)
print("TRAINING FROZEN ETAWAH MODEL")
print("=" * 75)

training = etawah.dropna(
    subset=FEATURES + [TARGET]
).copy()

X_train = training[
    FEATURES
]

y_train = training[
    TARGET
]

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=2,
    loss="huber",
    random_state=42,
)

model.fit(
    X_train,
    y_train
)

print(
    f"Training observations: {len(training)}"
)

print(
    "Model: GradientBoostingRegressor"
)

print(
    "n_estimators=300"
)

print(
    "learning_rate=0.03"
)

print(
    "max_depth=2"
)

print(
    "loss='huber'"
)

print(
    "random_state=42"
)


# ============================================================
# PREPARE INDEPENDENT TEST DATA
# ============================================================

test = corrected.dropna(
    subset=FEATURES + [TARGET]
).copy()

X_test = test[
    FEATURES
]

y_test = test[
    TARGET
]


# ============================================================
# PREDICT
# ============================================================

test[
    "predicted_turbidity_NTU_scale_corrected"
] = model.predict(
    X_test
)


# ============================================================
# METRICS
# ============================================================

y_pred = test[
    "predicted_turbidity_NTU_scale_corrected"
].values

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

bias = np.mean(
    y_test.values - y_pred
)

correlation = np.corrcoef(
    y_test.values,
    y_pred
)[0, 1]


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 75)
print("SCALE-CORRECTED INDEPENDENT VALIDATION")
print("=" * 75)

print(
    f"\nN       : {len(test)}"
)

print(
    f"MAE     : {mae:.3f} NTU"
)

print(
    f"RMSE    : {rmse:.3f} NTU"
)

print(
    f"R²      : {r2:.3f}"
)

print(
    f"Bias    : {bias:.3f} NTU"
)

print(
    f"Pearson r: {correlation:.3f}"
)


# ============================================================
# COMPARISON WITH PREVIOUS RESULT
# ============================================================

print("\n" + "=" * 75)
print("COMPARISON WITH ORIGINAL INDEPENDENT TEST")
print("=" * 75)

print(
    "\nOriginal:"
)

print(
    "MAE  = 47.315 NTU"
)

print(
    "RMSE = 97.447 NTU"
)

print(
    "R²   = -0.348"
)

print(
    "\nScale-corrected:"
)

print(
    f"MAE  = {mae:.3f} NTU"
)

print(
    f"RMSE = {rmse:.3f} NTU"
)

print(
    f"R²   = {r2:.3f}"
)


# ============================================================
# IMPROVEMENT
# ============================================================

original_mae = 47.315
original_rmse = 97.447
original_r2 = -0.348

mae_change = (
    (mae - original_mae)
    / original_mae
    * 100
)

rmse_change = (
    (rmse - original_rmse)
    / original_rmse
    * 100
)

print("\nChange relative to original:")

print(
    f"MAE change : {mae_change:.2f}%"
)

print(
    f"RMSE change: {rmse_change:.2f}%"
)

print(
    f"R² change  : {r2 - original_r2:.3f}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

output_columns = [
    "station",
    "cwc_datetime",
    TARGET,
    "sentinel2_id",
    "satellite_date",
    "days_difference",
    "cloud_percentage",
] + FEATURES + [
    "predicted_turbidity_NTU_scale_corrected",
]

output_columns = [
    c for c in output_columns
    if c in test.columns
]

test[
    output_columns
].to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("STEP 3D COMPLETE")
print("=" * 75)

print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)

print(
    "\nIMPORTANT:"
)

print(
    "The Etawah model was not retrained using "
    "independent test observations."
)

print(
    "Only the independent B8/B11/B12 feature "
    "scale was corrected to match the training representation."
)

print("=" * 75)