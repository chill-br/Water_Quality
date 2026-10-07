from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import pearsonr


# ============================================================
# STEP 13
# FROZEN ETawah MODEL — INDEPENDENT TEMPORAL TEST
#
# Train: 2021–2023
# Test:  2024
#
# IMPORTANT:
# - 2024 observations are NOT used during training.
# - Model hyperparameters are exactly frozen from Etawah.
# ============================================================


ROOT = Path(
    r"D:\Etawah_Water_Quality\Yamuna_Independent_Test"
)

INPUT_FILE = (
    ROOT
    / "results"
    / "correct_temporal_compact8_verified.csv"
)

RESULTS_DIR = ROOT / "results"

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "temporal_2024_frozen_model_predictions.csv"
)

METRICS_FILE = (
    RESULTS_DIR
    / "temporal_2024_frozen_model_metrics.csv"
)

CATEGORY_FILE = (
    RESULTS_DIR
    / "temporal_2024_category_performance.csv"
)

EXTREME_FILE = (
    RESULTS_DIR
    / "temporal_2024_extreme_events.csv"
)


# ============================================================
# EXACT FROZEN COMPACT-8 MODEL
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


MODEL = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=2,
    loss="huber",
    random_state=42,
)


# ============================================================
# LOAD VERIFIED DATA
# ============================================================

print("=" * 75)
print("STEP 13 — FROZEN MODEL INDEPENDENT TEMPORAL TEST")
print("=" * 75)

print("\nInput:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nVerified input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nTotal rows:", len(df))


# ============================================================
# BASIC VALIDATION
# ============================================================

required = FEATURES + [
    "station",
    "cwc_datetime",
    "turbidity_NTU",
    "satellite_id",
    "satellite_date",
    "days_difference",
    "cloud_percentage",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# CREATE YEAR
# ============================================================

df["cwc_datetime_parsed"] = pd.to_datetime(
    df["cwc_datetime"],
    dayfirst=True,
    errors="coerce",
)

df["cwc_year"] = (
    df["cwc_datetime_parsed"]
    .dt.year
)


# ============================================================
# TEMPORAL SPLIT
# ============================================================

train = df[
    df["cwc_year"].isin([2021, 2022, 2023])
].copy()

test = df[
    df["cwc_year"] == 2024
].copy()


print("\n" + "-" * 75)
print("TEMPORAL SPLIT")
print("-" * 75)

print("Training years: 2021–2023")
print("Test year:      2024")

print("\nTraining rows:", len(train))
print("2024 test rows:", len(test))


if len(train) == 0:
    raise ValueError("No training observations found.")

if len(test) == 0:
    raise ValueError("No 2024 test observations found.")


# ============================================================
# DUPLICATE / SENTINEL CHECK
# ============================================================

print("\n" + "-" * 75)
print("SENTINEL-2 GROUP CHECK")
print("-" * 75)

train_s2 = train["satellite_id"].nunique()
test_s2 = test["satellite_id"].nunique()

print("Unique training Sentinel-2 IDs:", train_s2)
print("Unique test Sentinel-2 IDs:", test_s2)

overlap = set(train["satellite_id"]) & set(
    test["satellite_id"]
)

print(
    "Sentinel-2 IDs shared between train/test:",
    len(overlap)
)

if overlap:
    print("\nWARNING — Sentinel-2 leakage detected:")
    for sid in sorted(overlap):
        print(" ", sid)
    raise ValueError(
        "Training and 2024 test contain the same Sentinel-2 ID."
    )

print("Train/test Sentinel-2 separation: PASS")


# ============================================================
# PREPARE X / y
# ============================================================

X_train = train[FEATURES].copy()
y_train = pd.to_numeric(
    train["turbidity_NTU"],
    errors="coerce",
)

X_test = test[FEATURES].copy()
y_test = pd.to_numeric(
    test["turbidity_NTU"],
    errors="coerce",
)


# ============================================================
# NUMERICAL CHECK
# ============================================================

if X_train.isna().any().any():
    raise ValueError(
        "Missing feature values in training data."
    )

if X_test.isna().any().any():
    raise ValueError(
        "Missing feature values in test data."
    )

if y_train.isna().any():
    raise ValueError(
        "Missing turbidity values in training data."
    )

if y_test.isna().any():
    raise ValueError(
        "Missing turbidity values in test data."
    )


# ============================================================
# TRAIN FROZEN MODEL
# ============================================================

print("\n" + "-" * 75)
print("TRAINING FROZEN MODEL")
print("-" * 75)

print("\nFeatures:")
for feature in FEATURES:
    print(" ", feature)

print("\nModel:")
print(MODEL)

print(
    "\nFitting ONLY on 2021–2023 observations..."
)

MODEL.fit(
    X_train,
    y_train,
)

print("Training complete.")


# ============================================================
# PREDICT 2024
# ============================================================

test["predicted_turbidity_NTU"] = (
    MODEL.predict(X_test)
)

test["residual_observed_minus_predicted"] = (
    test["turbidity_NTU"]
    - test["predicted_turbidity_NTU"]
)


# ============================================================
# OVERALL METRICS
# ============================================================

observed = y_test.to_numpy(dtype=float)

predicted = test[
    "predicted_turbidity_NTU"
].to_numpy(dtype=float)

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

r2 = r2_score(
    observed,
    predicted,
)

bias = np.mean(
    observed - predicted
)

if len(observed) >= 2:
    pearson_r, pearson_p = pearsonr(
        observed,
        predicted,
    )
else:
    pearson_r = np.nan
    pearson_p = np.nan


print("\n" + "=" * 75)
print("2024 INDEPENDENT TEMPORAL TEST RESULTS")
print("=" * 75)

print(f"\nN test observations: {len(test)}")

print(f"MAE:          {mae:.3f} NTU")
print(f"RMSE:         {rmse:.3f} NTU")
print(f"R²:           {r2:.3f}")
print(f"Bias:         {bias:.3f} NTU")
print(f"Pearson r:    {pearson_r:.3f}")
print(f"Pearson p:    {pearson_p:.5f}")

print(
    "\nObserved mean:",
    f"{observed.mean():.3f}",
)

print(
    "Predicted mean:",
    f"{predicted.mean():.3f}",
)

print(
    "Observed median:",
    f"{np.median(observed):.3f}",
)

print(
    "Predicted median:",
    f"{np.median(predicted):.3f}",
)


# ============================================================
# UNDERPREDICTION
# ============================================================

underprediction = (
    predicted < observed
)

under_count = int(
    underprediction.sum()
)

under_percent = (
    100 * under_count / len(test)
)

print(
    "\nUnderpredicted observations:",
    under_count,
    f"({under_percent:.1f}%)",
)


# ============================================================
# CATEGORY PERFORMANCE
# ============================================================

print("\n" + "-" * 75)
print("2024 TURBIDITY CATEGORY PERFORMANCE")
print("-" * 75)

categories = [
    (
        "Low (<10)",
        lambda x: x < 10,
    ),
    (
        "Moderate (10–20)",
        lambda x: (x >= 10) & (x < 20),
    ),
    (
        "Elevated (20–50)",
        lambda x: (x >= 20) & (x < 50),
    ),
    (
        "High (50–100)",
        lambda x: (x >= 50) & (x < 100),
    ),
    (
        "Very high (>=100)",
        lambda x: x >= 100,
    ),
]


category_rows = []

for name, condition in categories:

    mask = condition(observed)

    n = int(mask.sum())

    if n == 0:
        print(f"\n{name}: n=0")
        continue

    obs_cat = observed[mask]
    pred_cat = predicted[mask]

    cat_mae = mean_absolute_error(
        obs_cat,
        pred_cat,
    )

    cat_rmse = np.sqrt(
        mean_squared_error(
            obs_cat,
            pred_cat,
        )
    )

    cat_bias = np.mean(
        obs_cat - pred_cat
    )

    cat_under = (
        pred_cat < obs_cat
    ).mean() * 100

    print(f"\n{name}")
    print("  n:", n)
    print(
        "  observed mean:",
        f"{obs_cat.mean():.3f}",
    )
    print(
        "  predicted mean:",
        f"{pred_cat.mean():.3f}",
    )
    print(
        "  MAE:",
        f"{cat_mae:.3f}",
    )
    print(
        "  RMSE:",
        f"{cat_rmse:.3f}",
    )
    print(
        "  bias:",
        f"{cat_bias:.3f}",
    )
    print(
        "  underprediction:",
        f"{cat_under:.1f}%",
    )

    category_rows.append(
        {
            "category": name,
            "n": n,
            "observed_mean": obs_cat.mean(),
            "predicted_mean": pred_cat.mean(),
            "MAE": cat_mae,
            "RMSE": cat_rmse,
            "bias_observed_minus_predicted": cat_bias,
            "underprediction_percent": cat_under,
        }
    )


category_df = pd.DataFrame(
    category_rows
)


# ============================================================
# EXTREME EVENTS
# ============================================================

print("\n" + "-" * 75)
print("2024 EXTREME EVENTS")
print("-" * 75)

extreme_mask = observed >= 100

extreme = test.loc[
    extreme_mask,
    [
        "station",
        "cwc_datetime",
        "turbidity_NTU",
        "predicted_turbidity_NTU",
        "residual_observed_minus_predicted",
        "satellite_id",
        "satellite_date",
        "days_difference",
        "cloud_percentage",
    ],
].copy()

if len(extreme) == 0:

    print("\nNo 2024 observations >=100 NTU.")

else:

    extreme = extreme.sort_values(
        "turbidity_NTU",
        ascending=False,
    )

    print(
        extreme.to_string(
            index=False
        )
    )


# ============================================================
# INDIVIDUAL 2024 PREDICTIONS
# ============================================================

print("\n" + "=" * 75)
print("INDIVIDUAL 2024 PREDICTIONS")
print("=" * 75)

prediction_display = test[
    [
        "cwc_datetime",
        "turbidity_NTU",
        "predicted_turbidity_NTU",
        "residual_observed_minus_predicted",
        "satellite_date",
        "days_difference",
        "cloud_percentage",
    ]
].copy()

prediction_display = prediction_display.sort_values(
    "cwc_datetime_parsed"
    if "cwc_datetime_parsed" in prediction_display.columns
    else "cwc_datetime"
)

print(
    prediction_display.to_string(
        index=False
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

save_columns = [
    "station",
    "cwc_datetime",
    "cwc_year",
    "latitude",
    "longitude",
    "turbidity_NTU",
    "predicted_turbidity_NTU",
    "residual_observed_minus_predicted",
    "satellite_id",
    "satellite_date",
    "satellite_datetime",
    "days_difference",
    "cloud_percentage",
    "valid_B8_pixels",
    "MNDWI_median",
    "NDWI_median",
    "MNDWI_mean",
    "NDWI_mean",
    "water_fraction",
    "B11_mean",
    "B12_mean",
    "B8_mean",
]

prediction_output = test[
    save_columns
].copy()

prediction_output.to_csv(
    PREDICTIONS_FILE,
    index=False,
)


# ============================================================
# SAVE OVERALL METRICS
# ============================================================

metrics = pd.DataFrame(
    [
        {
            "test_year": 2024,
            "n_test": len(test),
            "training_years": "2021-2023",
            "MAE_NTU": mae,
            "RMSE_NTU": rmse,
            "R2": r2,
            "bias_observed_minus_predicted_NTU": bias,
            "Pearson_r": pearson_r,
            "Pearson_p": pearson_p,
            "observed_mean_NTU": observed.mean(),
            "predicted_mean_NTU": predicted.mean(),
            "observed_median_NTU": np.median(observed),
            "predicted_median_NTU": np.median(predicted),
            "underprediction_count": under_count,
            "underprediction_percent": under_percent,
        }
    ]
)

metrics.to_csv(
    METRICS_FILE,
    index=False,
)


# ============================================================
# SAVE CATEGORY RESULTS
# ============================================================

category_df.to_csv(
    CATEGORY_FILE,
    index=False,
)


# ============================================================
# SAVE EXTREME EVENTS
# ============================================================

extreme.to_csv(
    EXTREME_FILE,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FILES SAVED")
print("=" * 75)

print("\nPredictions:")
print(PREDICTIONS_FILE)

print("\nOverall metrics:")
print(METRICS_FILE)

print("\nCategory performance:")
print(CATEGORY_FILE)

print("\nExtreme events:")
print(EXTREME_FILE)

print("\n" + "=" * 75)
print("STEP 13 COMPLETE")
print("=" * 75)

print(
    "\nIMPORTANT: "
    "The 2024 observations were used only for independent evaluation."
)

print(
    "The model was fitted only on 2021–2023."
)