from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# STEP 8 — FORWARD TEMPORAL VALIDATION
# Train: 2021–2023
# Test : 2024
#
# IMPORTANT:
# This is a temporal generalization experiment.
# The 2024 observations are NOT used for training.
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

RESULTS = TEST_BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXACT FROZEN FEATURE DEFINITION
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
# MODEL
# ============================================================

MODEL_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=2,
    loss="huber",
    random_state=42,
)


print("=" * 75)
print("STEP 8 — FORWARD TEMPORAL VALIDATION")
print("=" * 75)

print("\nTraining dataset:")
print(TRAIN_FILE)

print("\nTesting dataset:")
print(TEST_FILE)


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"\nTraining rows loaded: {len(train)}")
print(f"Testing rows loaded : {len(test)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_train = FEATURES + [TARGET]

required_test = FEATURES + [TARGET]

missing_train = [
    c for c in required_train
    if c not in train.columns
]

missing_test = [
    c for c in required_test
    if c not in test.columns
]

if missing_train:
    raise ValueError(
        "Missing training columns:\n" +
        "\n".join(missing_train)
    )

if missing_test:
    raise ValueError(
        "Missing testing columns:\n" +
        "\n".join(missing_test)
    )


# ============================================================
# REMOVE EXACT DUPLICATE TEST OBSERVATIONS
# ============================================================

print("\n" + "=" * 75)
print("DUPLICATE CHECK")
print("=" * 75)

duplicate_subset = [
    c for c in [
        "station",
        "cwc_datetime",
        TARGET,
        "sentinel2_id",
        "satellite_date",
        "days_difference",
    ]
    if c in test.columns
]

duplicates_before = test.duplicated(
    subset=duplicate_subset,
    keep="first"
).sum()

print(
    f"Duplicate test rows detected: {duplicates_before}"
)

test = test.drop_duplicates(
    subset=duplicate_subset,
    keep="first"
).copy()

print(
    f"Testing rows after deduplication: {len(test)}"
)


# ============================================================
# CHECK TRAINING DUPLICATES
# ============================================================

train_duplicates = train.duplicated(
    subset=[
        c for c in [
            "station",
            "cwc_datetime",
            TARGET,
            "sentinel2_id",
            "satellite_date",
        ]
        if c in train.columns
    ],
    keep="first"
).sum()

print(
    f"Training duplicate rows: {train_duplicates}"
)


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

train = train.dropna(
    subset=FEATURES + [TARGET]
).copy()

test = test.dropna(
    subset=FEATURES + [TARGET]
).copy()

print("\nComplete cases:")
print(f"Training: {len(train)}")
print(f"Testing : {len(test)}")


# ============================================================
# FEATURE SCALE CHECK
# ============================================================

print("\n" + "=" * 75)
print("FEATURE SCALE CHECK")
print("=" * 75)

for feature in FEATURES:

    print(
        f"{feature:20s} "
        f"train_mean={train[feature].mean():.6f} "
        f"test_mean={test[feature].mean():.6f}"
    )


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 75)
print("TRAINING TEMPORAL MODEL")
print("=" * 75)

print("Training period: 2021–2023")
print("Testing period : 2024")

print("\nModel:")
print("GradientBoostingRegressor")
print("n_estimators=300")
print("learning_rate=0.03")
print("max_depth=2")
print("loss='huber'")
print("random_state=42")

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

model = GradientBoostingRegressor(
    **MODEL_PARAMS
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

test["predicted_turbidity_NTU"] = model.predict(
    X_test
)


# ============================================================
# RESIDUALS
# ============================================================

test["residual_NTU"] = (
    test[TARGET]
    - test["predicted_turbidity_NTU"]
)

test["absolute_error_NTU"] = (
    np.abs(test["residual_NTU"])
)


# ============================================================
# METRICS
# ============================================================

obs = test[TARGET].values
pred = test["predicted_turbidity_NTU"].values

mae = mean_absolute_error(
    obs,
    pred
)

rmse = np.sqrt(
    mean_squared_error(
        obs,
        pred
    )
)

r2 = r2_score(
    obs,
    pred
)

bias = np.mean(
    obs - pred
)

if len(test) >= 2:
    pearson_r = np.corrcoef(
        obs,
        pred
    )[0, 1]
else:
    pearson_r = np.nan


print("\n" + "=" * 75)
print("2024 INDEPENDENT TEMPORAL TEST")
print("=" * 75)

print(f"N          : {len(test)}")
print(f"MAE        : {mae:.3f} NTU")
print(f"RMSE       : {rmse:.3f} NTU")
print(f"R²         : {r2:.3f}")
print(f"Bias       : {bias:.3f} NTU")
print(f"Pearson r  : {pearson_r:.3f}")


# ============================================================
# TEST OBSERVATIONS
# ============================================================

print("\n" + "=" * 75)
print("2024 TEMPORAL TEST PREDICTIONS")
print("=" * 75)

display_columns = [
    "station",
    "cwc_datetime",
    TARGET,
    "predicted_turbidity_NTU",
    "residual_NTU",
    "absolute_error_NTU",
    "satellite_date",
    "days_difference",
    "cloud_percentage",
]

display_columns = [
    c for c in display_columns
    if c in test.columns
]

print(
    test[
        display_columns
    ]
    .sort_values("cwc_datetime")
    .to_string(index=False)
)


# ============================================================
# CATEGORY PERFORMANCE
# ============================================================

def turbidity_category(x):

    if x < 10:
        return "Low (<10)"

    elif x < 20:
        return "Moderate (10–20)"

    elif x < 50:
        return "Elevated (20–50)"

    elif x < 100:
        return "High (50–100)"

    else:
        return "Very high (≥100)"


test["turbidity_category"] = (
    test[TARGET]
    .apply(turbidity_category)
)


category_order = [
    "Low (<10)",
    "Moderate (10–20)",
    "Elevated (20–50)",
    "High (50–100)",
    "Very high (≥100)",
]

category_rows = []

for category in category_order:

    g = test[
        test["turbidity_category"] == category
    ]

    if len(g) == 0:
        continue

    category_rows.append({

        "category": category,

        "n": len(g),

        "observed_mean_NTU":
            g[TARGET].mean(),

        "predicted_mean_NTU":
            g["predicted_turbidity_NTU"].mean(),

        "MAE_NTU":
            g["absolute_error_NTU"].mean(),

        "RMSE_NTU":
            np.sqrt(
                np.mean(
                    g["residual_NTU"] ** 2
                )
            ),

        "Bias_NTU":
            g["residual_NTU"].mean(),

        "underprediction_percent":
            (
                g["predicted_turbidity_NTU"]
                < g[TARGET]
            ).mean() * 100,
    })


category_df = pd.DataFrame(
    category_rows
)


# ============================================================
# EXTREME EVENTS
# ============================================================

extreme = test[
    test[TARGET] >= 100
].copy()

print("\n" + "=" * 75)
print("TEMPORAL EXTREME EVENTS")
print("=" * 75)

print(
    f"Observations >=100 NTU: {len(extreme)}"
)

if len(extreme) > 0:

    print(
        extreme[
            [
                "cwc_datetime",
                TARGET,
                "predicted_turbidity_NTU",
                "residual_NTU",
                "cloud_percentage",
                "days_difference",
            ]
        ]
        .sort_values(
            TARGET,
            ascending=False
        )
        .to_string(index=False)
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_file = (
    RESULTS
    / "temporal_2024_predictions.csv"
)

test.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([{

    "training_period": "2021–2023",

    "testing_period": "2024",

    "n_train": len(train),

    "n_test": len(test),

    "MAE_NTU": mae,

    "RMSE_NTU": rmse,

    "R2": r2,

    "Bias_NTU": bias,

    "Pearson_r": pearson_r,

}])


metrics_file = (
    RESULTS
    / "temporal_2024_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# SAVE CATEGORY RESULTS
# ============================================================

category_file = (
    RESULTS
    / "temporal_2024_category_performance.csv"
)

category_df.to_csv(
    category_file,
    index=False
)


# ============================================================
# SAVE EXTREME EVENTS
# ============================================================

extreme_file = (
    RESULTS
    / "temporal_2024_extreme_events.csv"
)

extreme.to_csv(
    extreme_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("STEP 8 COMPLETE")
print("=" * 75)

print("\nFinal 2024 temporal test:")

print(
    f"Training N : {len(train)}"
)

print(
    f"Testing N  : {len(test)}"
)

print(
    f"MAE        : {mae:.3f} NTU"
)

print(
    f"RMSE       : {rmse:.3f} NTU"
)

print(
    f"R²         : {r2:.3f}"
)

print(
    f"Bias       : {bias:.3f} NTU"
)

print(
    f"Pearson r  : {pearson_r:.3f}"
)

print("\nSaved:")

print(
    prediction_file
)

print(
    metrics_file
)

print(
    category_file
)

print(
    extreme_file
)

print("\n2024 observations were NOT used for training.")

print("Temporal test model trained only on 2021–2023.")

print("=" * 75)