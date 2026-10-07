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
# STEP 3B
# FROZEN ETawah MODEL -> INDEPENDENT YAMUNA SPATIAL TEST
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# FILES
# ------------------------------------------------------------

ETAWAH_FILE = (
    Path(r"D:\Etawah_Water_Quality\data")
    / "ml_ready_turbidity.csv"
)

TEST_FILE = (
    DATA_DIR
    / "Yamuna_Independent_Spatial_Test_Features.csv"
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "independent_spatial_predictions.csv"
)

STATION_RESULTS_FILE = (
    RESULTS_DIR
    / "independent_station_performance.csv"
)

CATEGORY_RESULTS_FILE = (
    RESULTS_DIR
    / "independent_category_performance.csv"
)


# ------------------------------------------------------------
# EXACT FROZEN COMPACT-8 FEATURES
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# EXACT FROZEN MODEL
# ------------------------------------------------------------

MODEL = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=2,
    loss="huber",
    random_state=42,
)


# ============================================================
# START
# ============================================================

print("=" * 75)
print("STEP 3B — FROZEN MODEL INDEPENDENT SPATIAL VALIDATION")
print("=" * 75)


# ------------------------------------------------------------
# LOAD ETAWAH DEVELOPMENT DATA
# ------------------------------------------------------------

print("\nLoading Etawah development dataset:")
print(ETAWAH_FILE)

if not ETAWAH_FILE.exists():
    raise FileNotFoundError(
        f"Etawah dataset not found:\n{ETAWAH_FILE}"
    )

etawah = pd.read_csv(ETAWAH_FILE)

print(f"Etawah rows: {len(etawah)}")


# ------------------------------------------------------------
# LOAD INDEPENDENT DATA
# ------------------------------------------------------------

print("\nLoading independent Yamuna dataset:")
print(TEST_FILE)

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Independent dataset not found:\n{TEST_FILE}"
    )

test = pd.read_csv(TEST_FILE)

print(f"Independent rows: {len(test)}")


# ============================================================
# VERIFY FEATURES
# ============================================================

print("\n" + "-" * 75)
print("FEATURE VERIFICATION")
print("-" * 75)

for feature in FEATURES:

    if feature not in etawah.columns:
        raise ValueError(
            f"Missing Etawah feature: {feature}"
        )

    if feature not in test.columns:
        raise ValueError(
            f"Missing independent feature: {feature}"
        )

    print(f"✓ {feature}")


if TARGET not in etawah.columns:
    raise ValueError(
        "Etawah dataset does not contain turbidity_NTU."
    )

if TARGET not in test.columns:
    raise ValueError(
        "Independent dataset does not contain turbidity_NTU."
    )


# ============================================================
# PREPARE ETAWAH TRAINING DATA
# ============================================================

print("\n" + "-" * 75)
print("PREPARING ETAWAH DEVELOPMENT DATA")
print("-" * 75)

train = etawah[
    FEATURES + [TARGET]
].copy()

before = len(train)

train = train.dropna(
    subset=FEATURES + [TARGET]
).copy()

after = len(train)

print(f"Rows before cleaning : {before}")
print(f"Rows after cleaning  : {after}")

if after != 199:
    print(
        "\nWARNING:"
        f" Expected 199 labelled Etawah rows, but found {after}."
    )

X_train = train[FEATURES]
y_train = train[TARGET]


# ============================================================
# PREPARE INDEPENDENT TEST DATA
# ============================================================

print("\n" + "-" * 75)
print("PREPARING INDEPENDENT TEST DATA")
print("-" * 75)

test_labelled = test[
    FEATURES + [
        TARGET,
        "station",
        "satellite_id",
        "satellite_date",
        "cwc_datetime",
        "days_difference",
        "cloud_percentage",
        "latitude",
        "longitude",
    ]
].copy()

before_test = len(test_labelled)

test_labelled = test_labelled.dropna(
    subset=FEATURES + [TARGET]
).copy()

after_test = len(test_labelled)

print(f"Rows before removing missing target : {before_test}")
print(f"Rows used for accuracy evaluation   : {after_test}")

print(
    f"Rows without turbidity target       : "
    f"{before_test - after_test}"
)

if after_test != 660:
    print(
        "\nWARNING:"
        f" Expected 660 labelled independent rows, but found {after_test}."
    )


# ============================================================
# FIT FROZEN MODEL ON ETAWAH DEVELOPMENT DATA
# ============================================================

print("\n" + "-" * 75)
print("FITTING FROZEN ETAWAH MODEL")
print("-" * 75)

print("Training dataset : Etawah")
print(f"Training rows    : {len(X_train)}")
print(f"Features         : {len(FEATURES)}")

print("\nModel parameters:")
print("  n_estimators   = 300")
print("  learning_rate  = 0.03")
print("  max_depth      = 2")
print("  loss           = huber")
print("  random_state   = 42")

MODEL.fit(
    X_train,
    y_train
)

print("\n✓ Model fitted on Etawah development data.")
print("✓ Independent Yamuna data NOT used for fitting.")


# ============================================================
# PREDICT INDEPENDENT TEST
# ============================================================

print("\n" + "-" * 75)
print("GENERATING INDEPENDENT PREDICTIONS")
print("-" * 75)

X_test = test_labelled[FEATURES]

y_true = test_labelled[TARGET].to_numpy()

y_pred = MODEL.predict(X_test)

# Prevent negative predictions if any occur.
y_pred = np.maximum(y_pred, 0)


test_labelled["predicted_turbidity_NTU"] = y_pred

test_labelled["residual_NTU"] = (
    test_labelled[TARGET]
    - test_labelled["predicted_turbidity_NTU"]
)

test_labelled["absolute_error_NTU"] = (
    test_labelled["residual_NTU"].abs()
)

test_labelled["percentage_error"] = np.where(
    test_labelled[TARGET] != 0,
    (
        test_labelled["absolute_error_NTU"]
        / test_labelled[TARGET]
    ) * 100,
    np.nan,
)


# ============================================================
# OVERALL METRICS
# ============================================================

mae = mean_absolute_error(
    y_true,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred
    )
)

r2 = r2_score(
    y_true,
    y_pred
)

bias = np.mean(
    y_true - y_pred
)

correlation = np.corrcoef(
    y_true,
    y_pred
)[0, 1]


print("\n" + "=" * 75)
print("INDEPENDENT SPATIAL TEST — OVERALL RESULTS")
print("=" * 75)

print(f"\nN observations : {len(test_labelled)}")

print(f"\nMAE  : {mae:.3f} NTU")
print(f"RMSE : {rmse:.3f} NTU")
print(f"R²   : {r2:.3f}")
print(f"Bias : {bias:.3f} NTU")
print(f"r    : {correlation:.3f}")


# ============================================================
# STATION PERFORMANCE
# ============================================================

print("\n" + "=" * 75)
print("STATION-LEVEL PERFORMANCE")
print("=" * 75)

station_results = []

for station, group in test_labelled.groupby(
    "station",
    sort=True
):

    observed = group[TARGET].to_numpy()
    predicted = group[
        "predicted_turbidity_NTU"
    ].to_numpy()

    station_mae = mean_absolute_error(
        observed,
        predicted
    )

    station_rmse = np.sqrt(
        mean_squared_error(
            observed,
            predicted
        )
    )

    station_r2 = r2_score(
        observed,
        predicted
    )

    station_bias = np.mean(
        observed - predicted
    )

    station_results.append(
        {
            "station": station,
            "n": len(group),
            "observed_mean_NTU": np.mean(observed),
            "predicted_mean_NTU": np.mean(predicted),
            "observed_min_NTU": np.min(observed),
            "observed_max_NTU": np.max(observed),
            "MAE_NTU": station_mae,
            "RMSE_NTU": station_rmse,
            "R2": station_r2,
            "bias_NTU": station_bias,
        }
    )

station_results_df = pd.DataFrame(
    station_results
)

print(
    station_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

station_results_df.to_csv(
    STATION_RESULTS_FILE,
    index=False
)


# ============================================================
# TURBIDITY CATEGORIES
# ============================================================

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


test_labelled["turbidity_category"] = (
    test_labelled[TARGET]
    .apply(turbidity_category)
)


print("\n" + "=" * 75)
print("TURBIDITY CATEGORY PERFORMANCE")
print("=" * 75)

category_results = []

category_order = [
    "Low (<10)",
    "Moderate (10–20)",
    "Elevated (20–50)",
    "High (50–100)",
    "Very high (>=100)",
]

for category in category_order:

    group = test_labelled[
        test_labelled["turbidity_category"]
        == category
    ]

    if len(group) == 0:
        continue

    observed = group[TARGET].to_numpy()

    predicted = group[
        "predicted_turbidity_NTU"
    ].to_numpy()

    residual = (
        observed - predicted
    )

    category_results.append(
        {
            "category": category,
            "n": len(group),
            "observed_mean_NTU": np.mean(observed),
            "predicted_mean_NTU": np.mean(predicted),
            "MAE_NTU": mean_absolute_error(
                observed,
                predicted
            ),
            "RMSE_NTU": np.sqrt(
                mean_squared_error(
                    observed,
                    predicted
                )
            ),
            "mean_residual_NTU": np.mean(
                residual
            ),
            "underprediction_percent": (
                np.mean(
                    predicted < observed
                ) * 100
            ),
        }
    )

category_results_df = pd.DataFrame(
    category_results
)

print(
    category_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

category_results_df.to_csv(
    CATEGORY_RESULTS_FILE,
    index=False
)


# ============================================================
# EXTREME EVENTS
# ============================================================

print("\n" + "=" * 75)
print("EXTREME TURBIDITY EVENTS")
print("=" * 75)

extreme = test_labelled[
    test_labelled[TARGET] >= 100
].copy()

extreme = extreme.sort_values(
    TARGET,
    ascending=False
)

print(
    extreme[
        [
            "station",
            "cwc_datetime",
            TARGET,
            "predicted_turbidity_NTU",
            "residual_NTU",
            "cloud_percentage",
            "days_difference",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


if len(extreme) > 0:

    extreme_mae = mean_absolute_error(
        extreme[TARGET],
        extreme["predicted_turbidity_NTU"]
    )

    extreme_rmse = np.sqrt(
        mean_squared_error(
            extreme[TARGET],
            extreme["predicted_turbidity_NTU"]
        )
    )

    extreme_underprediction = (
        np.mean(
            extreme["predicted_turbidity_NTU"]
            < extreme[TARGET]
        ) * 100
    )

    print(
        f"\nExtreme-event N : {len(extreme)}"
    )

    print(
        f"Extreme MAE     : {extreme_mae:.3f} NTU"
    )

    print(
        f"Extreme RMSE    : {extreme_rmse:.3f} NTU"
    )

    print(
        "Extreme underprediction : "
        f"{extreme_underprediction:.1f}%"
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

test_labelled.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("STEP 3B COMPLETE")
print("=" * 75)

print("\nFrozen Etawah model:")
print("  ✓ trained only on Etawah development data")
print("  ✓ 8 Compact-8 features")
print("  ✓ no independent-test fitting")
print("  ✓ no independent-test tuning")

print("\nIndependent evaluation:")
print(f"  Observations : {len(test_labelled)}")
print(f"  MAE          : {mae:.3f} NTU")
print(f"  RMSE         : {rmse:.3f} NTU")
print(f"  R²           : {r2:.3f}")
print(f"  Bias         : {bias:.3f} NTU")
print(f"  Correlation  : {correlation:.3f}")

print("\nSaved files:")

print(
    f"  Predictions:\n  {PREDICTIONS_FILE}"
)

print(
    f"\n  Station performance:\n  "
    f"{STATION_RESULTS_FILE}"
)

print(
    f"\n  Category performance:\n  "
    f"{CATEGORY_RESULTS_FILE}"
)

print("\n" + "=" * 75)