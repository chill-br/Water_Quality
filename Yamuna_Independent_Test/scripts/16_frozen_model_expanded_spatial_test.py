from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# ============================================================
# PATHS
# ============================================================

BASE = Path(r"D:\Etawah_Water_Quality")

TRAIN_FILE = (
    BASE
    / "data"
    / "ml_ready_turbidity.csv"
)


TEST_FILE = (
    BASE
    / "Yamuna_Independent_Test"
    / "data"
    / "Yamuna_Expanded_Independent_Spatial_Match_2021_2024.csv"
)

OUT_DIR = BASE / "Yamuna_Independent_Test" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FROZEN MODEL
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

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=2,
    loss="huber",
    random_state=42,
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FROZEN ETWAH MODEL — EXPANDED YAMUNA SPATIAL TEST")
print("=" * 70)

print("\nLoading training data:")
print(TRAIN_FILE)

train = pd.read_csv(TRAIN_FILE)

print("Training rows:", len(train))

print("\nLoading expanded spatial test:")
print(TEST_FILE)

test = pd.read_csv(TEST_FILE)

print("Test rows:", len(test))
print("Test columns:", len(test.columns))

# ============================================================
# BASIC COLUMN CHECK
# ============================================================

missing_train = [c for c in FEATURES + [TARGET] if c not in train.columns]
missing_test = [c for c in FEATURES if c not in test.columns]

if missing_train:
    raise ValueError(
        f"Missing required training columns: {missing_train}"
    )

if missing_test:
    raise ValueError(
        f"Missing required test feature columns: {missing_test}"
    )

# ============================================================
# IDENTIFY TARGET IN TEST
# ============================================================

if TARGET not in test.columns:
    raise ValueError(
        f"Test file does not contain '{TARGET}'. "
        "The expanded matching file must contain CWC turbidity "
        "before model accuracy can be calculated."
    )

# ============================================================
# REMOVE UNLABELLED TEST ROWS
# ============================================================

test[TARGET] = pd.to_numeric(test[TARGET], errors="coerce")

n_total = len(test)
n_missing_target = test[TARGET].isna().sum()

labelled = test.dropna(subset=[TARGET]).copy()

print("\nTest data:")
print("Total rows:", n_total)
print("Missing turbidity:", n_missing_target)
print("Labelled rows:", len(labelled))

if len(labelled) == 0:
    raise ValueError("No labelled test rows available.")

# ============================================================
# NUMERIC CONVERSION
# ============================================================

for c in FEATURES:
    train[c] = pd.to_numeric(train[c], errors="coerce")
    labelled[c] = pd.to_numeric(labelled[c], errors="coerce")

# ============================================================
# REFLECTANCE SCALE CHECK
# ============================================================

print("\nReflectance scale check:")

for c in ["B8_mean", "B11_mean", "B12_mean"]:
    print(
        f"{c}: train median={train[c].median():.6f}, "
        f"test median={labelled[c].median():.6f}, "
        f"test max={labelled[c].max():.6f}"
    )

# Training model expects reflectance approximately 0–1.
# If test reflectance is clearly Sentinel-2 integer-scaled (~0–10000),
# divide by 10000.

scale_corrected = False

reflectance_test_max = labelled[
    ["B8_mean", "B11_mean", "B12_mean"]
].max().max()

reflectance_train_max = train[
    ["B8_mean", "B11_mean", "B12_mean"]
].max().max()

if reflectance_test_max > 2 and reflectance_train_max <= 2:
    print("\nApplying /10000 reflectance correction to test data.")
    for c in ["B8_mean", "B11_mean", "B12_mean"]:
        labelled[c] = labelled[c] / 10000.0
    scale_corrected = True
else:
    print("\nNo reflectance scale correction required.")

# ============================================================
# VALID FEATURE ROWS
# ============================================================

feature_missing = labelled[FEATURES].isna().any(axis=1)

n_missing_features = feature_missing.sum()

print("\nMissing feature rows:", n_missing_features)

if n_missing_features > 0:
    print("These rows will be excluded from model metrics.")

valid = labelled.loc[~feature_missing].copy()

print("Valid labelled rows:", len(valid))

# ============================================================
# TRAIN FROZEN ETWAH MODEL
# ============================================================

X_train = train[FEATURES]
y_train = train[TARGET]

train_valid = ~(X_train.isna().any(axis=1) | y_train.isna())

X_train = X_train.loc[train_valid]
y_train = y_train.loc[train_valid]

print("\nFrozen training set:")
print("Rows:", len(X_train))
print("Features:", len(FEATURES))

model.fit(X_train, y_train)

# ============================================================
# PREDICTION
# ============================================================

X_test = valid[FEATURES]

valid["predicted_turbidity_NTU"] = model.predict(X_test)

valid["residual_observed_minus_predicted"] = (
    valid[TARGET] - valid["predicted_turbidity_NTU"]
)

valid["absolute_error_NTU"] = (
    valid["residual_observed_minus_predicted"].abs()
)

valid["squared_error"] = (
    valid["residual_observed_minus_predicted"] ** 2
)

valid["underpredicted"] = (
    valid["predicted_turbidity_NTU"] < valid[TARGET]
)

# ============================================================
# OVERALL METRICS
# ============================================================

y_true = valid[TARGET].values
y_pred = valid["predicted_turbidity_NTU"].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
bias = np.mean(y_true - y_pred)

if len(valid) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
    r, p = pearsonr(y_true, y_pred)
else:
    r, p = np.nan, np.nan

overall = pd.DataFrame([{
    "test_dataset": "Expanded Yamuna 8-station spatial test",
    "total_rows": n_total,
    "missing_turbidity": n_missing_target,
    "labelled_rows": len(labelled),
    "valid_prediction_rows": len(valid),
    "scale_corrected": scale_corrected,
    "MAE_NTU": mae,
    "RMSE_NTU": rmse,
    "R2": r2,
    "bias_observed_minus_predicted_NTU": bias,
    "Pearson_r": r,
    "Pearson_p": p,
    "observed_mean_NTU": np.mean(y_true),
    "predicted_mean_NTU": np.mean(y_pred),
    "observed_median_NTU": np.median(y_true),
    "predicted_median_NTU": np.median(y_pred),
    "observed_min_NTU": np.min(y_true),
    "observed_max_NTU": np.max(y_true),
    "predicted_min_NTU": np.min(y_pred),
    "predicted_max_NTU": np.max(y_pred),
    "underprediction_count": int(valid["underpredicted"].sum()),
    "underprediction_percent": 100 * valid["underpredicted"].mean(),
}])

print("\n" + "=" * 70)
print("OVERALL EXPANDED SPATIAL TEST")
print("=" * 70)

for c in [
    "valid_prediction_rows",
    "MAE_NTU",
    "RMSE_NTU",
    "R2",
    "bias_observed_minus_predicted_NTU",
    "Pearson_r",
    "Pearson_p",
]:
    print(f"{c}: {overall.iloc[0][c]:.6f}")

# ============================================================
# STATION PERFORMANCE
# ============================================================

station_col = None

for candidate in [
    "station",
    "Station",
    "station_name",
]:
    if candidate in valid.columns:
        station_col = candidate
        break

if station_col is None:
    raise ValueError(
        "Could not find station column. "
        "Expected 'station' or 'Station'."
    )

station_results = []

for station, g in valid.groupby(station_col):

    yt = g[TARGET].values
    yp = g["predicted_turbidity_NTU"].values

    station_results.append({
        "station": station,
        "n": len(g),
        "MAE_NTU": mean_absolute_error(yt, yp),
        "RMSE_NTU": np.sqrt(mean_squared_error(yt, yp)),
        "R2": r2_score(yt, yp) if len(g) >= 2 else np.nan,
        "bias_observed_minus_predicted_NTU": np.mean(yt - yp),
        "Pearson_r": (
            pearsonr(yt, yp)[0]
            if len(g) >= 2 and np.std(yt) > 0 and np.std(yp) > 0
            else np.nan
        ),
        "observed_mean_NTU": np.mean(yt),
        "predicted_mean_NTU": np.mean(yp),
        "observed_median_NTU": np.median(yt),
        "predicted_median_NTU": np.median(yp),
        "underprediction_percent": 100 * np.mean(yp < yt),
    })

station_df = pd.DataFrame(station_results)
station_df = station_df.sort_values("MAE_NTU")

print("\n" + "=" * 70)
print("STATION PERFORMANCE")
print("=" * 70)
print(station_df.to_string(index=False))

# ============================================================
# TURBIDITY CATEGORIES
# ============================================================

def turbidity_category(x):
    if x < 10:
        return "Low"
    elif x < 20:
        return "Moderate"
    elif x < 50:
        return "Elevated"
    elif x < 100:
        return "High"
    else:
        return "Very high"

valid["turbidity_category"] = valid[TARGET].apply(
    turbidity_category
)

category_order = [
    "Low",
    "Moderate",
    "Elevated",
    "High",
    "Very high",
]

category_results = []

for category in category_order:

    g = valid[valid["turbidity_category"] == category]

    if len(g) == 0:
        continue

    yt = g[TARGET].values
    yp = g["predicted_turbidity_NTU"].values

    category_results.append({
        "category": category,
        "n": len(g),
        "observed_mean_NTU": np.mean(yt),
        "predicted_mean_NTU": np.mean(yp),
        "MAE_NTU": mean_absolute_error(yt, yp),
        "RMSE_NTU": np.sqrt(mean_squared_error(yt, yp)),
        "bias_observed_minus_predicted_NTU": np.mean(yt - yp),
        "underprediction_percent": 100 * np.mean(yp < yt),
    })

category_df = pd.DataFrame(category_results)

print("\n" + "=" * 70)
print("TURBIDITY CATEGORY PERFORMANCE")
print("=" * 70)
print(category_df.to_string(index=False))

# ============================================================
# EXTREME EVENTS >=100 NTU
# ============================================================

extreme = valid[valid[TARGET] >= 100].copy()

extreme = extreme.sort_values(
    TARGET,
    ascending=False
)

extreme_cols = [
    c for c in [
        station_col,
        "cwc_datetime",
        "cwc_datetime_iso",
        "satellite2_id",
        "sentinel2_id",
        "satellite_date",
        "days_difference",
        "cloud_percentage",
        TARGET,
        "predicted_turbidity_NTU",
        "residual_observed_minus_predicted",
        "absolute_error_NTU",
    ]
    if c in extreme.columns
]

extreme_df = extreme[extreme_cols].copy()

print("\n" + "=" * 70)
print("EXTREME EVENTS >=100 NTU")
print("=" * 70)

if len(extreme_df) > 0:
    print(extreme_df.to_string(index=False))
else:
    print("No >=100 NTU observations.")

# ============================================================
# SENTINEL GROUP CHECK
# ============================================================

s2_col = None

for candidate in [
    "sentinel2_id",
    "satellite2_id",
    "satellite_id",
]:
    if candidate in valid.columns:
        s2_col = candidate
        break

if s2_col is not None:

    group_counts = (
        valid.groupby(s2_col)
        .size()
        .reset_index(name="n_observations")
    )

    repeated_groups = group_counts[
        group_counts["n_observations"] > 1
    ]

    print("\n" + "=" * 70)
    print("SENTINEL-2 GROUP STRUCTURE")
    print("=" * 70)

    print("Unique Sentinel-2 IDs:", group_counts.shape[0])
    print("Repeated Sentinel-2 IDs:", len(repeated_groups))
    print(
        "Rows belonging to repeated groups:",
        int(repeated_groups["n_observations"].sum())
        if len(repeated_groups) > 0 else 0
    )

else:
    print("\nSentinel-2 ID column not found; group check skipped.")

# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_file = (
    OUT_DIR
    / "expanded_spatial_frozen_model_predictions.csv"
)

valid.to_csv(
    prediction_file,
    index=False
)

# ============================================================
# SAVE METRICS
# ============================================================

overall_file = (
    OUT_DIR
    / "expanded_spatial_frozen_model_metrics.csv"
)

overall.to_csv(
    overall_file,
    index=False
)

station_file = (
    OUT_DIR
    / "expanded_spatial_station_performance.csv"
)

station_df.to_csv(
    station_file,
    index=False
)

category_file = (
    OUT_DIR
    / "expanded_spatial_category_performance.csv"
)

category_df.to_csv(
    category_file,
    index=False
)

extreme_file = (
    OUT_DIR
    / "expanded_spatial_extreme_events.csv"
)

extreme_df.to_csv(
    extreme_file,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 16 FROZEN MODEL TEST COMPLETE")
print("=" * 70)

print("\nSaved:")
print(prediction_file)
print(overall_file)
print(station_file)
print(category_file)
print(extreme_file)

print("\nFinal:")
print(f"Total test rows:       {n_total}")
print(f"Missing turbidity:     {n_missing_target}")
print(f"Labelled rows:         {len(labelled)}")
print(f"Valid predictions:     {len(valid)}")
print(f"MAE:                   {mae:.3f} NTU")
print(f"RMSE:                  {rmse:.3f} NTU")
print(f"R²:                    {r2:.3f}")
print(f"Bias:                  {bias:.3f} NTU")
print(f"Pearson r:              {r:.3f}")
print(f"Scale corrected:       {scale_corrected}")

print("\nDO NOT retrain or tune the model using these spatial-test results.")