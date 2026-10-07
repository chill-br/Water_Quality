import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"D:\Etawah_Water_Quality")
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "ml_ready_turbidity.csv"

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ETAWAH TURBIDITY MODEL TRAINING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nRows loaded: {len(df)}")

# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "B2_mean",
    "B3_mean",
    "B4_mean",
    "B8_mean",
    "B11_mean",
    "B12_mean",
    "NDWI_mean",
    "MNDWI_mean",
    "water_fraction"
]

TARGET = "turbidity_NTU"

X = df[FEATURES].copy()
y = df[TARGET].copy()

# Make sure everything is numeric

X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")

valid = X.notna().all(axis=1) & y.notna()

X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

df_model = df.loc[valid].reset_index(drop=True)

print(f"Valid modelling rows: {len(X)}")

# ============================================================
# SENTINEL-GROUPED SPLIT
# ============================================================

print("\n" + "=" * 70)
print("SENTINEL-GROUPED TRAIN / TEST SPLIT")
print("=" * 70)

sentinel_ids = (
    df_model["sentinel2_id"]
    .astype(str)
    .unique()
)

# Convert to normal NumPy array
sentinel_ids = np.array(sentinel_ids)

rng = np.random.default_rng(42)
rng.shuffle(sentinel_ids)

n_train = int(len(sentinel_ids) * 0.80)

train_sentinels = set(sentinel_ids[:n_train])
test_sentinels = set(sentinel_ids[n_train:])

train_mask = df_model["sentinel2_id"].isin(train_sentinels)
test_mask = df_model["sentinel2_id"].isin(test_sentinels)

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

print(f"Unique Sentinel groups: {len(sentinel_ids)}")
print(f"Training groups:        {len(train_sentinels)}")
print(f"Testing groups:         {len(test_sentinels)}")

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")

overlap = train_sentinels.intersection(test_sentinels)

print(f"\nSentinel leakage: {len(overlap)}")

if len(overlap) != 0:
    raise RuntimeError("Sentinel leakage detected!")

print("✓ No Sentinel leakage")

# ============================================================
# MODEL EVALUATION FUNCTION
# ============================================================

results = []


def evaluate_model(name, model, Xtr, ytr, Xte, yte):

    model.fit(Xtr, ytr)

    pred = model.predict(Xte)

    mae = mean_absolute_error(yte, pred)

    rmse = np.sqrt(
        mean_squared_error(yte, pred)
    )

    r2 = r2_score(yte, pred)

    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print(f"MAE :  {mae:.3f} NTU")
    print(f"RMSE:  {rmse:.3f} NTU")
    print(f"R²  :  {r2:.3f}")

    return model, pred


# ============================================================
# MODEL 1 — RANDOM FOREST
# ============================================================

rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model, rf_pred = evaluate_model(
    "Random Forest",
    rf,
    X_train,
    y_train,
    X_test,
    y_test
)

# ============================================================
# MODEL 2 — EXTRA TREES
# ============================================================

et = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

et_model, et_pred = evaluate_model(
    "Extra Trees",
    et,
    X_train,
    y_train,
    X_test,
    y_test
)

# ============================================================
# MODEL 3 — GRADIENT BOOSTING
# ============================================================

gb = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=2,
    min_samples_leaf=3,
    loss="huber",
    random_state=42
)

gb_model, gb_pred = evaluate_model(
    "Gradient Boosting",
    gb,
    X_train,
    y_train,
    X_test,
    y_test
)

# ============================================================
# MODEL 4 — RIDGE REGRESSION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=10.0)

ridge_model, ridge_pred = evaluate_model(
    "Ridge Regression",
    ridge,
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test
)

# ============================================================
# LOG-TARGET RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("LOG-TARGET RANDOM FOREST")
print("=" * 70)

y_train_log = np.log1p(y_train)

rf_log = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_log.fit(X_train, y_train_log)

log_pred = rf_log.predict(X_test)

pred_log_rf = np.expm1(log_pred)

mae = mean_absolute_error(
    y_test,
    pred_log_rf
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred_log_rf
    )
)

r2 = r2_score(
    y_test,
    pred_log_rf
)

results.append({
    "model": "Random Forest - log1p target",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
})

print(f"MAE :  {mae:.3f} NTU")
print(f"RMSE:  {rmse:.3f} NTU")
print(f"R²  :  {r2:.3f}")

# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "RMSE"
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

# ============================================================
# SAVE RESULTS
# ============================================================

results_file = DATA_DIR / "model_comparison.csv"

results_df.to_csv(
    results_file,
    index=False
)

importance_file = DATA_DIR / "random_forest_feature_importance.csv"

importance.to_csv(
    importance_file,
    index=False
)

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_df = df_model.loc[test_mask, [
    "patch_number",
    "sentinel2_id",
    "satellite_date",
    "turbidity_NTU"
]].copy()

prediction_df["RF_prediction"] = rf_pred
prediction_df["ExtraTrees_prediction"] = et_pred
prediction_df["GradientBoosting_prediction"] = gb_pred
prediction_df["Ridge_prediction"] = ridge_pred
prediction_df["RF_log_target_prediction"] = pred_log_rf

prediction_file = DATA_DIR / "test_set_predictions.csv"

prediction_df.to_csv(
    prediction_file,
    index=False
)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(results_file)
print(importance_file)
print(prediction_file)

print("\nDONE.")