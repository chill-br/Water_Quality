from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\Etawah_Water_Quality")
DATA_FILE = PROJECT_ROOT / "data" / "ml_ready_turbidity.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "compact_feature_cv_results.csv"

TARGET = "turbidity_NTU"
GROUP = "sentinel2_id"

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# FEATURE SETS
# ============================================================

# Existing 9-feature baseline
BASELINE_FEATURES = [
    "B2_mean",
    "B3_mean",
    "B4_mean",
    "B8_mean",
    "B11_mean",
    "B12_mean",
    "NDWI_mean",
    "MNDWI_mean",
    "water_fraction",
]


# Compact physically meaningful set.
# These are selected from the previous correlation analysis.
COMPACT_8_FEATURES = [
    "MNDWI_median",
    "NDWI_median",
    "MNDWI_mean",
    "NDWI_mean",
    "water_fraction",
    "B11_mean",
    "B12_mean",
    "B8_mean",
]


# Even smaller set for testing whether the model can remain
# competitive with fewer predictors.
COMPACT_5_FEATURES = [
    "MNDWI_median",
    "NDWI_median",
    "water_fraction",
    "B11_mean",
    "B12_mean",
]


# Very small spectral/index-only model.
COMPACT_3_FEATURES = [
    "MNDWI_median",
    "NDWI_median",
    "water_fraction",
]


FEATURE_SETS = {
    "Baseline_9": BASELINE_FEATURES,
    "Compact_8": COMPACT_8_FEATURES,
    "Compact_5": COMPACT_5_FEATURES,
    "Compact_3": COMPACT_3_FEATURES,
}


# ============================================================
# MODELS
# ============================================================

MODELS = {
    "Ridge": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        loss="huber",
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=500,
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ETAWAH COMPACT FEATURE MODEL TEST")
print("=" * 70)

print(f"Reading: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print(f"Rows loaded: {len(df)}")


# Keep only rows with a turbidity label
df = df[df[TARGET].notna()].copy()

print(f"Rows with turbidity label: {len(df)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = {TARGET, GROUP}

for features in FEATURE_SETS.values():
    required_columns.update(features)

missing_columns = [
    column for column in sorted(required_columns)
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

all_features = sorted(
    set(feature for features in FEATURE_SETS.values() for feature in features)
)

df = df.replace([np.inf, -np.inf], np.nan)

before = len(df)

df = df.dropna(
    subset=all_features + [TARGET, GROUP]
).copy()

after = len(df)

print(f"Rows after feature cleaning: {after}")

if before != after:
    print(f"Rows removed: {before - after}")


# ============================================================
# BASIC INFORMATION
# ============================================================

print(f"Unique Sentinel groups: {df[GROUP].nunique()}")
print(f"Turbidity range: {df[TARGET].min():.3f} - {df[TARGET].max():.3f} NTU")


# ============================================================
# GROUPED CROSS VALIDATION
# ============================================================

groups = df[GROUP]

n_groups = groups.nunique()

if n_groups < N_SPLITS:
    raise ValueError(
        f"Need at least {N_SPLITS} groups, but only {n_groups} are available."
    )

cv = GroupKFold(n_splits=N_SPLITS)


# ============================================================
# RESULTS STORAGE
# ============================================================

results = []


# ============================================================
# RUN MODELS
# ============================================================

for feature_set_name, features in FEATURE_SETS.items():

    print("\n" + "=" * 70)
    print(f"{feature_set_name.upper()}")
    print("=" * 70)

    print(f"Number of features: {len(features)}")
    print("Features:")
    for feature in features:
        print(f"  - {feature}")

    X = df[features]
    y = df[TARGET]

    for model_name, model in MODELS.items():

        print(f"\n--- {model_name} ---")

        fold_mae = []
        fold_rmse = []
        fold_r2 = []

        for fold_number, (train_idx, test_idx) in enumerate(
            cv.split(X, y, groups=groups),
            start=1,
        ):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # Create a fresh model for every fold
            if model_name == "Ridge":
                fold_model = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=1.0)),
                    ]
                )

            elif model_name == "Gradient Boosting":
                fold_model = GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.03,
                    max_depth=2,
                    loss="huber",
                    random_state=RANDOM_STATE,
                )

            elif model_name == "Random Forest":
                fold_model = RandomForestRegressor(
                    n_estimators=500,
                    max_features=1.0,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )

            elif model_name == "Extra Trees":
                fold_model = ExtraTreesRegressor(
                    n_estimators=500,
                    max_features=1.0,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )

            else:
                raise ValueError(f"Unknown model: {model_name}")

            fold_model.fit(X_train, y_train)

            predictions = fold_model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)

            rmse = np.sqrt(
                mean_squared_error(y_test, predictions)
            )

            r2 = r2_score(y_test, predictions)

            fold_mae.append(mae)
            fold_rmse.append(rmse)
            fold_r2.append(r2)

            print(
                f"Fold {fold_number}: "
                f"MAE={mae:.3f} | "
                f"RMSE={rmse:.3f} | "
                f"R²={r2:.3f}"
            )

        mean_mae = np.mean(fold_mae)
        std_mae = np.std(fold_mae, ddof=1)

        mean_rmse = np.mean(fold_rmse)
        std_rmse = np.std(fold_rmse, ddof=1)

        mean_r2 = np.mean(fold_r2)
        std_r2 = np.std(fold_r2, ddof=1)

        print(
            f"Mean: "
            f"MAE={mean_mae:.3f} ± {std_mae:.3f} | "
            f"RMSE={mean_rmse:.3f} ± {std_rmse:.3f} | "
            f"R²={mean_r2:.3f} ± {std_r2:.3f}"
        )

        results.append(
            {
                "feature_set": feature_set_name,
                "num_features": len(features),
                "model": model_name,
                "MAE_mean": mean_mae,
                "MAE_std": std_mae,
                "RMSE_mean": mean_rmse,
                "RMSE_std": std_rmse,
                "R2_mean": mean_r2,
                "R2_std": std_r2,
            }
        )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["RMSE_mean", "MAE_mean"],
    ascending=[True, True],
).reset_index(drop=True)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    results_df[
        [
            "feature_set",
            "model",
            "num_features",
            "MAE_mean",
            "MAE_std",
            "RMSE_mean",
            "RMSE_std",
            "R2_mean",
            "R2_std",
        ]
    ].to_string(index=False)
)


# ============================================================
# BEST MODEL BY EACH METRIC
# ============================================================

print("\n" + "=" * 70)
print("BEST RESULTS")
print("=" * 70)

best_rmse = results_df.loc[
    results_df["RMSE_mean"].idxmin()
]

best_mae = results_df.loc[
    results_df["MAE_mean"].idxmin()
]

best_r2 = results_df.loc[
    results_df["R2_mean"].idxmax()
]

print("\nBest mean RMSE:")
print(
    f"  {best_rmse['feature_set']} + "
    f"{best_rmse['model']}"
)
print(
    f"  RMSE = {best_rmse['RMSE_mean']:.3f} "
    f"± {best_rmse['RMSE_std']:.3f}"
)

print("\nBest mean MAE:")
print(
    f"  {best_mae['feature_set']} + "
    f"{best_mae['model']}"
)
print(
    f"  MAE = {best_mae['MAE_mean']:.3f} "
    f"± {best_mae['MAE_std']:.3f}"
)

print("\nBest mean R²:")
print(
    f"  {best_r2['feature_set']} + "
    f"{best_r2['model']}"
)
print(
    f"  R² = {best_r2['R2_mean']:.3f} "
    f"± {best_r2['R2_std']:.3f}"
)


# ============================================================
# COMPARE FEATURE SETS FOR EACH MODEL
# ============================================================

print("\n" + "=" * 70)
print("FEATURE SET IMPROVEMENT")
print("=" * 70)

for model_name in MODELS.keys():

    model_results = results_df[
        results_df["model"] == model_name
    ].copy()

    baseline_row = model_results[
        model_results["feature_set"] == "Baseline_9"
    ].iloc[0]

    print(f"\n{model_name}")

    for feature_set_name in [
        "Compact_8",
        "Compact_5",
        "Compact_3",
    ]:

        row = model_results[
            model_results["feature_set"] == feature_set_name
        ].iloc[0]

        rmse_change = (
            row["RMSE_mean"]
            - baseline_row["RMSE_mean"]
        )

        mae_change = (
            row["MAE_mean"]
            - baseline_row["MAE_mean"]
        )

        print(
            f"  {feature_set_name}: "
            f"RMSE {baseline_row['RMSE_mean']:.3f} "
            f"-> {row['RMSE_mean']:.3f} "
            f"({rmse_change:+.3f}) | "
            f"MAE {baseline_row['MAE_mean']:.3f} "
            f"-> {row['MAE_mean']:.3f} "
            f"({mae_change:+.3f})"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_FILE)

print("\nDONE.")