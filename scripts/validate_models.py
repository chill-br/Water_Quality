from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

INPUT_FILE = DATA_DIR / "ml_ready_turbidity.csv"

OUTPUT_CV = DATA_DIR / "grouped_cross_validation_results.csv"
OUTPUT_DUPLICATES = DATA_DIR / "duplicate_sentinel_analysis.csv"
OUTPUT_PREDICTIONS = DATA_DIR / "grouped_cv_predictions.csv"

PLOT_OBSERVED = DATA_DIR / "observed_vs_predicted_grouped_cv.png"
PLOT_RESIDUALS = DATA_DIR / "residuals_grouped_cv.png"


FEATURES = [
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

TARGET = "turbidity_NTU"
GROUP = "sentinel2_id"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ETAWAH GROUPED CROSS-VALIDATION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")

required = FEATURES + [TARGET, GROUP]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required).copy()

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()

print(f"Valid rows: {len(df)}")
print(f"Unique Sentinel groups: {df[GROUP].nunique()}")

X = df[FEATURES].astype(float)
y = df[TARGET].astype(float)

groups = df[GROUP].astype(str).to_numpy()


# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {
    "Ridge": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        max_features=1.0,
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        max_features=1.0,
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        random_state=42,
        loss="huber",
    ),
}


# ============================================================
# GROUPED CROSS-VALIDATION
# ============================================================

N_SPLITS = 5

gkf = GroupKFold(n_splits=N_SPLITS)

all_results = []
all_predictions = []

print()
print("=" * 70)
print(f"{N_SPLITS}-FOLD GROUPED CROSS-VALIDATION")
print("=" * 70)

for model_name, model in models.items():

    print()
    print("-" * 60)
    print(model_name)
    print("-" * 60)

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=groups), start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        leakage = train_groups.intersection(test_groups)

        if leakage:
            raise RuntimeError(
                f"Sentinel leakage detected in fold {fold}: {leakage}"
            )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        fold_metrics.append(
            {
                "model": model_name,
                "fold": fold,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_groups": len(train_groups),
                "test_groups": len(test_groups),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

        print(
            f"Fold {fold}: "
            f"MAE={mae:.3f} | "
            f"RMSE={rmse:.3f} | "
            f"R²={r2:.3f}"
        )

        for idx, prediction in zip(test_idx, predictions):

            all_predictions.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "row_index": idx,
                    "sentinel2_id": groups[idx],
                    "observed_turbidity": y.iloc[idx],
                    "predicted_turbidity": prediction,
                    "residual": y.iloc[idx] - prediction,
                }
            )

    all_results.extend(fold_metrics)

    fold_df = pd.DataFrame(fold_metrics)

    print()
    print(
        f"Mean: "
        f"MAE={fold_df['MAE'].mean():.3f} ± {fold_df['MAE'].std():.3f} | "
        f"RMSE={fold_df['RMSE'].mean():.3f} ± {fold_df['RMSE'].std():.3f} | "
        f"R²={fold_df['R2'].mean():.3f} ± {fold_df['R2'].std():.3f}"
    )


# ============================================================
# SAVE CROSS-VALIDATION RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

summary = (
    results_df
    .groupby("model")
    .agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
    )
    .reset_index()
    .sort_values("MAE_mean")
)

print()
print("=" * 70)
print("GROUPED CROSS-VALIDATION SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))

summary.to_csv(OUTPUT_CV, index=False)


# ============================================================
# DUPLICATE SENTINEL ACQUISITION ANALYSIS
# ============================================================

print()
print("=" * 70)
print("DUPLICATE SENTINEL ACQUISITION ANALYSIS")
print("=" * 70)

group_counts = (
    df.groupby(GROUP)
    .size()
    .reset_index(name="row_count")
)

duplicate_groups = group_counts[group_counts["row_count"] > 1].copy()

print(f"Total Sentinel groups: {len(group_counts)}")
print(f"Groups with multiple rows: {len(duplicate_groups)}")

duplicate_records = []

for sentinel_id, group_df in df.groupby(GROUP):

    if len(group_df) <= 1:
        continue

    feature_variation = group_df[FEATURES].nunique()

    turbidity_values = group_df[TARGET].unique()

    duplicate_records.append(
        {
            "sentinel2_id": sentinel_id,
            "row_count": len(group_df),
            "unique_turbidity_values": len(turbidity_values),
            "turbidity_values": ",".join(
                [str(v) for v in sorted(turbidity_values)]
            ),
            "features_with_variation": int(
                (feature_variation > 1).sum()
            ),
            "all_features_identical": bool(
                (feature_variation <= 1).all()
            ),
        }
    )

duplicate_df = pd.DataFrame(duplicate_records)

if len(duplicate_df) > 0:

    print()
    print(
        "Sentinel groups where all spectral features are identical:"
    )

    identical_count = duplicate_df[
        duplicate_df["all_features_identical"]
    ].shape[0]

    print(f"{identical_count} / {len(duplicate_df)}")

    print()
    print("First duplicate groups:")
    print(
        duplicate_df.head(20).to_string(index=False)
    )

duplicate_df.to_csv(OUTPUT_DUPLICATES, index=False)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame(all_predictions)

predictions_df.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)


# ============================================================
# OBSERVED VS PREDICTED PLOT
# ============================================================

best_model = summary.iloc[0]["model"]

best_predictions = predictions_df[
    predictions_df["model"] == best_model
].copy()

observed = best_predictions["observed_turbidity"]
predicted = best_predictions["predicted_turbidity"]

plt.figure(figsize=(8, 7))

plt.scatter(
    observed,
    predicted,
    alpha=0.7
)

minimum = min(observed.min(), predicted.min())
maximum = max(observed.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Observed Turbidity (NTU)")
plt.ylabel("Predicted Turbidity (NTU)")
plt.title(
    f"Observed vs Predicted Turbidity\n"
    f"{best_model} — Grouped 5-Fold CV"
)

plt.tight_layout()

plt.savefig(
    PLOT_OBSERVED,
    dpi=300
)

plt.close()


# ============================================================
# RESIDUAL PLOT
# ============================================================

residuals = best_predictions["residual"]

plt.figure(figsize=(8, 7))

plt.scatter(
    observed,
    residuals,
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Observed Turbidity (NTU)")
plt.ylabel("Residual (Observed - Predicted)")
plt.title(
    f"Residual Plot\n"
    f"{best_model} — Grouped 5-Fold CV"
)

plt.tight_layout()

plt.savefig(
    PLOT_RESIDUALS,
    dpi=300
)

plt.close()


# ============================================================
# EXTREME TURBIDITY CHECK
# ============================================================

print()
print("=" * 70)
print("EXTREME TURBIDITY PREDICTION CHECK")
print("=" * 70)

extreme = best_predictions[
    best_predictions["observed_turbidity"] >= 100
].copy()

if len(extreme) == 0:

    print("No observations >= 100 NTU.")

else:

    extreme["absolute_error"] = (
        extreme["observed_turbidity"]
        - extreme["predicted_turbidity"]
    ).abs()

    print(
        extreme[
            [
                "sentinel2_id",
                "observed_turbidity",
                "predicted_turbidity",
                "residual",
                "absolute_error",
            ]
        ]
        .sort_values(
            "observed_turbidity",
            ascending=False
        )
        .to_string(index=False)
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(OUTPUT_CV)
print(OUTPUT_DUPLICATES)
print(OUTPUT_PREDICTIONS)
print(PLOT_OBSERVED)
print(PLOT_RESIDUALS)

print()
print("DONE.")