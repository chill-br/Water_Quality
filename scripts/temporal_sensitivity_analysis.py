"""
Temporal Sensitivity Analysis
Etawah Yamuna Turbidity ML

Purpose:
    Test whether CWC-Sentinel temporal mismatch is contributing
    to turbidity prediction error.

Method:
    - Compact_8 spectral features
    - Gradient Boosting Regressor
    - GroupKFold by Sentinel-2 acquisition
    - Test multiple maximum CWC-Sentinel time differences
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"D:\Etawah_Water_Quality")

DATA_FILE = BASE_DIR / "data" / "ml_ready_turbidity.csv"
OUTPUT_FILE = BASE_DIR / "data" / "temporal_sensitivity_results.csv"


# ============================================================
# FEATURES
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
GROUP = "sentinel2_id"

# Maximum allowed CWC-Sentinel temporal difference
THRESHOLDS = [
    None,   # all observations
    2.0,
    1.5,
    1.0,
    0.5,
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
        random_state=42,
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ETAWAH TEMPORAL SENSITIVITY ANALYSIS")
print("=" * 70)

print(f"Reading: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print(f"Rows loaded: {len(df)}")

# Keep only required columns
required_columns = FEATURES + [
    TARGET,
    GROUP,
    "days_difference",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(
    subset=required_columns
).copy()

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=required_columns
).copy()

print(f"Rows after cleaning: {len(df)}")
print(
    f"Unique Sentinel groups: "
    f"{df[GROUP].nunique()}"
)


# ============================================================
# BASIC TEMPORAL DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("TEMPORAL DIFFERENCE DISTRIBUTION")
print("=" * 70)

print(
    df["days_difference"].describe()
)

print()
print(
    f"<= 0.5 days : "
    f"{(df['days_difference'] <= 0.5).sum()}"
)

print(
    f"<= 1.0 days : "
    f"{(df['days_difference'] <= 1.0).sum()}"
)

print(
    f"<= 1.5 days : "
    f"{(df['days_difference'] <= 1.5).sum()}"
)

print(
    f"<= 2.0 days : "
    f"{(df['days_difference'] <= 2.0).sum()}"
)


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def evaluate_dataset(data, threshold_label):

    X = data[FEATURES]
    y = data[TARGET]
    groups = data[GROUP]

    unique_groups = groups.nunique()

    # Need enough groups for 5-fold CV
    if unique_groups < 5:

        print(
            f"Skipping {threshold_label}: "
            f"only {unique_groups} groups."
        )

        return None

    n_splits = 5

    gkf = GroupKFold(
        n_splits=n_splits
    )

    predictions = np.full(
        len(data),
        np.nan
    )

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(
            X,
            y,
            groups=groups
        ),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_groups = set(
            groups.iloc[train_idx]
        )

        test_groups = set(
            groups.iloc[test_idx]
        )

        overlap = (
            train_groups &
            test_groups
        )

        if overlap:
            raise RuntimeError(
                "Sentinel group leakage detected!"
            )

        model = create_model()

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

        predictions[test_idx] = y_pred

        fold_mae = mean_absolute_error(
            y_test,
            y_pred
        )

        fold_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )

        fold_r2 = r2_score(
            y_test,
            y_pred
        )

        fold_results.append({
            "fold": fold,
            "MAE": fold_mae,
            "RMSE": fold_rmse,
            "R2": fold_r2,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
        })

        print(
            f"    Fold {fold}: "
            f"MAE={fold_mae:.3f} | "
            f"RMSE={fold_rmse:.3f} | "
            f"R²={fold_r2:.3f}"
        )

    # ========================================================
    # OVERALL OUT-OF-FOLD METRICS
    # ========================================================

    valid = np.isfinite(predictions)

    y_true = y.iloc[
        np.where(valid)[0]
    ]

    y_pred = predictions[valid]

    overall_mae = mean_absolute_error(
        y_true,
        y_pred
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    overall_r2 = r2_score(
        y_true,
        y_pred
    )

    # ========================================================
    # EXTREME TURBIDITY
    # ========================================================

    extreme_mask = (
        y_true.to_numpy() >= 100
    )

    extreme_count = int(
        extreme_mask.sum()
    )

    if extreme_count > 0:

        extreme_true = (
            y_true.to_numpy()[extreme_mask]
        )

        extreme_pred = (
            y_pred[extreme_mask]
        )

        extreme_mae = mean_absolute_error(
            extreme_true,
            extreme_pred
        )

        extreme_rmse = np.sqrt(
            mean_squared_error(
                extreme_true,
                extreme_pred
            )
        )

        extreme_underprediction_rate = np.mean(
            extreme_pred < extreme_true
        ) * 100

    else:

        extreme_mae = np.nan
        extreme_rmse = np.nan
        extreme_underprediction_rate = np.nan

    # ========================================================
    # RESULTS
    # ========================================================

    fold_df = pd.DataFrame(
        fold_results
    )

    result = {
        "temporal_threshold_days": threshold_label,
        "rows": len(data),
        "sentinel_groups": unique_groups,

        "turbidity_min": data[TARGET].min(),
        "turbidity_max": data[TARGET].max(),

        "overall_MAE": overall_mae,
        "overall_RMSE": overall_rmse,
        "overall_R2": overall_r2,

        "extreme_n": extreme_count,
        "extreme_MAE": extreme_mae,
        "extreme_RMSE": extreme_rmse,
        "extreme_underprediction_rate_percent":
            extreme_underprediction_rate,

        "fold_MAE_mean":
            fold_df["MAE"].mean(),

        "fold_MAE_std":
            fold_df["MAE"].std(),

        "fold_RMSE_mean":
            fold_df["RMSE"].mean(),

        "fold_RMSE_std":
            fold_df["RMSE"].std(),

        "fold_R2_mean":
            fold_df["R2"].mean(),

        "fold_R2_std":
            fold_df["R2"].std(),
    }

    return result


# ============================================================
# RUN ALL THRESHOLDS
# ============================================================

results = []

for threshold in THRESHOLDS:

    print()
    print("=" * 70)

    if threshold is None:

        print(
            "ANALYSIS: ALL TEMPORAL MATCHES"
        )

        subset = df.copy()
        label = "ALL"

    else:

        print(
            f"ANALYSIS: days_difference <= "
            f"{threshold} days"
        )

        subset = df[
            df["days_difference"] <= threshold
        ].copy()

        label = threshold

    print(
        f"Rows: {len(subset)}"
    )

    print(
        f"Sentinel groups: "
        f"{subset[GROUP].nunique()}"
    )

    if len(subset) < 10:
        print(
            "Too few observations. Skipping."
        )
        continue

    result = evaluate_dataset(
        subset,
        label
    )

    if result is not None:

        results.append(result)

        print()
        print(
            f"Overall MAE  : "
            f"{result['overall_MAE']:.3f}"
        )

        print(
            f"Overall RMSE : "
            f"{result['overall_RMSE']:.3f}"
        )

        print(
            f"Overall R²   : "
            f"{result['overall_R2']:.3f}"
        )

        print(
            f"Extreme n    : "
            f"{result['extreme_n']}"
        )

        if not np.isnan(
            result["extreme_MAE"]
        ):

            print(
                f"Extreme MAE  : "
                f"{result['extreme_MAE']:.3f}"
            )

            print(
                f"Extreme RMSE : "
                f"{result['extreme_RMSE']:.3f}"
            )

            print(
                f"Extreme underprediction: "
                f"{result['extreme_underprediction_rate_percent']:.1f}%"
            )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("TEMPORAL SENSITIVITY SUMMARY")
print("=" * 70)

print(
    results_df[
        [
            "temporal_threshold_days",
            "rows",
            "sentinel_groups",
            "overall_MAE",
            "overall_RMSE",
            "overall_R2",
            "extreme_n",
            "extreme_MAE",
            "extreme_RMSE",
            "extreme_underprediction_rate_percent",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    f"Saved: {OUTPUT_FILE}"
)

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)