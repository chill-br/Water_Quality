"""
Cloud Sensitivity Analysis
Etawah Yamuna Turbidity ML Project

Purpose:
    Test whether cloud contamination contributes to turbidity
    prediction error.

Method:
    - Compact_8 features
    - Gradient Boosting Regressor
    - 5-fold GroupKFold
    - Group = Sentinel-2 acquisition
    - Compare several maximum cloud-percentage thresholds

Important:
    This is a sensitivity analysis.
    It does NOT automatically justify removing cloudy observations
    from the final dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"D:\Etawah_Water_Quality")

DATA_FILE = (
    BASE_DIR
    / "data"
    / "ml_ready_turbidity.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "cloud_sensitivity_results.csv"
)


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
CLOUD_COLUMN = "cloud_percentage"


# ============================================================
# CLOUD THRESHOLDS
# ============================================================

# None = all observations
CLOUD_THRESHOLDS = [
    None,
    60.0,
    40.0,
    30.0,
    20.0,
    10.0,
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
print("ETAWAH CLOUD SENSITIVITY ANALYSIS")
print("=" * 70)

print()
print(f"Reading:")
print(DATA_FILE)

df = pd.read_csv(DATA_FILE)

print(
    f"Rows loaded: {len(df)}"
)


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = FEATURES + [
    TARGET,
    GROUP,
    CLOUD_COLUMN,
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=required_columns
).copy()

print(
    f"Rows after cleaning: {len(df)}"
)

print(
    f"Unique Sentinel groups: "
    f"{df[GROUP].nunique()}"
)


# ============================================================
# CLOUD DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("CLOUD DISTRIBUTION")
print("=" * 70)

print(
    df[CLOUD_COLUMN].describe()
)

print()

for threshold in [
    10,
    20,
    30,
    40,
    60,
]:

    count = (
        df[CLOUD_COLUMN]
        <= threshold
    ).sum()

    print(
        f"<= {threshold:>2}% cloud: "
        f"{count} observations"
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_dataset(
    data,
    threshold_label,
):

    X = data[FEATURES]

    y = data[TARGET]

    groups = data[GROUP]

    unique_groups = groups.nunique()

    if unique_groups < 5:

        print(
            f"Skipping {threshold_label}: "
            f"only {unique_groups} Sentinel groups."
        )

        return None

    gkf = GroupKFold(
        n_splits=5
    )

    predictions = np.full(
        len(data),
        np.nan
    )

    fold_results = []

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        gkf.split(
            X,
            y,
            groups=groups
        ),
        start=1
    ):

        train_groups = set(
            groups.iloc[train_idx]
        )

        test_groups = set(
            groups.iloc[test_idx]
        )

        overlap = (
            train_groups
            &
            test_groups
        )

        if overlap:

            raise RuntimeError(
                "Sentinel group leakage detected!"
            )

        model = create_model()

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        y_pred = model.predict(
            X.iloc[test_idx]
        )

        predictions[
            test_idx
        ] = y_pred

        fold_mae = (
            mean_absolute_error(
                y.iloc[test_idx],
                y_pred
            )
        )

        fold_rmse = np.sqrt(
            mean_squared_error(
                y.iloc[test_idx],
                y_pred
            )
        )

        fold_r2 = r2_score(
            y.iloc[test_idx],
            y_pred
        )

        fold_results.append({

            "fold":
                fold,

            "MAE":
                fold_mae,

            "RMSE":
                fold_rmse,

            "R2":
                fold_r2,

            "train_rows":
                len(train_idx),

            "test_rows":
                len(test_idx),

            "train_groups":
                len(train_groups),

            "test_groups":
                len(test_groups),
        })

        print(
            f"    Fold {fold}: "
            f"MAE={fold_mae:.3f} | "
            f"RMSE={fold_rmse:.3f} | "
            f"R²={fold_r2:.3f}"
        )

    # --------------------------------------------------------
    # OVERALL OOF METRICS
    # --------------------------------------------------------

    valid = np.isfinite(
        predictions
    )

    y_true = y.iloc[
        np.where(valid)[0]
    ]

    y_pred = predictions[
        valid
    ]

    overall_mae = (
        mean_absolute_error(
            y_true,
            y_pred
        )
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    overall_r2 = (
        r2_score(
            y_true,
            y_pred
        )
    )

    # --------------------------------------------------------
    # EXTREME TURBIDITY
    # --------------------------------------------------------

    extreme_mask = (
        y_true.to_numpy() >= 100
    )

    extreme_count = int(
        extreme_mask.sum()
    )

    if extreme_count > 0:

        extreme_true = (
            y_true.to_numpy()[
                extreme_mask
            ]
        )

        extreme_pred = (
            y_pred[
                extreme_mask
            ]
        )

        extreme_mae = (
            mean_absolute_error(
                extreme_true,
                extreme_pred
            )
        )

        extreme_rmse = np.sqrt(
            mean_squared_error(
                extreme_true,
                extreme_pred
            )
        )

        extreme_underprediction = (
            np.mean(
                extreme_pred
                < extreme_true
            )
            * 100
        )

    else:

        extreme_mae = np.nan
        extreme_rmse = np.nan
        extreme_underprediction = np.nan

    # --------------------------------------------------------
    # FOLD SUMMARY
    # --------------------------------------------------------

    fold_df = pd.DataFrame(
        fold_results
    )

    result = {

        "cloud_threshold_percent":
            threshold_label,

        "rows":
            len(data),

        "sentinel_groups":
            unique_groups,

        "mean_cloud_percentage":
            data[CLOUD_COLUMN].mean(),

        "median_cloud_percentage":
            data[CLOUD_COLUMN].median(),

        "turbidity_min":
            data[TARGET].min(),

        "turbidity_max":
            data[TARGET].max(),

        "overall_MAE":
            overall_mae,

        "overall_RMSE":
            overall_rmse,

        "overall_R2":
            overall_r2,

        "extreme_n":
            extreme_count,

        "extreme_MAE":
            extreme_mae,

        "extreme_RMSE":
            extreme_rmse,

        "extreme_underprediction_rate_percent":
            extreme_underprediction,

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
# RUN CLOUD THRESHOLDS
# ============================================================

results = []

for threshold in CLOUD_THRESHOLDS:

    print()
    print("=" * 70)

    if threshold is None:

        print(
            "ANALYSIS: ALL CLOUD CONDITIONS"
        )

        subset = df.copy()

        label = "ALL"

    else:

        print(
            f"ANALYSIS: cloud_percentage <= "
            f"{threshold}%"
        )

        subset = df[
            df[CLOUD_COLUMN]
            <= threshold
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


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("CLOUD SENSITIVITY SUMMARY")
print("=" * 70)

print(
    results_df[
        [
            "cloud_threshold_percent",
            "rows",
            "sentinel_groups",
            "mean_cloud_percentage",
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


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

if len(results_df) > 0:

    baseline = results_df[
        results_df[
            "cloud_threshold_percent"
        ].astype(str)
        == "ALL"
    ]

    if len(baseline) == 1:

        baseline_mae = float(
            baseline[
                "overall_MAE"
            ].iloc[0]
        )

        baseline_rmse = float(
            baseline[
                "overall_RMSE"
            ].iloc[0]
        )

        baseline_r2 = float(
            baseline[
                "overall_R2"
            ].iloc[0]
        )

        print(
            f"Baseline MAE  : "
            f"{baseline_mae:.3f}"
        )

        print(
            f"Baseline RMSE : "
            f"{baseline_rmse:.3f}"
        )

        print(
            f"Baseline R²   : "
            f"{baseline_r2:.3f}"
        )

        print()

        for _, row in results_df.iterrows():

            if str(
                row[
                    "cloud_threshold_percent"
                ]
            ) == "ALL":

                continue

            mae_change = (
                row["overall_MAE"]
                - baseline_mae
            )

            rmse_change = (
                row["overall_RMSE"]
                - baseline_rmse
            )

            r2_change = (
                row["overall_R2"]
                - baseline_r2
            )

            threshold = (
                row[
                    "cloud_threshold_percent"
                ]
            )

            print(
                f"Cloud <= {threshold}%: "
                f"MAE change={mae_change:+.3f}, "
                f"RMSE change={rmse_change:+.3f}, "
                f"R² change={r2_change:+.3f}"
            )


print()
print(
    "IMPORTANT:"
)

print(
    "Cloud filtering is being evaluated "
    "as a sensitivity analysis."
)

print(
    "Do not remove observations from the "
    "main dataset based on these results alone."
)

print()
print(
    f"Saved:"
)

print(
    OUTPUT_FILE
)

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)