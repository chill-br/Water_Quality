"""
FINAL DIAGNOSTICS
Etawah Yamuna Water Quality ML Project

Primary model:
    Compact_8 features
    GradientBoostingRegressor
    5-fold GroupKFold
    Group = Sentinel-2 acquisition

Purpose:
    Generate final publication-ready diagnostic plots and tables
    from the existing grouped OOF predictions.

Outputs:
    data/final_model_metrics.csv
    data/final_prediction_summary.csv
    data/final_category_performance.csv
    data/final_extreme_events.csv
    data/final_feature_correlations.csv

    data/final_figures/
        01_observed_vs_predicted.png
        02_residuals_vs_predicted.png
        03_turbidity_distribution.png
        04_observed_vs_predicted_by_category.png
        05_feature_correlations.png
        06_prediction_error_distribution.png
        07_extreme_event_predictions.png
"""


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "final_figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_FILE = (
    BASE_DIR
    / "data"
    / "final_model_metrics.csv"
)

SUMMARY_FILE = (
    BASE_DIR
    / "data"
    / "final_prediction_summary.csv"
)

CATEGORY_FILE = (
    BASE_DIR
    / "data"
    / "final_category_performance.csv"
)

EXTREME_FILE = (
    BASE_DIR
    / "data"
    / "final_extreme_events.csv"
)

CORRELATION_FILE = (
    BASE_DIR
    / "data"
    / "final_feature_correlations.csv"
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
print("FINAL MODEL DIAGNOSTICS")
print("=" * 70)

print()
print("Reading:")
print(DATA_FILE)

df = pd.read_csv(
    DATA_FILE
)

required = FEATURES + [
    TARGET,
    GROUP,
]

missing = [
    col
    for col in required
    if col not in df.columns
]

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=required
).copy()

print(
    f"Rows: {len(df)}"
)

print(
    f"Sentinel groups: "
    f"{df[GROUP].nunique()}"
)


# ============================================================
# GROUPED OOF PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("GENERATING GROUPED OOF PREDICTIONS")
print("=" * 70)

X = df[FEATURES]
y = df[TARGET]
groups = df[GROUP]

gkf = GroupKFold(
    n_splits=5
)

oof_predictions = np.full(
    len(df),
    np.nan
)

fold_records = []

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
            "Sentinel-2 group leakage detected!"
        )

    model = create_model()

    model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx]
    )

    predictions = model.predict(
        X.iloc[test_idx]
    )

    oof_predictions[
        test_idx
    ] = predictions

    fold_mae = mean_absolute_error(
        y.iloc[test_idx],
        predictions
    )

    fold_rmse = np.sqrt(
        mean_squared_error(
            y.iloc[test_idx],
            predictions
        )
    )

    fold_r2 = r2_score(
        y.iloc[test_idx],
        predictions
    )

    fold_records.append({

        "fold": fold,

        "rows":
            len(test_idx),

        "groups":
            len(test_groups),

        "MAE":
            fold_mae,

        "RMSE":
            fold_rmse,

        "R2":
            fold_r2,
    })

    print(
        f"Fold {fold}: "
        f"MAE={fold_mae:.3f} | "
        f"RMSE={fold_rmse:.3f} | "
        f"R²={fold_r2:.3f}"
    )


# ============================================================
# ADD PREDICTIONS
# ============================================================

df["predicted_turbidity_NTU"] = (
    oof_predictions
)

df["residual_NTU"] = (
    df[TARGET]
    - df["predicted_turbidity_NTU"]
)

df["absolute_error_NTU"] = (
    df["residual_NTU"]
    .abs()
)


# ============================================================
# OVERALL METRICS
# ============================================================

valid = np.isfinite(
    df["predicted_turbidity_NTU"]
)

y_true = df.loc[
    valid,
    TARGET
]

y_pred = df.loc[
    valid,
    "predicted_turbidity_NTU"
]

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

fold_df = pd.DataFrame(
    fold_records
)


# ============================================================
# METRICS TABLE
# ============================================================

metrics_rows = []

for _, row in fold_df.iterrows():

    metrics_rows.append({

        "model":
            "Gradient Boosting",

        "validation":
            "GroupKFold",

        "fold":
            int(row["fold"]),

        "rows":
            int(row["rows"]),

        "groups":
            int(row["groups"]),

        "MAE":
            row["MAE"],

        "RMSE":
            row["RMSE"],

        "R2":
            row["R2"],
    })


metrics_rows.append({

    "model":
        "Gradient Boosting",

    "validation":
        "Pooled OOF",

    "fold":
        "ALL",

    "rows":
        len(df),

    "groups":
        df[GROUP].nunique(),

    "MAE":
        overall_mae,

    "RMSE":
        overall_rmse,

    "R2":
        overall_r2,
})


metrics_rows.append({

    "model":
        "Gradient Boosting",

    "validation":
        "Mean ± SD",

    "fold":
        "ALL",

    "rows":
        len(df),

    "groups":
        df[GROUP].nunique(),

    "MAE":
        fold_df["MAE"].mean(),

    "RMSE":
        fold_df["RMSE"].mean(),

    "R2":
        fold_df["R2"].mean(),
})


metrics_df = pd.DataFrame(
    metrics_rows
)

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


# ============================================================
# PRINT PERFORMANCE
# ============================================================

print()
print("=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)

print(
    f"Pooled OOF MAE  : "
    f"{overall_mae:.3f} NTU"
)

print(
    f"Pooled OOF RMSE : "
    f"{overall_rmse:.3f} NTU"
)

print(
    f"Pooled OOF R²   : "
    f"{overall_r2:.3f}"
)

print()

print(
    f"Fold mean MAE   : "
    f"{fold_df['MAE'].mean():.3f} "
    f"+/- {fold_df['MAE'].std():.3f}"
)

print(
    f"Fold mean RMSE  : "
    f"{fold_df['RMSE'].mean():.3f} "
    f"+/- {fold_df['RMSE'].std():.3f}"
)

print(
    f"Fold mean R²    : "
    f"{fold_df['R2'].mean():.3f} "
    f"+/- {fold_df['R2'].std():.3f}"
)


# ============================================================
# PREDICTION SUMMARY
# ============================================================

summary = pd.DataFrame({

    "metric": [

        "n_observations",
        "n_sentinel_groups",
        "observed_min_NTU",
        "observed_median_NTU",
        "observed_mean_NTU",
        "observed_max_NTU",
        "predicted_min_NTU",
        "predicted_median_NTU",
        "predicted_mean_NTU",
        "predicted_max_NTU",
        "MAE_NTU",
        "RMSE_NTU",
        "R2",
    ],

    "value": [

        len(df),

        df[GROUP].nunique(),

        y_true.min(),

        y_true.median(),

        y_true.mean(),

        y_true.max(),

        y_pred.min(),

        y_pred.median(),

        y_pred.mean(),

        y_pred.max(),

        overall_mae,

        overall_rmse,

        overall_r2,
    ]
})

summary.to_csv(
    SUMMARY_FILE,
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
        return "Very high (≥100)"


df["turbidity_category"] = (
    df[TARGET]
    .apply(turbidity_category)
)


# ============================================================
# CATEGORY PERFORMANCE
# ============================================================

category_records = []

category_order = [
    "Low (<10)",
    "Moderate (10–20)",
    "Elevated (20–50)",
    "High (50–100)",
    "Very high (≥100)",
]

for category in category_order:

    subset = df[
        df["turbidity_category"]
        == category
    ]

    if len(subset) == 0:
        continue

    observed = subset[TARGET]
    predicted = subset[
        "predicted_turbidity_NTU"
    ]

    category_records.append({

        "category":
            category,

        "n":
            len(subset),

        "observed_mean":
            observed.mean(),

        "predicted_mean":
            predicted.mean(),

        "MAE":
            mean_absolute_error(
                observed,
                predicted
            ),

        "RMSE":
            np.sqrt(
                mean_squared_error(
                    observed,
                    predicted
                )
            ),

        "mean_residual":
            subset[
                "residual_NTU"
            ].mean(),

        "median_absolute_error":
            subset[
                "absolute_error_NTU"
            ].median(),

        "underprediction_percent":
            (
                predicted < observed
            ).mean()
            * 100,
    })


category_df = pd.DataFrame(
    category_records
)

category_df.to_csv(
    CATEGORY_FILE,
    index=False
)


# ============================================================
# EXTREME EVENTS
# ============================================================

extreme = df[
    df[TARGET] >= 100
].copy()

extreme = extreme.sort_values(
    TARGET,
    ascending=False
)

extreme_output_columns = [
    "patch_number",
    "sentinel2_id",
    "satellite_date",
    "cwc_datetime",
    "turbidity_NTU",
    "predicted_turbidity_NTU",
    "residual_NTU",
    "absolute_error_NTU",
]

available_extreme_columns = [
    col
    for col in extreme_output_columns
    if col in extreme.columns
]

extreme[
    available_extreme_columns
].to_csv(
    EXTREME_FILE,
    index=False
)


# ============================================================
# FEATURE CORRELATIONS
# ============================================================

correlation_records = []

for feature in FEATURES:

    correlation = (
        df[
            [
                feature,
                TARGET
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    correlation_records.append({

        "feature":
            feature,

        "pearson_correlation_with_turbidity":
            correlation,
    })


correlation_df = pd.DataFrame(
    correlation_records
)

correlation_df[
    "absolute_correlation"
] = (
    correlation_df[
        "pearson_correlation_with_turbidity"
    ]
    .abs()
)

correlation_df = (
    correlation_df
    .sort_values(
        "absolute_correlation",
        ascending=False
    )
)

correlation_df.to_csv(
    CORRELATION_FILE,
    index=False
)


# ============================================================
# FIGURE 1
# OBSERVED VS PREDICTED
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.scatter(
    y_true,
    y_pred,
    alpha=0.70,
    edgecolor="none"
)

line_min = min(
    y_true.min(),
    y_pred.min()
)

line_max = max(
    y_true.max(),
    y_pred.max()
)

plt.plot(
    [line_min, line_max],
    [line_min, line_max],
    linestyle="--",
    linewidth=2
)

plt.xlabel(
    "Observed turbidity (NTU)"
)

plt.ylabel(
    "Predicted turbidity (NTU)"
)

plt.title(
    "Observed vs Predicted Turbidity\n"
    "Gradient Boosting — Grouped OOF"
)

plt.text(
    0.05,
    0.95,
    (
        f"MAE = {overall_mae:.2f} NTU\n"
        f"RMSE = {overall_rmse:.2f} NTU\n"
        f"R² = {overall_r2:.3f}"
    ),
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.8
    )
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "01_observed_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2
# RESIDUALS VS PREDICTED
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.scatter(
    y_pred,
    df.loc[
        valid,
        "residual_NTU"
    ],
    alpha=0.70,
    edgecolor="none"
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=2
)

plt.xlabel(
    "Predicted turbidity (NTU)"
)

plt.ylabel(
    "Residual (Observed − Predicted) (NTU)"
)

plt.title(
    "Residuals vs Predicted Turbidity"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "02_residuals_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3
# TURBIDITY DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    y_true,
    bins=30,
    edgecolor="black",
    alpha=0.75
)

plt.xlabel(
    "Observed turbidity (NTU)"
)

plt.ylabel(
    "Number of observations"
)

plt.title(
    "Distribution of Observed Turbidity"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "03_turbidity_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 4
# OBSERVED VS PREDICTED BY CATEGORY
# ============================================================

plt.figure(
    figsize=(10, 6)
)

positions = np.arange(
    len(category_order)
)

observed_means = []
predicted_means = []
labels = []

for category in category_order:

    subset = df[
        df["turbidity_category"]
        == category
    ]

    if len(subset) == 0:
        continue

    labels.append(category)

    observed_means.append(
        subset[TARGET].mean()
    )

    predicted_means.append(
        subset[
            "predicted_turbidity_NTU"
        ].mean()
    )

x = np.arange(
    len(labels)
)

width = 0.36

plt.bar(
    x - width / 2,
    observed_means,
    width,
    label="Observed"
)

plt.bar(
    x + width / 2,
    predicted_means,
    width,
    label="Predicted"
)

plt.xticks(
    x,
    labels,
    rotation=20,
    ha="right"
)

plt.ylabel(
    "Mean turbidity (NTU)"
)

plt.title(
    "Observed and Predicted Turbidity by Category"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "04_observed_vs_predicted_by_category.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 5
# FEATURE CORRELATIONS
# ============================================================

plot_corr = correlation_df.copy()

plot_corr = plot_corr.sort_values(
    "pearson_correlation_with_turbidity"
)

plt.figure(
    figsize=(9, 6)
)

plt.barh(
    plot_corr["feature"],
    plot_corr[
        "pearson_correlation_with_turbidity"
    ]
)

plt.axvline(
    0,
    linestyle="-",
    linewidth=1
)

plt.xlabel(
    "Pearson correlation with turbidity"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Spectral Feature Relationships with Turbidity"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "05_feature_correlations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 6
# PREDICTION ERROR DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.hist(
    df["residual_NTU"],
    bins=30,
    edgecolor="black",
    alpha=0.75
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=2
)

plt.xlabel(
    "Residual (Observed − Predicted) (NTU)"
)

plt.ylabel(
    "Number of observations"
)

plt.title(
    "Distribution of Prediction Residuals"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "06_prediction_error_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 7
# EXTREME EVENTS
# ============================================================

if len(extreme) > 0:

    extreme_plot = extreme.copy()

    extreme_plot = extreme_plot.sort_values(
        TARGET,
        ascending=True
    )

    labels = []

    for _, row in extreme_plot.iterrows():

        if "patch_number" in row:

            labels.append(
                f"Patch {int(row['patch_number'])}"
            )

        else:

            labels.append("Observation")

    y_pos = np.arange(
        len(extreme_plot)
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        y_pos - 0.18,
        extreme_plot[TARGET],
        height=0.35,
        label="Observed"
    )

    plt.barh(
        y_pos + 0.18,
        extreme_plot[
            "predicted_turbidity_NTU"
        ],
        height=0.35,
        label="Predicted"
    )

    plt.yticks(
        y_pos,
        labels
    )

    plt.xlabel(
        "Turbidity (NTU)"
    )

    plt.ylabel(
        "Extreme observations"
    )

    plt.title(
        "Observed vs Predicted Extreme Turbidity Events"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "07_extreme_event_predictions.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("CATEGORY PERFORMANCE")
print("=" * 70)

print(
    category_df.to_string(
        index=False
    )
)

print()
print("=" * 70)
print("EXTREME EVENTS")
print("=" * 70)

print(
    extreme[
        [
            col
            for col in [
                "patch_number",
                "turbidity_NTU",
                "predicted_turbidity_NTU",
                "residual_NTU",
                "absolute_error_NTU",
            ]
            if col in extreme.columns
        ]
    ].to_string(
        index=False
    )
)

print()
print("=" * 70)
print("FEATURE CORRELATIONS")
print("=" * 70)

print(
    correlation_df[
        [
            "feature",
            "pearson_correlation_with_turbidity"
        ]
    ].to_string(
        index=False
    )
)

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print()
print(METRICS_FILE)
print(SUMMARY_FILE)
print(CATEGORY_FILE)
print(EXTREME_FILE)
print(CORRELATION_FILE)

print()

for file in sorted(
    OUTPUT_DIR.glob("*.png")
):

    print(file)

print()
print("=" * 70)
print("FINAL DIAGNOSTICS COMPLETE")
print("=" * 70)