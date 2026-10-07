
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import pearsonr


# ============================================================
# STEP 20
# FINAL PUBLICATION-QUALITY SPATIAL VALIDATION
#
# IMPORTANT:
# - Uses the frozen Step 18 predictions only
# - No model retraining
# - No test observations added to training
# - No prediction modification
# - Diagnostic / reporting stage only
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    r"D:\Etawah_Water_Quality\Yamuna_Independent_Test"
)

RESULTS = BASE / "results"

PRED_FILE = (
    RESULTS /
    "expanded_spatial_frozen_model_predictions.csv"
)

FEATURE_FILE = (
    BASE /
    "data" /
    "Yamuna_Expanded_Compact8_Features_820.csv"
)

OUTPUT_DIR = RESULTS / "final_spatial_figures"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GLOBAL FIGURE SETTINGS
# ============================================================

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_pearson(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = (
        np.isfinite(y_true) &
        np.isfinite(y_pred)
    )

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if (
        len(y_true) < 2 or
        np.unique(y_true).size < 2 or
        np.unique(y_pred).size < 2
    ):
        return np.nan, np.nan

    return pearsonr(y_true, y_pred)


def calculate_metrics(df):

    df = df.dropna(
        subset=[
            "Turbidity (NTU)",
            "predicted_turbidity_NTU",
        ]
    )

    if len(df) == 0:
        return {
            "n": 0,
            "MAE_NTU": np.nan,
            "RMSE_NTU": np.nan,
            "R2": np.nan,
            "bias_observed_minus_predicted_NTU": np.nan,
            "Pearson_r": np.nan,
            "Pearson_p": np.nan,
            "observed_mean_NTU": np.nan,
            "predicted_mean_NTU": np.nan,
            "observed_median_NTU": np.nan,
            "predicted_median_NTU": np.nan,
            "observed_min_NTU": np.nan,
            "observed_max_NTU": np.nan,
            "underpredicted_percent": np.nan,
        }

    y = df["Turbidity (NTU)"].astype(float)
    p = df["predicted_turbidity_NTU"].astype(float)

    r, pval = safe_pearson(y, p)

    return {
        "n": len(df),
        "MAE_NTU": mean_absolute_error(y, p),
        "RMSE_NTU": np.sqrt(
            mean_squared_error(y, p)
        ),
        "R2": (
            r2_score(y, p)
            if y.nunique() > 1
            else np.nan
        ),
        "bias_observed_minus_predicted_NTU":
            (y - p).mean(),
        "Pearson_r": r,
        "Pearson_p": pval,
        "observed_mean_NTU": y.mean(),
        "predicted_mean_NTU": p.mean(),
        "observed_median_NTU": y.median(),
        "predicted_median_NTU": p.median(),
        "observed_min_NTU": y.min(),
        "observed_max_NTU": y.max(),
        "underpredicted_percent":
            (p < y).mean() * 100,
    }


def save_figure(fig, filename):

    path = OUTPUT_DIR / filename

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", path)


def add_1to1_line(ax, xmin, xmax):

    ax.plot(
        [xmin, xmax],
        [xmin, xmax],
        linestyle="--",
        linewidth=1.2,
        label="1:1 line"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 20 — FINAL SPATIAL VALIDATION FIGURES")
print("=" * 70)

print("\nLoading frozen spatial predictions...")

df = pd.read_csv(PRED_FILE)

print("Prediction rows:", len(df))


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "Station",
    "Data Acquisition Time",
    "Latitude",
    "Longitude",
    "Turbidity (NTU)",
    "predicted_turbidity_NTU",
    "residual_observed_minus_predicted_NTU",
    "absolute_error_NTU",
    "cloud_percentage",
    "days_difference",
    "sentinel2_id",
    "B8_mean",
    "B11_mean",
    "B12_mean",
    "MNDWI_mean",
    "MNDWI_median",
    "NDWI_mean",
    "NDWI_median",
    "water_fraction",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# NUMERIC CLEANING
# ============================================================

numeric_cols = [
    "Latitude",
    "Longitude",
    "Turbidity (NTU)",
    "predicted_turbidity_NTU",
    "residual_observed_minus_predicted_NTU",
    "absolute_error_NTU",
    "cloud_percentage",
    "days_difference",
    "B8_mean",
    "B11_mean",
    "B12_mean",
    "MNDWI_mean",
    "MNDWI_median",
    "NDWI_mean",
    "NDWI_median",
    "water_fraction",
]

for col in numeric_cols:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# Recalculate these reporting fields only as a consistency check.
# Predictions themselves are NOT changed.

df["diagnostic_residual"] = (
    df["Turbidity (NTU)"]
    - df["predicted_turbidity_NTU"]
)

df["diagnostic_absolute_error"] = (
    df["diagnostic_residual"].abs()
)

residual_difference = (
    df["diagnostic_residual"]
    - df["residual_observed_minus_predicted_NTU"]
).abs().max()

error_difference = (
    df["diagnostic_absolute_error"]
    - df["absolute_error_NTU"]
).abs().max()

print(
    "\nMaximum stored-vs-recalculated residual difference:",
    residual_difference
)

print(
    "Maximum stored-vs-recalculated absolute-error difference:",
    error_difference
)

if residual_difference > 1e-8:
    raise ValueError(
        "Stored residuals do not match recalculated residuals."
    )

if error_difference > 1e-8:
    raise ValueError(
        "Stored absolute errors do not match recalculated errors."
    )


# ============================================================
# LABELLED DATA
# ============================================================

df = df.dropna(
    subset=[
        "Turbidity (NTU)",
        "predicted_turbidity_NTU",
    ]
).copy()

print(
    "\nFinal labelled spatial test rows:",
    len(df)
)


# ============================================================
# ADD REPORTING CATEGORIES
# ============================================================

def turbidity_category(x):

    if x < 10:
        return "Low (<10)"

    if x < 20:
        return "Moderate (10–<20)"

    if x < 50:
        return "Elevated (20–<50)"

    if x < 100:
        return "High (50–<100)"

    return "Very high (≥100)"


df["turbidity_category"] = (
    df["Turbidity (NTU)"]
    .apply(turbidity_category)
)


# ============================================================
# 1. OVERALL SUMMARY TABLE
# ============================================================

print("\n" + "=" * 70)
print("1. OVERALL SUMMARY")
print("=" * 70)

overall_metrics = calculate_metrics(df)

overall_df = pd.DataFrame([
    {
        "validation": "Independent spatial test",
        **overall_metrics,
    }
])

print(
    overall_df.round(4).to_string(index=False)
)

overall_df.to_csv(
    RESULTS /
    "final_spatial_validation_overall_summary.csv",
    index=False
)


# ============================================================
# 2. STATION PERFORMANCE TABLE
# ============================================================

print("\n" + "=" * 70)
print("2. STATION PERFORMANCE")
print("=" * 70)

station_rows = []

for station, group in df.groupby(
    "Station",
    sort=True
):

    metrics = calculate_metrics(group)

    station_rows.append({
        "Station": station,
        **metrics,
        "cloud_mean_percent":
            group["cloud_percentage"].mean(),
        "days_difference_mean":
            group["days_difference"].mean(),
        "percent_turbidity_ge_100":
            (
                group["Turbidity (NTU)"] >= 100
            ).mean() * 100,
        "percent_turbidity_ge_500":
            (
                group["Turbidity (NTU)"] >= 500
            ).mean() * 100,
    })

station_df = pd.DataFrame(
    station_rows
)

station_df = station_df.sort_values(
    "MAE_NTU"
)

print(
    station_df.round(3).to_string(index=False)
)

station_df.to_csv(
    RESULTS /
    "final_spatial_station_performance.csv",
    index=False
)


# ============================================================
# 3. TURBIDITY CATEGORY PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("3. TURBIDITY CATEGORY PERFORMANCE")
print("=" * 70)

category_order = [
    "Low (<10)",
    "Moderate (10–<20)",
    "Elevated (20–<50)",
    "High (50–<100)",
    "Very high (≥100)",
]

category_rows = []

for category in category_order:

    group = df[
        df["turbidity_category"] == category
    ]

    metrics = calculate_metrics(group)

    category_rows.append({
        "turbidity_category": category,
        **metrics,
    })

category_df = pd.DataFrame(
    category_rows
)

print(
    category_df.round(3).to_string(index=False)
)

category_df.to_csv(
    RESULTS /
    "final_spatial_turbidity_category_performance.csv",
    index=False
)


# ============================================================
# 4. EXTREME EVENT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("4. EXTREME EVENT SUMMARY")
print("=" * 70)

extreme = df[
    df["Turbidity (NTU)"] >= 100
].copy()

extreme = extreme.sort_values(
    "Turbidity (NTU)",
    ascending=False
)

extreme_export_cols = [
    "Station",
    "Data Acquisition Time",
    "Turbidity (NTU)",
    "predicted_turbidity_NTU",
    "diagnostic_residual",
    "diagnostic_absolute_error",
    "cloud_percentage",
    "days_difference",
    "B8_mean",
    "B11_mean",
    "B12_mean",
    "MNDWI_mean",
    "MNDWI_median",
    "NDWI_mean",
    "NDWI_median",
    "water_fraction",
    "sentinel2_id",
]

extreme[
    extreme_export_cols
].to_csv(
    RESULTS /
    "final_spatial_extreme_events.csv",
    index=False
)

print(
    "Extreme observations >=100 NTU:",
    len(extreme)
)

print(
    "Extreme observations >=500 NTU:",
    len(
        df[
            df["Turbidity (NTU)"] >= 500
        ]
    )
)


# ============================================================
# 5. CLOUD-SENSITIVITY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("5. CLOUD SENSITIVITY")
print("=" * 70)

cloud_rows = []

cloud_conditions = [
    ("All", lambda x: np.ones(len(x), dtype=bool)),
    ("Cloud <=20%", lambda x: x["cloud_percentage"] <= 20),
    ("Cloud <=40%", lambda x: x["cloud_percentage"] <= 40),
    ("Cloud <=60%", lambda x: x["cloud_percentage"] <= 60),
    ("Cloud >60%", lambda x: x["cloud_percentage"] > 60),
]

for label, condition in cloud_conditions:

    mask = condition(df)

    group = df.loc[mask].copy()

    metrics = calculate_metrics(group)

    cloud_rows.append({
        "subset": label,
        **metrics,
    })

cloud_df = pd.DataFrame(
    cloud_rows
)

print(
    cloud_df.round(3).to_string(index=False)
)

cloud_df.to_csv(
    RESULTS /
    "final_spatial_cloud_sensitivity.csv",
    index=False
)


# ============================================================
# 6. TEMPORAL GAP SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("6. TEMPORAL GAP SENSITIVITY")
print("=" * 70)

gap_rows = []

gap_conditions = [
    ("All", lambda x: np.ones(len(x), dtype=bool)),
    ("Gap <=0.5 day", lambda x: x["days_difference"] <= 0.5),
    ("Gap <=1 day", lambda x: x["days_difference"] <= 1.0),
    ("Gap <=1.5 days", lambda x: x["days_difference"] <= 1.5),
    ("Gap <=2 days", lambda x: x["days_difference"] <= 2.0),
]

for label, condition in gap_conditions:

    mask = condition(df)

    group = df.loc[mask].copy()

    metrics = calculate_metrics(group)

    gap_rows.append({
        "subset": label,
        **metrics,
    })

gap_df = pd.DataFrame(
    gap_rows
)

print(
    gap_df.round(3).to_string(index=False)
)

gap_df.to_csv(
    RESULTS /
    "final_spatial_temporal_gap_sensitivity.csv",
    index=False
)


# ============================================================
# 7. B8 QUALITY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("7. B8 QUALITY SUMMARY")
print("=" * 70)

b8_rows = []

for label, group in [
    (
        "B8 <=1",
        df[df["B8_mean"] <= 1]
    ),
    (
        "B8 >1",
        df[df["B8_mean"] > 1]
    ),
]:

    metrics = calculate_metrics(group)

    b8_rows.append({
        "B8_group": label,
        **metrics,
        "mean_B8":
            group["B8_mean"].mean(),
        "median_B8":
            group["B8_mean"].median(),
        "mean_cloud_percent":
            group["cloud_percentage"].mean(),
    })

b8_df = pd.DataFrame(
    b8_rows
)

print(
    b8_df.round(3).to_string(index=False)
)

b8_df.to_csv(
    RESULTS /
    "final_spatial_B8_quality_summary.csv",
    index=False
)


# ============================================================
# 8. MASTER SUMMARY TABLE
# ============================================================

master_rows = []

# Overall
master_rows.append({
    "analysis": "Overall",
    "subset": "All labelled spatial test",
    **calculate_metrics(df),
})

# Categories
for category in category_order:

    group = df[
        df["turbidity_category"] == category
    ]

    master_rows.append({
        "analysis": "Turbidity category",
        "subset": category,
        **calculate_metrics(group),
    })

# Cloud
for label, condition in cloud_conditions:

    group = df.loc[
        condition(df)
    ]

    master_rows.append({
        "analysis": "Cloud sensitivity",
        "subset": label,
        **calculate_metrics(group),
    })

# Temporal gap
for label, condition in gap_conditions:

    group = df.loc[
        condition(df)
    ]

    master_rows.append({
        "analysis": "Temporal-gap sensitivity",
        "subset": label,
        **calculate_metrics(group),
    })

master_df = pd.DataFrame(
    master_rows
)

master_df.to_csv(
    RESULTS /
    "final_spatial_validation_master_summary.csv",
    index=False
)


# ============================================================
# FIGURE 1
# OBSERVED VS PREDICTED
# ============================================================

print("\nGenerating Figure 1...")

fig, ax = plt.subplots(
    figsize=(7.2, 6.2)
)

x = df["Turbidity (NTU)"]
y = df["predicted_turbidity_NTU"]

ax.scatter(
    x,
    y,
    s=22,
    alpha=0.55,
    linewidths=0
)

limit = max(
    x.max(),
    y.max()
)

ax.plot(
    [1, limit],
    [1, limit],
    linestyle="--",
    linewidth=1.2,
    label="1:1 line"
)

ax.set_xlabel(
    "Observed turbidity (NTU)"
)

ax.set_ylabel(
    "Predicted turbidity (NTU)"
)

ax.set_title(
    "Independent spatial validation: observed vs predicted turbidity"
)

ax.set_xlim(
    0,
    limit * 1.03
)

ax.set_ylim(
    0,
    limit * 1.03
)

metrics = calculate_metrics(df)

annotation = (
    f"N = {metrics['n']}\n"
    f"MAE = {metrics['MAE_NTU']:.1f} NTU\n"
    f"RMSE = {metrics['RMSE_NTU']:.1f} NTU\n"
    f"R² = {metrics['R2']:.3f}\n"
    f"r = {metrics['Pearson_r']:.3f}"
)

ax.text(
    0.04,
    0.96,
    annotation,
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85
    )
)

ax.legend(
    loc="lower right"
)

ax.grid(
    alpha=0.2
)

save_figure(
    fig,
    "22_spatial_final_observed_vs_predicted.png"
)


# ============================================================
# FIGURE 2
# LOG-SCALE OBSERVED VS PREDICTED
# ============================================================

print("Generating Figure 2...")

positive = df[
    (df["Turbidity (NTU)"] > 0) &
    (df["predicted_turbidity_NTU"] > 0)
].copy()

fig, ax = plt.subplots(
    figsize=(7.2, 6.2)
)

ax.scatter(
    positive["Turbidity (NTU)"],
    positive["predicted_turbidity_NTU"],
    s=22,
    alpha=0.55,
    linewidths=0
)

log_min = min(
    positive["Turbidity (NTU)"].min(),
    positive["predicted_turbidity_NTU"].min()
)

log_max = max(
    positive["Turbidity (NTU)"].max(),
    positive["predicted_turbidity_NTU"].max()
)

ax.plot(
    [log_min, log_max],
    [log_min, log_max],
    linestyle="--",
    linewidth=1.2,
    label="1:1 line"
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(
    "Observed turbidity (NTU, log scale)"
)

ax.set_ylabel(
    "Predicted turbidity (NTU, log scale)"
)

ax.set_title(
    "Independent spatial validation on logarithmic scale"
)

ax.legend(
    loc="lower right"
)

ax.grid(
    alpha=0.2,
    which="both"
)

save_figure(
    fig,
    "23_spatial_final_log_observed_vs_predicted.png"
)


# ============================================================
# FIGURE 3
# RESIDUALS VS OBSERVED TURBIDITY
# ============================================================

print("Generating Figure 3...")

fig, ax = plt.subplots(
    figsize=(7.4, 6.0)
)

ax.scatter(
    df["Turbidity (NTU)"],
    df["diagnostic_residual"],
    s=22,
    alpha=0.55,
    linewidths=0
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1.2
)

ax.set_xlabel(
    "Observed turbidity (NTU)"
)

ax.set_ylabel(
    "Residual: observed − predicted (NTU)"
)

ax.set_title(
    "Spatial validation residuals"
)

ax.grid(
    alpha=0.2
)

save_figure(
    fig,
    "24_spatial_final_residuals.png"
)


# ============================================================
# FIGURE 4
# RESIDUALS VS PREDICTED
# ============================================================

print("Generating Figure 4...")

fig, ax = plt.subplots(
    figsize=(7.4, 6.0)
)

ax.scatter(
    df["predicted_turbidity_NTU"],
    df["diagnostic_residual"],
    s=22,
    alpha=0.55,
    linewidths=0
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1.2
)

ax.set_xlabel(
    "Predicted turbidity (NTU)"
)

ax.set_ylabel(
    "Residual: observed − predicted (NTU)"
)

ax.set_title(
    "Residuals as a function of predicted turbidity"
)

ax.grid(
    alpha=0.2
)

save_figure(
    fig,
    "25_spatial_final_residuals_vs_predicted.png"
)


# ============================================================
# FIGURE 5
# STATION MAE
# ============================================================

print("Generating Figure 5...")

plot_station = station_df.sort_values(
    "MAE_NTU",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(9.0, 5.8)
)

ax.barh(
    plot_station["Station"],
    plot_station["MAE_NTU"]
)

ax.set_xlabel(
    "MAE (NTU)"
)

ax.set_ylabel(
    "Station"
)

ax.set_title(
    "Independent spatial validation performance by station"
)

ax.grid(
    axis="x",
    alpha=0.2
)

save_figure(
    fig,
    "26_spatial_final_station_MAE.png"
)


# ============================================================
# FIGURE 6
# STATION R2
# ============================================================

print("Generating Figure 6...")

plot_station = station_df.sort_values(
    "R2",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(9.0, 5.8)
)

ax.barh(
    plot_station["Station"],
    plot_station["R2"]
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1.0
)

ax.set_xlabel(
    "R²"
)

ax.set_ylabel(
    "Station"
)

ax.set_title(
    "Independent spatial validation R² by station"
)

ax.grid(
    axis="x",
    alpha=0.2
)

save_figure(
    fig,
    "27_spatial_final_station_R2.png"
)


# ============================================================
# FIGURE 7
# STATION BIAS
# ============================================================

print("Generating Figure 7...")

plot_station = station_df.sort_values(
    "bias_observed_minus_predicted_NTU",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(9.0, 5.8)
)

ax.barh(
    plot_station["Station"],
    plot_station[
        "bias_observed_minus_predicted_NTU"
    ]
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1.0
)

ax.set_xlabel(
    "Bias: observed − predicted (NTU)"
)

ax.set_ylabel(
    "Station"
)

ax.set_title(
    "Spatial prediction bias by station"
)

ax.grid(
    axis="x",
    alpha=0.2
)

save_figure(
    fig,
    "28_spatial_final_station_bias.png"
)


# ============================================================
# FIGURE 8
# TURBIDITY CATEGORY MAE
# ============================================================

print("Generating Figure 8...")

fig, ax = plt.subplots(
    figsize=(8.2, 5.8)
)

ax.bar(
    category_df["turbidity_category"],
    category_df["MAE_NTU"]
)

ax.set_xlabel(
    "Observed turbidity category"
)

ax.set_ylabel(
    "MAE (NTU)"
)

ax.set_title(
    "Spatial validation error across turbidity regimes"
)

ax.tick_params(
    axis="x",
    rotation=25
)

ax.grid(
    axis="y",
    alpha=0.2
)

save_figure(
    fig,
    "29_spatial_final_category_MAE.png"
)


# ============================================================
# FIGURE 9
# OBSERVED VS PREDICTED BY TURBIDITY CATEGORY
# ============================================================

print("Generating Figure 9...")

category_plot = category_df.copy()

fig, ax = plt.subplots(
    figsize=(8.4, 5.8)
)

positions = np.arange(
    len(category_plot)
)

width = 0.36

ax.bar(
    positions - width / 2,
    category_plot["observed_mean_NTU"],
    width,
    label="Observed mean"
)

ax.bar(
    positions + width / 2,
    category_plot["predicted_mean_NTU"],
    width,
    label="Predicted mean"
)

ax.set_xticks(
    positions
)

ax.set_xticklabels(
    category_plot[
        "turbidity_category"
    ],
    rotation=25,
    ha="right"
)

ax.set_ylabel(
    "Mean turbidity (NTU)"
)

ax.set_title(
    "Observed and predicted turbidity by category"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.2
)

save_figure(
    fig,
    "30_spatial_final_category_observed_predicted.png"
)


# ============================================================
# FIGURE 10
# EXTREME EVENT PERFORMANCE
# ============================================================

print("Generating Figure 10...")

top_extreme = extreme.head(
    min(25, len(extreme))
).copy()

top_extreme = top_extreme.sort_values(
    "Turbidity (NTU)",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(9.0, 7.0)
)

positions = np.arange(
    len(top_extreme)
)

ax.barh(
    positions - 0.18,
    top_extreme["Turbidity (NTU)"],
    height=0.34,
    label="Observed"
)

ax.barh(
    positions + 0.18,
    top_extreme["predicted_turbidity_NTU"],
    height=0.34,
    label="Predicted"
)

labels = [
    f"{s} | {d}"
    for s, d in zip(
        top_extreme["Station"],
        top_extreme["Data Acquisition Time"]
    )
]

ax.set_yticks(
    positions
)

ax.set_yticklabels(
    labels,
    fontsize=7
)

ax.set_xlabel(
    "Turbidity (NTU)"
)

ax.set_title(
    "Extreme spatial-test events: observed vs predicted"
)

ax.legend()

ax.grid(
    axis="x",
    alpha=0.2
)

save_figure(
    fig,
    "31_spatial_final_extreme_events.png"
)


# ============================================================
# FIGURE 11
# CLOUD VS ABSOLUTE ERROR
# ============================================================

print("Generating Figure 11...")

fig, ax = plt.subplots(
    figsize=(7.6, 6.0)
)

ax.scatter(
    df["cloud_percentage"],
    df["diagnostic_absolute_error"],
    s=22,
    alpha=0.55,
    linewidths=0
)

ax.set_xlabel(
    "Sentinel-2 cloud percentage (%)"
)

ax.set_ylabel(
    "Absolute error (NTU)"
)

ax.set_title(
    "Cloud contamination versus spatial prediction error"
)

ax.grid(
    alpha=0.2
)

save_figure(
    fig,
    "32_spatial_final_cloud_vs_error.png"
)


# ============================================================
# FIGURE 12
# TEMPORAL GAP VS ABSOLUTE ERROR
# ============================================================

print("Generating Figure 12...")

fig, ax = plt.subplots(
    figsize=(7.6, 6.0)
)

ax.scatter(
    df["days_difference"],
    df["diagnostic_absolute_error"],
    s=22,
    alpha=0.55,
    linewidths=0
)

ax.set_xlabel(
    "CWC–Sentinel-2 temporal difference (days)"
)

ax.set_ylabel(
    "Absolute error (NTU)"
)

ax.set_title(
    "Temporal mismatch versus spatial prediction error"
)

ax.grid(
    alpha=0.2
)

save_figure(
    fig,
    "33_spatial_final_temporal_gap_vs_error.png"
)


# ============================================================
# FIGURE 13
# OBSERVED TURBIDITY DISTRIBUTION
# ============================================================

print("Generating Figure 13...")

fig, ax = plt.subplots(
    figsize=(7.6, 5.8)
)

ax.hist(
    df["Turbidity (NTU)"],
    bins=40
)

ax.set_xlabel(
    "Observed turbidity (NTU)"
)

ax.set_ylabel(
    "Number of observations"
)

ax.set_title(
    "Observed turbidity distribution in independent spatial test"
)

ax.grid(
    axis="y",
    alpha=0.2
)

save_figure(
    fig,
    "34_spatial_final_observed_turbidity_distribution.png"
)


# ============================================================
# FIGURE 14
# LOG OBSERVED TURBIDITY DISTRIBUTION
# ============================================================

print("Generating Figure 14...")

fig, ax = plt.subplots(
    figsize=(7.6, 5.8)
)

ax.hist(
    positive["Turbidity (NTU)"],
    bins=35
)

ax.set_xscale(
    "log"
)

ax.set_xlabel(
    "Observed turbidity (NTU, log scale)"
)

ax.set_ylabel(
    "Number of observations"
)

ax.set_title(
    "Observed spatial-test turbidity distribution"
)

ax.grid(
    axis="y",
    alpha=0.2
)

save_figure(
    fig,
    "35_spatial_final_log_turbidity_distribution.png"
)


# ============================================================
# FIGURE 15
# FEATURE SHIFT
# ============================================================

print("Generating Figure 15...")

feature_shift = pd.read_csv(
    RESULTS /
    "spatial_feature_distribution_shift.csv"
)

feature_shift_plot = feature_shift.copy()

feature_shift_plot = feature_shift_plot.sort_values(
    "standardized_mean_shift"
)

fig, ax = plt.subplots(
    figsize=(8.5, 5.8)
)

ax.barh(
    feature_shift_plot["feature"],
    feature_shift_plot[
        "standardized_mean_shift"
    ]
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1.0
)

ax.set_xlabel(
    "Standardized mean shift"
)

ax.set_ylabel(
    "Compact-8 feature"
)

ax.set_title(
    "Feature-distribution shift: Etawah training vs spatial test"
)

ax.grid(
    axis="x",
    alpha=0.2
)

save_figure(
    fig,
    "36_spatial_final_feature_distribution_shift.png"
)


# ============================================================
# FIGURE 16
# STATION EXTREME-TURBIDITY BURDEN
# ============================================================

print("Generating Figure 16...")

station_burden = pd.read_csv(
    RESULTS /
    "spatial_station_extreme_burden.csv"
)

station_burden = station_burden.sort_values(
    "percent_ge_100_NTU",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(9.0, 5.8)
)

ax.barh(
    station_burden["Station"],
    station_burden["percent_ge_100_NTU"]
)

ax.set_xlabel(
    "Observations ≥100 NTU (%)"
)

ax.set_ylabel(
    "Station"
)

ax.set_title(
    "Extreme-turbidity burden across independent Yamuna stations"
)

ax.grid(
    axis="x",
    alpha=0.2
)

save_figure(
    fig,
    "37_spatial_final_station_extreme_burden.png"
)


# ============================================================
# FIGURE 17
# HIGH TURBIDITY ERROR BY CLOUD GROUP
# ============================================================

print("Generating Figure 17...")

high_cloud = pd.read_csv(
    RESULTS /
    "spatial_high_turbidity_cloud_diagnostics.csv"
)

high_cloud = high_cloud[
    high_cloud["subset"] !=
    "All high turbidity >=100"
].copy()

fig, ax = plt.subplots(
    figsize=(8.0, 5.8)
)

ax.bar(
    high_cloud["subset"],
    high_cloud["MAE_NTU"]
)

ax.set_ylabel(
    "MAE (NTU)"
)

ax.set_xlabel(
    "Cloud subset"
)

ax.set_title(
    "High-turbidity (≥100 NTU) error under cloud conditions"
)

ax.tick_params(
    axis="x",
    rotation=25
)

ax.grid(
    axis="y",
    alpha=0.2
)

save_figure(
    fig,
    "38_spatial_final_high_turbidity_cloud_MAE.png"
)


# ============================================================
# FIGURE 18
# STATION OBSERVED VS PREDICTED MEANS
# ============================================================

print("Generating Figure 18...")

station_means = station_df.sort_values(
    "observed_mean_NTU"
)

fig, ax = plt.subplots(
    figsize=(9.0, 5.8)
)

positions = np.arange(
    len(station_means)
)

width = 0.36

ax.bar(
    positions - width / 2,
    station_means["observed_mean_NTU"],
    width,
    label="Observed mean"
)

ax.bar(
    positions + width / 2,
    station_means["predicted_mean_NTU"],
    width,
    label="Predicted mean"
)

ax.set_xticks(
    positions
)

ax.set_xticklabels(
    station_means["Station"],
    rotation=35,
    ha="right"
)

ax.set_ylabel(
    "Mean turbidity (NTU)"
)

ax.set_title(
    "Observed and predicted turbidity means by station"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.2
)

save_figure(
    fig,
    "39_spatial_final_station_observed_predicted_means.png"
)


# ============================================================
# FINAL MANIFEST
# ============================================================

figure_files = sorted(
    OUTPUT_DIR.glob("*.png")
)

manifest = pd.DataFrame({
    "figure_number": range(
        1,
        len(figure_files) + 1
    ),
    "filename": [
        f.name
        for f in figure_files
    ],
    "path": [
        str(f)
        for f in figure_files
    ],
})

manifest.to_csv(
    RESULTS /
    "final_spatial_figure_manifest.csv",
    index=False
)


# ============================================================
# FINAL VALIDATION REPORT
# ============================================================

report = {
    "prediction_file": str(PRED_FILE),
    "feature_file": str(FEATURE_FILE),
    "labelled_rows": len(df),
    "stations": df["Station"].nunique(),
    "unique_sentinel2_images":
        df["sentinel2_id"].nunique(),
    "overall_MAE_NTU":
        overall_metrics["MAE_NTU"],
    "overall_RMSE_NTU":
        overall_metrics["RMSE_NTU"],
    "overall_R2":
        overall_metrics["R2"],
    "overall_bias_NTU":
        overall_metrics[
            "bias_observed_minus_predicted_NTU"
        ],
    "overall_Pearson_r":
        overall_metrics["Pearson_r"],
    "extreme_ge_100_count":
        len(extreme),
    "extreme_ge_500_count":
        len(
            df[
                df["Turbidity (NTU)"] >= 500
            ]
        ),
    "B8_gt_1_count":
        int(
            (df["B8_mean"] > 1).sum()
        ),
    "B8_gt_1_percent":
        (
            (df["B8_mean"] > 1).mean()
            * 100
        ),
    "model_retrained":
        "NO",
    "test_added_to_training":
        "NO",
    "predictions_modified":
        "NO",
}

pd.DataFrame(
    [report]
).to_csv(
    RESULTS /
    "final_spatial_validation_report.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STEP 20 COMPLETE")
print("=" * 70)

print(
    "\nFinal labelled spatial test rows:",
    len(df)
)

print(
    "Stations:",
    df["Station"].nunique()
)

print(
    "Unique Sentinel-2 images:",
    df["sentinel2_id"].nunique()
)

print(
    f"Overall MAE: "
    f"{overall_metrics['MAE_NTU']:.3f} NTU"
)

print(
    f"Overall RMSE: "
    f"{overall_metrics['RMSE_NTU']:.3f} NTU"
)

print(
    f"Overall R²: "
    f"{overall_metrics['R2']:.3f}"
)

print(
    f"Overall bias: "
    f"{overall_metrics['bias_observed_minus_predicted_NTU']:.3f} NTU"
)

print(
    f"Pearson r: "
    f"{overall_metrics['Pearson_r']:.3f}"
)

print(
    "\nFigures generated:",
    len(figure_files)
)

print(
    "Figure directory:",
    OUTPUT_DIR
)

print(
    "\nSummary tables written to:",
    RESULTS
)

print("\nModel retrained: NO")
print("Test observations added to training: NO")
print("Predictions modified: NO")

print("\nSTEP 20 COMPLETE.")

