from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# STEP 5 — SCALE-CORRECTED SPATIAL VALIDATION
# Station-wise performance + publication figures
# ============================================================

BASE = Path(r"D:\Etawah_Water_Quality\Yamuna_Independent_Test")

INPUT = BASE / "results" / "independent_spatial_scale_corrected_predictions.csv"
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


TARGET = "turbidity_NTU"
PRED = "predicted_turbidity_NTU_scale_corrected"


print("=" * 75)
print("STEP 5 — SCALE-CORRECTED SPATIAL VALIDATION")
print("=" * 75)

print("\nLoading:")
print(INPUT)

df = pd.read_csv(INPUT)

print(f"Rows loaded: {len(df)}")

# ------------------------------------------------------------
# Check required columns
# ------------------------------------------------------------

required = [
    "station",
    TARGET,
    PRED,
    "satellite_date",
    "days_difference",
    "cloud_percentage",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        "Missing required columns:\n" +
        "\n".join(missing)
    )

# ------------------------------------------------------------
# Keep labelled observations
# ------------------------------------------------------------

df = df.dropna(subset=[TARGET, PRED]).copy()

print(f"Labelled observations: {len(df)}")

# ------------------------------------------------------------
# Residuals
# ------------------------------------------------------------

df["residual_NTU"] = df[TARGET] - df[PRED]
df["absolute_error_NTU"] = np.abs(df["residual_NTU"])
df["squared_error_NTU"] = df["residual_NTU"] ** 2

# ------------------------------------------------------------
# Overall metrics
# ------------------------------------------------------------

y_true = df[TARGET].values
y_pred = df[PRED].values

overall_mae = mean_absolute_error(y_true, y_pred)
overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
overall_r2 = r2_score(y_true, y_pred)
overall_bias = np.mean(y_true - y_pred)
overall_r = np.corrcoef(y_true, y_pred)[0, 1]

print("\n" + "=" * 75)
print("OVERALL SCALE-CORRECTED SPATIAL VALIDATION")
print("=" * 75)

print(f"N          : {len(df)}")
print(f"MAE        : {overall_mae:.3f} NTU")
print(f"RMSE       : {overall_rmse:.3f} NTU")
print(f"R²         : {overall_r2:.3f}")
print(f"Bias       : {overall_bias:.3f} NTU")
print(f"Pearson r  : {overall_r:.3f}")


# ============================================================
# STATION-WISE PERFORMANCE
# ============================================================

station_rows = []

for station, g in df.groupby("station", sort=True):

    obs = g[TARGET].values
    pred = g[PRED].values

    mae = mean_absolute_error(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    r2 = r2_score(obs, pred) if len(g) >= 2 else np.nan
    r = np.corrcoef(obs, pred)[0, 1] if len(g) >= 2 else np.nan

    station_rows.append({
        "station": station,
        "n": len(g),
        "observed_mean_NTU": np.mean(obs),
        "predicted_mean_NTU": np.mean(pred),
        "observed_median_NTU": np.median(obs),
        "predicted_median_NTU": np.median(pred),
        "observed_min_NTU": np.min(obs),
        "observed_max_NTU": np.max(obs),
        "predicted_min_NTU": np.min(pred),
        "predicted_max_NTU": np.max(pred),
        "MAE_NTU": mae,
        "RMSE_NTU": rmse,
        "R2": r2,
        "Bias_NTU": np.mean(obs - pred),
        "Pearson_r": r,
        "underprediction_percent": np.mean(pred < obs) * 100,
    })


station_df = pd.DataFrame(station_rows)

station_df.to_csv(
    RESULTS / "scale_corrected_station_performance.csv",
    index=False
)

print("\n" + "=" * 75)
print("STATION PERFORMANCE")
print("=" * 75)

print(
    station_df[
        [
            "station",
            "n",
            "MAE_NTU",
            "RMSE_NTU",
            "R2",
            "Bias_NTU",
            "Pearson_r",
        ]
    ].to_string(index=False)
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


df["turbidity_category"] = df[TARGET].apply(turbidity_category)

category_order = [
    "Low (<10)",
    "Moderate (10–20)",
    "Elevated (20–50)",
    "High (50–100)",
    "Very high (≥100)",
]

category_rows = []

for category in category_order:

    g = df[df["turbidity_category"] == category]

    if len(g) == 0:
        continue

    obs = g[TARGET].values
    pred = g[PRED].values

    category_rows.append({
        "category": category,
        "n": len(g),
        "observed_mean_NTU": np.mean(obs),
        "predicted_mean_NTU": np.mean(pred),
        "MAE_NTU": mean_absolute_error(obs, pred),
        "RMSE_NTU": np.sqrt(mean_squared_error(obs, pred)),
        "Bias_NTU": np.mean(obs - pred),
        "underprediction_percent": np.mean(pred < obs) * 100,
    })


category_df = pd.DataFrame(category_rows)

category_df.to_csv(
    RESULTS / "scale_corrected_category_performance.csv",
    index=False
)


# ============================================================
# EXTREME EVENTS
# ============================================================

extreme = df[df[TARGET] >= 100].copy()

extreme = extreme.sort_values(
    TARGET,
    ascending=False
)

extreme.to_csv(
    RESULTS / "scale_corrected_extreme_events.csv",
    index=False
)

print("\n" + "=" * 75)
print("EXTREME EVENTS")
print("=" * 75)

print(f"Extreme observations (>=100 NTU): {len(extreme)}")

print(
    extreme[
        [
            "station",
            TARGET,
            PRED,
            "residual_NTU",
            "cloud_percentage",
            "days_difference",
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# FIGURE 1 — OBSERVED VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 7))

plt.scatter(
    df[TARGET],
    df[PRED],
    alpha=0.55,
    s=35
)

max_value = max(
    df[TARGET].max(),
    df[PRED].max()
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--",
    linewidth=1.5
)

plt.xlabel("Observed turbidity (NTU)")
plt.ylabel("Predicted turbidity (NTU)")
plt.title(
    "Independent Spatial Validation — Observed vs Predicted"
)

plt.text(
    0.05,
    0.95,
    f"N = {len(df)}\n"
    f"MAE = {overall_mae:.2f} NTU\n"
    f"RMSE = {overall_rmse:.2f} NTU\n"
    f"R² = {overall_r2:.3f}\n"
    f"r = {overall_r:.3f}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", alpha=0.8)
)

plt.tight_layout()

plt.savefig(
    FIGURES / "08_spatial_observed_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2 — RESIDUALS VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 7))

plt.scatter(
    df[PRED],
    df["residual_NTU"],
    alpha=0.55,
    s=35
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1.5
)

plt.xlabel("Predicted turbidity (NTU)")
plt.ylabel("Residual: observed − predicted (NTU)")
plt.title(
    "Independent Spatial Validation — Residuals"
)

plt.tight_layout()

plt.savefig(
    FIGURES / "09_spatial_residuals.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3 — STATION MAE
# ============================================================

plot_df = station_df.sort_values(
    "MAE_NTU",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    plot_df["station"],
    plot_df["MAE_NTU"]
)

plt.xlabel("MAE (NTU)")
plt.ylabel("Yamuna station")
plt.title(
    "Independent Spatial Validation — Station-wise MAE"
)

plt.tight_layout()

plt.savefig(
    FIGURES / "10_station_MAE.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 4 — STATION R²
# ============================================================

plot_df = station_df.sort_values(
    "R2",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    plot_df["station"],
    plot_df["R2"]
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1.5
)

plt.xlabel("R²")
plt.ylabel("Yamuna station")
plt.title(
    "Independent Spatial Validation — Station-wise R²"
)

plt.tight_layout()

plt.savefig(
    FIGURES / "11_station_R2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 5 — CATEGORY OBSERVED VS PREDICTED
# ============================================================

x = np.arange(len(category_df))
width = 0.38

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    category_df["observed_mean_NTU"],
    width,
    label="Observed"
)

plt.bar(
    x + width / 2,
    category_df["predicted_mean_NTU"],
    width,
    label="Predicted"
)

plt.xticks(
    x,
    category_df["category"],
    rotation=25,
    ha="right"
)

plt.ylabel("Mean turbidity (NTU)")
plt.xlabel("Observed turbidity category")
plt.title(
    "Independent Spatial Validation — Turbidity Categories"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES / "12_spatial_category_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 6 — EXTREME EVENTS
# ============================================================

extreme_plot = extreme.head(20).copy()

extreme_plot = extreme_plot.sort_values(
    TARGET,
    ascending=True
)

labels = [
    f"{s}\n{v:.0f}"
    for s, v in zip(
        extreme_plot["station"],
        extreme_plot[TARGET]
    )
]

y = np.arange(len(extreme_plot))

plt.figure(figsize=(10, 9))

plt.scatter(
    extreme_plot[TARGET],
    y,
    s=50,
    label="Observed"
)

plt.scatter(
    extreme_plot[PRED],
    y,
    s=50,
    marker="x",
    label="Predicted"
)

plt.yticks(
    y,
    labels
)

plt.xlabel("Turbidity (NTU)")
plt.ylabel("Station / observed NTU")
plt.title(
    "Independent Spatial Validation — Highest Turbidity Events"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES / "13_spatial_extreme_events.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE COMPLETE DIAGNOSTIC DATASET
# ============================================================

df.to_csv(
    RESULTS / "scale_corrected_spatial_validation_complete.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("STEP 5 COMPLETE")
print("=" * 75)

print("\nOverall:")
print(f"  N       = {len(df)}")
print(f"  MAE     = {overall_mae:.3f} NTU")
print(f"  RMSE    = {overall_rmse:.3f} NTU")
print(f"  R²      = {overall_r2:.3f}")
print(f"  Bias    = {overall_bias:.3f} NTU")
print(f"  r       = {overall_r:.3f}")

print("\nSaved results:")
print(
    RESULTS / "scale_corrected_station_performance.csv"
)
print(
    RESULTS / "scale_corrected_category_performance.csv"
)
print(
    RESULTS / "scale_corrected_extreme_events.csv"
)
print(
    RESULTS / "scale_corrected_spatial_validation_complete.csv"
)

print("\nSaved figures:")

figure_names = [
    "08_spatial_observed_vs_predicted.png",
    "09_spatial_residuals.png",
    "10_station_MAE.png",
    "11_station_R2.png",
    "12_spatial_category_comparison.png",
    "13_spatial_extreme_events.png",
]

for name in figure_names:
    print(FIGURES / name)

print("\nNo model retraining performed.")
print("Etawah model remains frozen.")