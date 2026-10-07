from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STEP 15
# FINAL TEMPORAL VALIDATION FIGURES
#
# Uses:
# temporal_2024_frozen_model_predictions.csv
#
# NO MODEL RETRAINING
# NO PARAMETER TUNING
# ============================================================


ROOT = Path(
    r"D:\Etawah_Water_Quality\Yamuna_Independent_Test"
)

INPUT_FILE = (
    ROOT
    / "results"
    / "temporal_2024_frozen_model_predictions.csv"
)

FIGURES_DIR = ROOT / "figures"

SUMMARY_FILE = (
    ROOT
    / "results"
    / "temporal_validation_summary.csv"
)

CATEGORY_FILE = (
    ROOT
    / "results"
    / "temporal_validation_category_summary.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("STEP 15 — FINAL TEMPORAL VALIDATION FIGURES")
print("=" * 75)

print("\nInput:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nRows:", len(df))


# ============================================================
# PREPARE
# ============================================================

required = [
    "cwc_datetime",
    "turbidity_NTU",
    "predicted_turbidity_NTU",
    "residual_observed_minus_predicted",
    "days_difference",
    "cloud_percentage",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


for col in [
    "turbidity_NTU",
    "predicted_turbidity_NTU",
    "residual_observed_minus_predicted",
    "days_difference",
    "cloud_percentage",
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )


df["absolute_error"] = (
    df["residual_observed_minus_predicted"]
    .abs()
)

df["cwc_datetime_parsed"] = pd.to_datetime(
    df["cwc_datetime"],
    dayfirst=True,
    errors="coerce",
)

df = df.sort_values(
    "cwc_datetime_parsed"
).reset_index(drop=True)


observed = df["turbidity_NTU"].to_numpy(
    dtype=float
)

predicted = df[
    "predicted_turbidity_NTU"
].to_numpy(
    dtype=float
)


FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# COMMON PLOT SETTINGS
# ============================================================

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# ============================================================
# FIGURE 1
# OBSERVED VS PREDICTED
# ============================================================

print("\nCreating Figure 1...")

fig, ax = plt.subplots(
    figsize=(7, 6)
)

ax.scatter(
    observed,
    predicted,
    s=55,
    alpha=0.8,
)

maximum = max(
    observed.max(),
    predicted.max(),
)

ax.plot(
    [0, maximum],
    [0, maximum],
    linestyle="--",
    linewidth=1.5,
)

ax.set_xlabel(
    "Observed turbidity (NTU)"
)

ax.set_ylabel(
    "Predicted turbidity (NTU)"
)

ax.set_title(
    "Independent Temporal Validation: 2024"
)

ax.text(
    0.05,
    0.95,
    "n = 14\nMAE = 26.27 NTU\nRMSE = 35.09 NTU\nR² = 0.064\nPearson r = 0.762",
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85,
    ),
)

ax.grid(
    alpha=0.25
)

fig.tight_layout()

file1 = (
    FIGURES_DIR
    / "14_temporal_observed_vs_predicted_2024.png"
)

fig.savefig(
    file1,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2
# RESIDUALS VS PREDICTED
# ============================================================

print("Creating Figure 2...")

fig, ax = plt.subplots(
    figsize=(7, 6)
)

ax.scatter(
    predicted,
    df["residual_observed_minus_predicted"],
    s=55,
    alpha=0.8,
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1.5,
)

ax.set_xlabel(
    "Predicted turbidity (NTU)"
)

ax.set_ylabel(
    "Residual: observed − predicted (NTU)"
)

ax.set_title(
    "2024 Temporal Validation Residuals"
)

ax.grid(
    alpha=0.25
)

fig.tight_layout()

file2 = (
    FIGURES_DIR
    / "15_temporal_residuals_2024.png"
)

fig.savefig(
    file2,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 3
# CLOUD COVER VS ABSOLUTE ERROR
# ============================================================

print("Creating Figure 3...")

fig, ax = plt.subplots(
    figsize=(7, 6)
)

ax.scatter(
    df["cloud_percentage"],
    df["absolute_error"],
    s=60,
    alpha=0.85,
)

ax.set_xlabel(
    "Sentinel-2 scene cloud percentage (%)"
)

ax.set_ylabel(
    "Absolute prediction error (NTU)"
)

ax.set_title(
    "Cloud Cover vs Prediction Error — 2024"
)

ax.grid(
    alpha=0.25
)

fig.tight_layout()

file3 = (
    FIGURES_DIR
    / "16_temporal_cloud_vs_error_2024.png"
)

fig.savefig(
    file3,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 4
# TEMPORAL GAP VS ABSOLUTE ERROR
# ============================================================

print("Creating Figure 4...")

fig, ax = plt.subplots(
    figsize=(7, 6)
)

ax.scatter(
    df["days_difference"],
    df["absolute_error"],
    s=60,
    alpha=0.85,
)

ax.set_xlabel(
    "CWC–Sentinel-2 temporal difference (days)"
)

ax.set_ylabel(
    "Absolute prediction error (NTU)"
)

ax.set_title(
    "Temporal Gap vs Prediction Error — 2024"
)

ax.grid(
    alpha=0.25
)

fig.tight_layout()

file4 = (
    FIGURES_DIR
    / "17_temporal_gap_vs_error_2024.png"
)

fig.savefig(
    file4,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 5
# OBSERVED / PREDICTED TIME SERIES
# ============================================================

print("Creating Figure 5...")

fig, ax = plt.subplots(
    figsize=(10, 6)
)

x = np.arange(
    len(df)
)

ax.plot(
    x,
    observed,
    marker="o",
    linewidth=1.8,
    label="Observed",
)

ax.plot(
    x,
    predicted,
    marker="s",
    linewidth=1.8,
    label="Predicted",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    df["cwc_datetime"].tolist(),
    rotation=60,
    ha="right",
)

ax.set_ylabel(
    "Turbidity (NTU)"
)

ax.set_xlabel(
    "CWC observation date/time"
)

ax.set_title(
    "Observed vs Predicted Turbidity — 2024"
)

ax.legend()

ax.grid(
    alpha=0.25
)

fig.tight_layout()

file5 = (
    FIGURES_DIR
    / "18_temporal_2024_observed_predicted_series.png"
)

fig.savefig(
    file5,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 6
# ERROR RANKING
# ============================================================

print("Creating Figure 6...")

error_plot = df.sort_values(
    "absolute_error",
    ascending=True,
).copy()

labels = (
    error_plot["cwc_datetime"]
    .astype(str)
    .tolist()
)

values = (
    error_plot["absolute_error"]
    .to_numpy()
)

fig, ax = plt.subplots(
    figsize=(9, 7)
)

ax.barh(
    np.arange(len(values)),
    values,
)

ax.set_yticks(
    np.arange(len(values))
)

ax.set_yticklabels(
    labels
)

ax.set_xlabel(
    "Absolute prediction error (NTU)"
)

ax.set_ylabel(
    "CWC observation"
)

ax.set_title(
    "2024 Temporal Validation Error Ranking"
)

ax.grid(
    axis="x",
    alpha=0.25,
)

fig.tight_layout()

file6 = (
    FIGURES_DIR
    / "19_temporal_2024_error_ranking.png"
)

fig.savefig(
    file6,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 7
# CLOUD GROUP MAE
# ============================================================

print("Creating Figure 7...")

cloud_groups = [
    ("≤20%", df["cloud_percentage"] <= 20),
    ("≤40%", df["cloud_percentage"] <= 40),
    ("≤60%", df["cloud_percentage"] <= 60),
    (">60%", df["cloud_percentage"] > 60),
]

cloud_labels = []
cloud_mae = []
cloud_n = []

for label, mask in cloud_groups:

    subset = df[mask]

    if len(subset) == 0:
        continue

    mae = np.mean(
        np.abs(
            subset["turbidity_NTU"]
            - subset["predicted_turbidity_NTU"]
        )
    )

    cloud_labels.append(label)
    cloud_mae.append(mae)
    cloud_n.append(len(subset))


fig, ax = plt.subplots(
    figsize=(7, 6)
)

bars = ax.bar(
    cloud_labels,
    cloud_mae,
)

ax.set_xlabel(
    "Cloud-cover subset"
)

ax.set_ylabel(
    "MAE (NTU)"
)

ax.set_title(
    "2024 Temporal Error by Cloud-Cover Subset"
)

for bar, mae, n in zip(
    bars,
    cloud_mae,
    cloud_n,
):
    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{mae:.1f}\n(n={n})",
        ha="center",
        va="bottom",
    )

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

file7 = (
    FIGURES_DIR
    / "20_temporal_cloud_group_mae_2024.png"
)

fig.savefig(
    file7,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 8
# TEMPORAL GAP GROUP MAE
# ============================================================

print("Creating Figure 8...")

time_groups = [
    ("≤0.5 d", df["days_difference"] <= 0.5),
    ("≤1.0 d", df["days_difference"] <= 1.0),
    ("≤1.5 d", df["days_difference"] <= 1.5),
    (">1.5 d", df["days_difference"] > 1.5),
]

time_labels = []
time_mae = []
time_n = []

for label, mask in time_groups:

    subset = df[mask]

    if len(subset) == 0:
        continue

    mae = np.mean(
        np.abs(
            subset["turbidity_NTU"]
            - subset["predicted_turbidity_NTU"]
        )
    )

    time_labels.append(label)
    time_mae.append(mae)
    time_n.append(len(subset))


fig, ax = plt.subplots(
    figsize=(7, 6)
)

bars = ax.bar(
    time_labels,
    time_mae,
)

ax.set_xlabel(
    "Temporal-match subset"
)

ax.set_ylabel(
    "MAE (NTU)"
)

ax.set_title(
    "2024 Temporal Error by CWC–Sentinel-2 Time Gap"
)

for bar, mae, n in zip(
    bars,
    time_mae,
    time_n,
):
    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{mae:.1f}\n(n={n})",
        ha="center",
        va="bottom",
    )

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

file8 = (
    FIGURES_DIR
    / "21_temporal_gap_group_mae_2024.png"
)

fig.savefig(
    file8,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# SUMMARY TABLE
# ============================================================

print("\nCreating summary tables...")


def metrics_for_subset(subset):

    n = len(subset)

    if n == 0:
        return {
            "n": 0,
            "MAE_NTU": np.nan,
            "RMSE_NTU": np.nan,
            "R2": np.nan,
            "bias_observed_minus_predicted_NTU": np.nan,
        }

    y = subset["turbidity_NTU"].to_numpy(
        dtype=float
    )

    p = subset[
        "predicted_turbidity_NTU"
    ].to_numpy(
        dtype=float
    )

    mae = np.mean(
        np.abs(y - p)
    )

    rmse = np.sqrt(
        np.mean((y - p) ** 2)
    )

    if n >= 2:
        ss_res = np.sum(
            (y - p) ** 2
        )

        ss_tot = np.sum(
            (y - y.mean()) ** 2
        )

        if ss_tot > 0:
            r2 = 1 - ss_res / ss_tot
        else:
            r2 = np.nan
    else:
        r2 = np.nan

    bias = np.mean(
        y - p
    )

    return {
        "n": n,
        "MAE_NTU": mae,
        "RMSE_NTU": rmse,
        "R2": r2,
        "bias_observed_minus_predicted_NTU": bias,
    }


summary_rows = []

summary_rows.append(
    {
        "subset": "Overall 2024",
        **metrics_for_subset(df),
    }
)

for label, mask in cloud_groups:

    summary_rows.append(
        {
            "subset": f"Cloud {label}",
            **metrics_for_subset(
                df[mask]
            ),
        }
    )

for label, mask in time_groups:

    summary_rows.append(
        {
            "subset": f"Time gap {label}",
            **metrics_for_subset(
                df[mask]
            ),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# CATEGORY SUMMARY
# ============================================================

categories = [
    (
        "Low (<10)",
        df["turbidity_NTU"] < 10,
    ),
    (
        "Moderate (10–20)",
        (
            (df["turbidity_NTU"] >= 10)
            & (df["turbidity_NTU"] < 20)
        ),
    ),
    (
        "Elevated (20–50)",
        (
            (df["turbidity_NTU"] >= 20)
            & (df["turbidity_NTU"] < 50)
        ),
    ),
    (
        "High (50–100)",
        (
            (df["turbidity_NTU"] >= 50)
            & (df["turbidity_NTU"] < 100)
        ),
    ),
    (
        "Very high (>=100)",
        df["turbidity_NTU"] >= 100,
    ),
]


category_rows = []

for label, mask in categories:

    subset = df[mask]

    metrics = metrics_for_subset(
        subset
    )

    category_rows.append(
        {
            "category": label,
            **metrics,
        }
    )


category_df = pd.DataFrame(
    category_rows
)

category_df.to_csv(
    CATEGORY_FILE,
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("STEP 15 COMPLETE")
print("=" * 75)

print("\nFigures saved to:")
print(FIGURES_DIR)

print("\nCreated figures:")

for f in [
    file1,
    file2,
    file3,
    file4,
    file5,
    file6,
    file7,
    file8,
]:
    print(" ", f.name)

print("\nSummary table:")
print(SUMMARY_FILE)

print("\nCategory table:")
print(CATEGORY_FILE)

print("\nNO MODEL RETRAINING WAS PERFORMED.")
print("2024 predictions remain completely frozen.")