from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

INPUT_FILE = DATA_DIR / "ml_ready_turbidity.csv"

OUTPUT_STATS = DATA_DIR / "turbidity_feature_statistics.csv"
OUTPUT_EXTREMES = DATA_DIR / "extreme_turbidity_analysis.csv"

PLOT_DIRECTORY = DATA_DIR / "turbidity_feature_plots"
PLOT_DIRECTORY.mkdir(exist_ok=True)


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
    "water_fraction",
]

TARGET = "turbidity_NTU"


# Additional patch-level statistics that already exist
PATCH_STAT_FEATURES = [
    "B2_std",
    "B3_std",
    "B4_std",
    "B8_std",
    "B11_std",
    "B12_std",
    "NDWI_std",
    "MNDWI_std",
    "B2_median",
    "B3_median",
    "B4_median",
    "B8_median",
    "B11_median",
    "B12_median",
    "NDWI_median",
    "MNDWI_median",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ETAWAH TURBIDITY–SPECTRAL RELATIONSHIP ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df.dropna(
    subset=[TARGET]
).copy()

print(f"Valid labelled rows: {len(df)}")


# ============================================================
# BASIC TURBIDITY STATISTICS
# ============================================================

print()
print("=" * 70)
print("TURBIDITY DISTRIBUTION")
print("=" * 70)

print(
    df[TARGET]
    .describe()
    .to_string()
)

print()
print(
    f"Skewness: {df[TARGET].skew():.3f}"
)

print(
    f"Unique turbidity values: "
    f"{df[TARGET].nunique()}"
)


# ============================================================
# TURBIDITY CLASSES
# ============================================================

def turbidity_class(value):

    if value < 10:
        return "Low (<10)"

    elif value < 20:
        return "Moderate (10–20)"

    elif value < 50:
        return "Elevated (20–50)"

    elif value < 100:
        return "High (50–100)"

    else:
        return "Very high (>=100)"


df["turbidity_class"] = df[
    TARGET
].apply(turbidity_class)


print()
print("=" * 70)
print("TURBIDITY CLASSES")
print("=" * 70)

class_counts = (
    df["turbidity_class"]
    .value_counts()
    .sort_index()
)

print(
    class_counts.to_string()
)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print()
print("=" * 70)
print("PEARSON CORRELATION WITH TURBIDITY")
print("=" * 70)

correlations = []

for feature in FEATURES + PATCH_STAT_FEATURES:

    if feature not in df.columns:
        continue

    correlation = df[
        [feature, TARGET]
    ].corr().iloc[0, 1]

    correlations.append(
        {
            "feature": feature,
            "pearson_correlation": correlation,
            "absolute_correlation": abs(correlation),
        }
    )

correlation_df = (
    pd.DataFrame(correlations)
    .sort_values(
        "absolute_correlation",
        ascending=False
    )
)

print(
    correlation_df[
        [
            "feature",
            "pearson_correlation"
        ]
    ].to_string(index=False)
)


# ============================================================
# GROUPED FEATURE STATISTICS
# ============================================================

print()
print("=" * 70)
print("FEATURE STATISTICS BY TURBIDITY CLASS")
print("=" * 70)

group_statistics = []

analysis_features = [
    f for f in FEATURES + PATCH_STAT_FEATURES
    if f in df.columns
]

for turb_class, group in df.groupby(
    "turbidity_class",
    sort=False
):

    for feature in analysis_features:

        group_statistics.append(
            {
                "turbidity_class": turb_class,
                "feature": feature,
                "n": len(group),
                "mean": group[feature].mean(),
                "median": group[feature].median(),
                "std": group[feature].std(),
                "min": group[feature].min(),
                "max": group[feature].max(),
            }
        )

group_stats_df = pd.DataFrame(
    group_statistics
)

group_stats_df.to_csv(
    OUTPUT_STATS,
    index=False
)


# Print mean values for primary features

mean_table = (
    df.groupby("turbidity_class")[
        FEATURES
    ]
    .mean()
)

print()
print("Mean spectral features by turbidity class:")
print(
    mean_table.to_string()
)


# ============================================================
# EXTREME TURBIDITY OBSERVATIONS
# ============================================================

print()
print("=" * 70)
print("EXTREME TURBIDITY OBSERVATIONS")
print("=" * 70)

extreme = df[
    df[TARGET] >= 100
].copy()

print(
    f"Observations >=100 NTU: {len(extreme)}"
)

if len(extreme) > 0:

    columns_to_show = [
        "patch_number",
        "sentinel2_id",
        "satellite_date",
        "cwc_datetime",
        TARGET,
        "days_difference",
        "cloud_percentage",
    ]

    columns_to_show = [
        c for c in columns_to_show
        if c in extreme.columns
    ]

    print(
        extreme[
            columns_to_show
        ]
        .sort_values(
            TARGET,
            ascending=False
        )
        .to_string(index=False)
    )


# ============================================================
# EXTREME OBSERVATION FEATURE VALUES
# ============================================================

if len(extreme) > 0:

    print()
    print("Feature values for extreme observations:")

    extreme_features = extreme[
        [TARGET] + FEATURES
    ].sort_values(
        TARGET,
        ascending=False
    )

    print(
        extreme_features.to_string(
            index=False
        )
    )

    extreme.to_csv(
        OUTPUT_EXTREMES,
        index=False
    )


# ============================================================
# SCATTER PLOTS
# ============================================================

print()
print("=" * 70)
print("CREATING FEATURE–TURBIDITY PLOTS")
print("=" * 70)

for feature in FEATURES:

    # --------------------------------------------------------
    # Linear turbidity
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        df[feature],
        df[TARGET],
        alpha=0.65
    )

    plt.xlabel(feature)
    plt.ylabel("Turbidity (NTU)")
    plt.title(
        f"{feature} vs Turbidity"
    )

    plt.tight_layout()

    output_file = (
        PLOT_DIRECTORY
        / f"{feature}_vs_turbidity.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # Log turbidity
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        df[feature],
        np.log1p(df[TARGET]),
        alpha=0.65
    )

    plt.xlabel(feature)
    plt.ylabel(
        "log1p(Turbidity)"
    )

    plt.title(
        f"{feature} vs log1p(Turbidity)"
    )

    plt.tight_layout()

    output_file = (
        PLOT_DIRECTORY
        / f"{feature}_vs_log_turbidity.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()


# ============================================================
# TURBIDITY DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    df[TARGET],
    bins=30
)

plt.xlabel("Turbidity (NTU)")
plt.ylabel("Number of observations")
plt.title(
    "Etawah CWC Turbidity Distribution"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIRECTORY
    / "turbidity_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# LOG TURBIDITY DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    np.log1p(df[TARGET]),
    bins=30
)

plt.xlabel(
    "log1p(Turbidity)"
)

plt.ylabel(
    "Number of observations"
)

plt.title(
    "Log-Transformed Turbidity Distribution"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIRECTORY
    / "log_turbidity_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# WATER FRACTION VS TURBIDITY
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    df["water_fraction"],
    df[TARGET],
    alpha=0.65
)

plt.xlabel("Water Fraction")
plt.ylabel("Turbidity (NTU)")
plt.title(
    "Water Fraction vs Turbidity"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIRECTORY
    / "water_fraction_vs_turbidity.png",
    dpi=300
)

plt.close()


# ============================================================
# MNDWI VS TURBIDITY
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    df["MNDWI_mean"],
    df[TARGET],
    alpha=0.65
)

plt.xlabel("Mean MNDWI")
plt.ylabel("Turbidity (NTU)")
plt.title(
    "MNDWI vs Turbidity"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIRECTORY
    / "MNDWI_vs_turbidity.png",
    dpi=300
)

plt.close()


# ============================================================
# NDWI VS TURBIDITY
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    df["NDWI_mean"],
    df[TARGET],
    alpha=0.65
)

plt.xlabel("Mean NDWI")
plt.ylabel("Turbidity (NTU)")
plt.title(
    "NDWI vs Turbidity"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIRECTORY
    / "NDWI_vs_turbidity.png",
    dpi=300
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("TOP FEATURES BY ABSOLUTE PEARSON CORRELATION")
print("=" * 70)

print(
    correlation_df[
        [
            "feature",
            "pearson_correlation",
            "absolute_correlation",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


print()
print("=" * 70)
print("OUTPUTS")
print("=" * 70)

print(
    f"Statistics: {OUTPUT_STATS}"
)

print(
    f"Extreme observations: {OUTPUT_EXTREMES}"
)

print(
    f"Plots: {PLOT_DIRECTORY}"
)

print()
print("DONE.")