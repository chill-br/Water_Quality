import pandas as pd
import numpy as np
from pathlib import Path

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
print("ETAWAH ML DATASET INSPECTION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nDataset shape: {df.shape}")

print("\nColumns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

# ============================================================
# BASIC CHECKS
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATA QUALITY")
print("=" * 70)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nMissing values:")
missing = df.isna().sum()
print(missing[missing > 0])

print("\nInfinite values:")

numeric_df = df.select_dtypes(include=[np.number])

inf_count = np.isinf(numeric_df).sum()

print(inf_count[inf_count > 0])

# ============================================================
# TARGET CHECK
# ============================================================

print("\n" + "=" * 70)
print("2. TURBIDITY TARGET")
print("=" * 70)

target = "turbidity_NTU"

print(df[target].describe())

print("\nUnique turbidity values:")
print(df[target].nunique())

print("\nLowest turbidity:")
print(df[[target, "patch_number", "sentinel2_id"]]
      .sort_values(target)
      .head(10)
      .to_string(index=False))

print("\nHighest turbidity:")
print(df[[target, "patch_number", "sentinel2_id"]]
      .sort_values(target, ascending=False)
      .head(10)
      .to_string(index=False))

# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("3. TURBIDITY RANGES")
print("=" * 70)

bins = [0, 5, 10, 20, 50, 100, 200, 500, np.inf]

labels = [
    "0-5",
    "5-10",
    "10-20",
    "20-50",
    "50-100",
    "100-200",
    "200-500",
    ">500"
]

df["turbidity_class"] = pd.cut(
    df[target],
    bins=bins,
    labels=labels,
    right=False
)

print(
    df["turbidity_class"]
    .value_counts()
    .sort_index()
)

# ============================================================
# FEATURE CHECK
# ============================================================

print("\n" + "=" * 70)
print("4. PATCH FEATURE CHECK")
print("=" * 70)

feature_cols = [
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

print("\nFeature statistics:")

print(
    df[feature_cols]
    .describe()
    .T
    .to_string()
)

# ============================================================
# CORRELATION WITH TURBIDITY
# ============================================================

print("\n" + "=" * 70)
print("5. FEATURE / TURBIDITY CORRELATION")
print("=" * 70)

correlations = (
    df[feature_cols + [target]]
    .corr()[target]
    .drop(target)
    .sort_values(key=lambda x: abs(x), ascending=False)
)

print(correlations)

# ============================================================
# SENTINEL DUPLICATION
# ============================================================

print("\n" + "=" * 70)
print("6. SENTINEL-2 GROUPING")
print("=" * 70)

sentinel_counts = (
    df.groupby("sentinel2_id")
      .size()
      .sort_values(ascending=False)
)

print(f"Unique Sentinel IDs: {len(sentinel_counts)}")

print("\nSentinel images used once:")
print((sentinel_counts == 1).sum())

print("Sentinel images used multiple times:")
print((sentinel_counts > 1).sum())

print("\nTop repeated Sentinel IDs:")

print(
    sentinel_counts
    .head(20)
    .to_string()
)

# ============================================================
# CHECK DUPLICATE PATCH NUMBERS
# ============================================================

print("\n" + "=" * 70)
print("7. DUPLICATE PATCH CHECK")
print("=" * 70)

duplicate_patches = df[
    df["patch_number"].duplicated(keep=False)
]

print(
    f"Duplicate patch-number rows: "
    f"{len(duplicate_patches)}"
)

if len(duplicate_patches) > 0:
    print(
        duplicate_patches[
            [
                "patch_number",
                "patch_filename",
                "sentinel2_id",
                target
            ]
        ].to_string(index=False)
    )

# ============================================================
# CHECK IDENTICAL SENTINEL + TARGET COMBINATIONS
# ============================================================

print("\n" + "=" * 70)
print("8. SENTINEL / TARGET DUPLICATES")
print("=" * 70)

duplicate_groups = (
    df.groupby(
        ["sentinel2_id", target]
    )
    .size()
    .reset_index(name="count")
)

duplicate_groups = duplicate_groups[
    duplicate_groups["count"] > 1
]

print(
    f"Repeated Sentinel + turbidity combinations: "
    f"{len(duplicate_groups)}"
)

if len(duplicate_groups) > 0:
    print(
        duplicate_groups
        .sort_values("count", ascending=False)
        .head(20)
        .to_string(index=False)
    )

# ============================================================
# LEAKAGE-SAFE GROUP SPLIT
# ============================================================

print("\n" + "=" * 70)
print("9. LEAKAGE-SAFE GROUP SPLIT PREVIEW")
print("=" * 70)

unique_sentinels = (
    df["sentinel2_id"]
    .dropna()
    .unique()
)

rng = np.random.default_rng(42)

shuffled = unique_sentinels.copy()
rng.shuffle(shuffled)

n_train_groups = int(len(shuffled) * 0.80)

train_groups = set(
    shuffled[:n_train_groups]
)

test_groups = set(
    shuffled[n_train_groups:]
)

train_df = df[
    df["sentinel2_id"].isin(train_groups)
]

test_df = df[
    df["sentinel2_id"].isin(test_groups)
]

print(f"Training Sentinel groups: {len(train_groups)}")
print(f"Testing Sentinel groups:  {len(test_groups)}")

print(f"\nTraining rows: {len(train_df)}")
print(f"Testing rows:  {len(test_df)}")

print(
    f"\nTraining percentage: "
    f"{len(train_df) / len(df) * 100:.1f}%"
)

print(
    f"Testing percentage: "
    f"{len(test_df) / len(df) * 100:.1f}%"
)

# Verify no leakage

overlap = train_groups.intersection(test_groups)

print(
    f"\nSentinel IDs appearing in BOTH train and test: "
    f"{len(overlap)}"
)

if len(overlap) == 0:
    print("✓ NO SENTINEL LEAKAGE")
else:
    print("!!! LEAKAGE DETECTED !!!")

# ============================================================
# SAVE INSPECTION TABLE
# ============================================================

OUTPUT_FILE = DATA_DIR / "ml_dataset_inspection.csv"

correlation_table = correlations.reset_index()

correlation_table.columns = [
    "feature",
    "correlation_with_turbidity"
]

correlation_table.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print(
    f"\nCorrelation report saved to:\n"
    f"{OUTPUT_FILE}"
)