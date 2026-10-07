from pathlib import Path
import pandas as pd
import numpy as np
import rasterio


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\Etawah_Water_Quality")

PATCH_DIR = PROJECT_DIR / "data" / "patches"
LINKAGE_FILE = PROJECT_DIR / "data" / "Etawah_patch_metadata_linkage.csv"
OUTPUT_DIR = PROJECT_DIR / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_BANDS = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
    "NDWI",
    "MNDWI",
    "MNDWI_WATER",
]

STAT_BANDS = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
    "NDWI",
    "MNDWI",
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def extract_patch_features(tif_path):
    """
    Read one TIFF patch and calculate:
      mean
      median
      standard deviation
      minimum
      maximum

    for spectral/index bands.

    Also calculate water-mask fraction.
    """

    with rasterio.open(tif_path) as src:

        band_names = list(src.descriptions)

        # Fallback if descriptions are missing
        if not band_names or all(x is None for x in band_names):
            band_names = EXPECTED_BANDS[:src.count]

        data = src.read().astype(np.float32)

        features = {
            "patch_width": src.width,
            "patch_height": src.height,
            "num_bands": src.count,
            "crs": str(src.crs),
        }

        # ----------------------------------------------------
        # Check band names
        # ----------------------------------------------------

        if len(band_names) != len(EXPECTED_BANDS):
            raise ValueError(
                f"Unexpected number of bands in {tif_path.name}: "
                f"{len(band_names)}"
            )

        # ----------------------------------------------------
        # Spectral/index statistics
        # ----------------------------------------------------

        for i, band_name in enumerate(band_names):

            if band_name not in STAT_BANDS:
                continue

            arr = data[i]

            # Remove NaN and Inf
            valid = arr[np.isfinite(arr)]

            if valid.size == 0:
                features[f"{band_name}_mean"] = np.nan
                features[f"{band_name}_median"] = np.nan
                features[f"{band_name}_std"] = np.nan
                features[f"{band_name}_min"] = np.nan
                features[f"{band_name}_max"] = np.nan

            else:
                features[f"{band_name}_mean"] = float(np.mean(valid))
                features[f"{band_name}_median"] = float(np.median(valid))
                features[f"{band_name}_std"] = float(np.std(valid))
                features[f"{band_name}_min"] = float(np.min(valid))
                features[f"{band_name}_max"] = float(np.max(valid))

        # ----------------------------------------------------
        # Water mask
        # ----------------------------------------------------

        if "MNDWI_WATER" in band_names:

            water_index = band_names.index("MNDWI_WATER")
            water_mask = data[water_index]

            valid_water = water_mask[np.isfinite(water_mask)]

            if valid_water.size == 0:
                features["water_fraction"] = np.nan
            else:
                # Water pixels are represented by 1
                features["water_fraction"] = float(
                    np.mean(valid_water == 1)
                )

        else:
            features["water_fraction"] = np.nan

        return features


# ============================================================
# READ LINKAGE TABLE
# ============================================================

print("=" * 70)
print("BUILDING ETAWAH ML DATASET")
print("=" * 70)

print("\nReading linkage table...")

linkage = pd.read_csv(LINKAGE_FILE)

print(f"Linkage rows: {len(linkage)}")
print("Columns:")
print(list(linkage.columns))


# ============================================================
# EXTRACT PATCH NUMBER
# ============================================================

def get_patch_number(filename):
    """
    Extract patch number from names such as:

    Etawah_1_20210101T052219_20210101T052222_T44RKQ.tif
    """

    try:
        return int(filename.split("_")[1])
    except Exception:
        return np.nan


# ============================================================
# READ ALL TIFF PATCHES
# ============================================================

patch_files = sorted(PATCH_DIR.glob("Etawah_*.tif"))

print(f"\nTIFF patches found: {len(patch_files)}")

if len(patch_files) == 0:
    raise RuntimeError("No TIFF patches found.")


records = []

print("\nExtracting patch statistics...\n")

for counter, tif_path in enumerate(patch_files, start=1):

    patch_number = get_patch_number(tif_path.name)

    print(
        f"[{counter:3d}/{len(patch_files):3d}] "
        f"{tif_path.name}"
    )

    try:

        features = extract_patch_features(tif_path)

        record = {
            "patch": patch_number,
            "patch_file": tif_path.name,
        }

        record.update(features)

        records.append(record)

    except Exception as e:

        print(f"  ERROR: {e}")


patch_features = pd.DataFrame(records)


# ============================================================
# BASIC PATCH CHECK
# ============================================================

print("\n" + "=" * 70)
print("PATCH FEATURE EXTRACTION")
print("=" * 70)

print(f"Successful patches: {len(patch_features)}")

if len(patch_features) != len(patch_files):
    print(
        f"WARNING: {len(patch_files) - len(patch_features)} "
        "patches could not be processed."
    )


# ============================================================
# PREPARE PATCH NUMBER
# ============================================================

# Your linkage CSV uses "patch_number"
# Rename it internally to "patch" so it matches the TIFF features.

if "patch_number" not in linkage.columns:
    raise ValueError(
        "Expected column 'patch_number' was not found "
        "in the linkage CSV."
    )

linkage["patch"] = pd.to_numeric(
    linkage["patch_number"],
    errors="coerce"
)

# Remove rows where patch number could not be read
linkage = linkage.dropna(
    subset=["patch"]
).copy()

# Convert to integer
linkage["patch"] = linkage["patch"].astype(int)

# Ensure exactly one metadata row per patch
linkage = (
    linkage
    .sort_values("patch")
    .drop_duplicates(
        subset=["patch"],
        keep="first"
    )
)


# ============================================================
# JOIN METADATA + PATCH FEATURES
# ============================================================

print("\nJoining CWC metadata with TIFF features...")

df = linkage.merge(
    patch_features,
    on="patch",
    how="left",
    suffixes=("", "_tif")
)


# ============================================================
# DETERMINE LABEL STATUS
# ============================================================

df["has_turbidity_label"] = (
    pd.to_numeric(
        df["turbidity_NTU"],
        errors="coerce"
    ).notna()
)

df["has_patch_features"] = (
    df["B2_mean"].notna()
)


# ============================================================
# SAVE COMPLETE DATASET
# ============================================================

complete_file = OUTPUT_DIR / "Etawah_complete_patch_dataset.csv"

df.to_csv(
    complete_file,
    index=False
)

print(f"\nComplete dataset saved:")
print(complete_file)


# ============================================================
# LABELLED DATASET
# ============================================================

labelled = df[
    df["has_turbidity_label"] &
    df["has_patch_features"]
].copy()


# ------------------------------------------------------------
# Remove rows with invalid spectral values
# ------------------------------------------------------------

required_feature_columns = [
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

for col in required_feature_columns:

    labelled[col] = pd.to_numeric(
        labelled[col],
        errors="coerce"
    )


labelled = labelled.dropna(
    subset=required_feature_columns +
           ["turbidity_NTU"]
)


# ============================================================
# SAVE ML DATASET
# ============================================================

ml_file = OUTPUT_DIR / "ml_ready_turbidity.csv"

labelled.to_csv(
    ml_file,
    index=False
)


# ============================================================
# UNLABELLED DATASET
# ============================================================

unlabelled = df[
    (~df["has_turbidity_label"]) &
    (df["has_patch_features"])
].copy()


unlabelled_file = OUTPUT_DIR / "unlabelled_patches.csv"

unlabelled.to_csv(
    unlabelled_file,
    index=False
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print(f"Total patch rows:       {len(df)}")
print(f"Labelled ML rows:       {len(labelled)}")
print(f"Unlabelled rows:        {len(unlabelled)}")

print(
    f"\nUnique Sentinel IDs:    "
    f"{df['sentinel2_id'].nunique() if 'sentinel2_id' in df.columns else 'N/A'}"
)

print(
    f"Unique satellite IDs:   "
    f"{df['satellite_id'].nunique() if 'satellite_id' in df.columns else 'N/A'}"
)


# ============================================================
# TURBIDITY SUMMARY
# ============================================================

if len(labelled) > 0:

    labelled["turbidity_NTU"] = pd.to_numeric(
        labelled["turbidity_NTU"],
        errors="coerce"
    )

    print("\nTurbidity statistics:")

    print(
        labelled["turbidity_NTU"].describe()
    )


# ============================================================
# WATER FRACTION SUMMARY
# ============================================================

if len(labelled) > 0:

    print("\nWater fraction statistics:")

    print(
        labelled["water_fraction"].describe()
    )


# ============================================================
# CHECK DUPLICATE SENTINEL IMAGES
# ============================================================

if "sentinel2_id" in labelled.columns:

    duplicate_counts = (
        labelled["sentinel2_id"]
        .value_counts()
    )

    duplicate_images = duplicate_counts[
        duplicate_counts > 1
    ]

    print(
        "\nSentinel images used by multiple labelled patches: "
        f"{len(duplicate_images)}"
    )

    if len(duplicate_images) > 0:

        print("\nTop repeated Sentinel IDs:")

        print(
            duplicate_images
            .head(10)
            .to_string()
        )


# ============================================================
# SAVE DUPLICATE SENTINEL SUMMARY
# ============================================================

if "sentinel2_id" in df.columns:

    duplicate_summary = (
        df.groupby("sentinel2_id")
        .agg(
            patch_count=("patch", "count"),
            patch_numbers=("patch", lambda x: ",".join(
                map(str, sorted(x.dropna().astype(int)))
            ))
        )
        .reset_index()
    )

    duplicate_summary = duplicate_summary[
        duplicate_summary["patch_count"] > 1
    ]

    duplicate_file = (
        OUTPUT_DIR /
        "duplicate_sentinel_acquisitions.csv"
    )

    duplicate_summary.to_csv(
        duplicate_file,
        index=False
    )

    print(
        f"\nDuplicate Sentinel summary saved:"
        f"\n{duplicate_file}"
    )


# ============================================================
# PRINT IMPORTANT COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("ML DATASET COLUMNS")
print("=" * 70)

print(
    "\n".join(
        f"{i+1:2d}. {col}"
        for i, col in enumerate(labelled.columns)
    )
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"\nML dataset:")
print(ml_file)

print(f"\nUnlabelled dataset:")
print(unlabelled_file)

print(f"\nComplete dataset:")
print(complete_file)

print("\nNext step: inspect the ML dataset before training.")