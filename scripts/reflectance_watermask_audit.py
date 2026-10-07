"""
Reflectance and Water-Mask Quality Audit
Etawah Yamuna Turbidity ML Project

Purpose:
    1. Check Sentinel-2 reflectance ranges.
    2. Identify values outside expected reflectance ranges.
    3. Check water-mask fractions.
    4. Identify problematic patches.
    5. Check whether problematic reflectance values are related
       to turbidity.
    6. Produce summary CSV files for scientific review.

Input:
    data/ml_ready_turbidity.csv
    data/patches/*.tif

Output:
    data/reflectance_audit_summary.csv
    data/reflectance_problematic_patches.csv
    data/watermask_audit_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"D:\Etawah_Water_Quality")

DATA_FILE = (
    BASE_DIR
    / "data"
    / "ml_ready_turbidity.csv"
)

PATCH_DIR = (
    BASE_DIR
    / "data"
    / "patches"
)

REFLECTANCE_SUMMARY = (
    BASE_DIR
    / "data"
    / "reflectance_audit_summary.csv"
)

PROBLEM_PATCHES = (
    BASE_DIR
    / "data"
    / "reflectance_problematic_patches.csv"
)

WATERMASK_SUMMARY = (
    BASE_DIR
    / "data"
    / "watermask_audit_summary.csv"
)


# ============================================================
# EXPECTED BANDS
# ============================================================

BANDS = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
]

BAND_NUMBERS = {
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B8": 4,
    "B11": 5,
    "B12": 6,
}

WATERMASK_BAND = 9


# ============================================================
# LOAD ML DATA
# ============================================================

print("=" * 70)
print("ETAWAH REFLECTANCE + WATER-MASK QUALITY AUDIT")
print("=" * 70)

print()
print(f"Reading metadata:")
print(DATA_FILE)

df = pd.read_csv(DATA_FILE)

print(f"Rows loaded: {len(df)}")

required_columns = [
    "patch_number",
    "patch_filename",
    "sentinel2_id",
    "turbidity_NTU",
    "days_difference",
    "cloud_percentage",
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
# REFLECTANCE AUDIT
# ============================================================

print()
print("=" * 70)
print("REFLECTANCE AUDIT")
print("=" * 70)

all_band_values = {
    band: []
    for band in BANDS
}

problem_records = []

water_records = []

successful = 0
failed = 0


for _, row in df.iterrows():

    patch_number = int(row["patch_number"])

    patch_filename = str(
        row["patch_filename"]
    )

    patch_path = PATCH_DIR / patch_filename

    if not patch_path.exists():

        print(
            f"WARNING: Missing patch "
            f"{patch_number}: {patch_filename}"
        )

        failed += 1

        continue

    try:

        with rasterio.open(patch_path) as src:

            data = src.read()

            # ------------------------------------------------
            # BASIC STRUCTURE CHECK
            # ------------------------------------------------

            if data.shape[0] < 9:

                print(
                    f"WARNING: Patch {patch_number} "
                    f"has only {data.shape[0]} bands."
                )

                failed += 1

                continue

            # ------------------------------------------------
            # BAND VALUES
            # ------------------------------------------------

            patch_problem = False

            patch_record = {
                "patch_number": patch_number,
                "patch_filename": patch_filename,
                "sentinel2_id":
                    row["sentinel2_id"],
                "turbidity_NTU":
                    row["turbidity_NTU"],
                "days_difference":
                    row["days_difference"],
                "cloud_percentage":
                    row["cloud_percentage"],
            }

            for band in BANDS:

                band_number = BAND_NUMBERS[band]

                values = data[
                    band_number - 1
                ].astype(
                    np.float64
                )

                values = values[
                    np.isfinite(values)
                ]

                if len(values) == 0:
                    continue

                all_band_values[
                    band
                ].extend(
                    values.tolist()
                )

                patch_min = float(
                    np.min(values)
                )

                patch_max = float(
                    np.max(values)
                )

                patch_mean = float(
                    np.mean(values)
                )

                patch_median = float(
                    np.median(values)
                )

                patch_record[
                    f"{band}_min"
                ] = patch_min

                patch_record[
                    f"{band}_max"
                ] = patch_max

                patch_record[
                    f"{band}_mean"
                ] = patch_mean

                patch_record[
                    f"{band}_median"
                ] = patch_median

                # Flag values outside broad physical range
                # for investigation.
                #
                # Do NOT automatically call these invalid.
                # They require scientific review.

                if (
                    patch_min < 0
                    or patch_max > 1.0
                ):

                    patch_problem = True

            # ------------------------------------------------
            # WATER MASK
            # ------------------------------------------------

            water_mask = data[
                WATERMASK_BAND - 1
            ]

            water_mask = np.isfinite(
                water_mask
            )

            total_pixels = int(
                water_mask.size
            )

            valid_mask_pixels = (
                water_mask.sum()
            )

            if valid_mask_pixels > 0:

                actual_mask = data[
                    WATERMASK_BAND - 1
                ][water_mask]

                water_fraction = float(
                    np.mean(
                        actual_mask > 0.5
                    )
                )

            else:

                water_fraction = np.nan

            water_records.append({
                "patch_number": patch_number,
                "patch_filename": patch_filename,
                "sentinel2_id":
                    row["sentinel2_id"],
                "turbidity_NTU":
                    row["turbidity_NTU"],
                "water_fraction":
                    water_fraction,
                "cloud_percentage":
                    row["cloud_percentage"],
                "days_difference":
                    row["days_difference"],
                "total_pixels":
                    total_pixels,
            })

            patch_record[
                "water_fraction"
            ] = water_fraction

            # ------------------------------------------------
            # FLAG EXTREME REFLECTANCE
            # ------------------------------------------------

            if patch_problem:

                problem_records.append(
                    patch_record
                )

            successful += 1

    except Exception as exc:

        print(
            f"ERROR reading patch "
            f"{patch_number}: {exc}"
        )

        failed += 1


# ============================================================
# REFLECTANCE SUMMARY
# ============================================================

summary_records = []

for band in BANDS:

    values = np.asarray(
        all_band_values[band],
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        continue

    summary_records.append({

        "band": band,

        "pixel_count":
            len(values),

        "min":
            np.min(values),

        "p01":
            np.percentile(values, 1),

        "p05":
            np.percentile(values, 5),

        "median":
            np.median(values),

        "mean":
            np.mean(values),

        "p95":
            np.percentile(values, 95),

        "p99":
            np.percentile(values, 99),

        "max":
            np.max(values),

        "percent_below_0":
            np.mean(values < 0) * 100,

        "percent_above_1":
            np.mean(values > 1.0) * 100,

        "percent_above_1_5":
            np.mean(values > 1.5) * 100,

    })


reflectance_df = pd.DataFrame(
    summary_records
)

reflectance_df.to_csv(
    REFLECTANCE_SUMMARY,
    index=False
)


# ============================================================
# PROBLEMATIC PATCHES
# ============================================================

problem_df = pd.DataFrame(
    problem_records
)

problem_df.to_csv(
    PROBLEM_PATCHES,
    index=False
)


# ============================================================
# WATER-MASK SUMMARY
# ============================================================

water_df = pd.DataFrame(
    water_records
)

if len(water_df) > 0:

    water_summary = pd.DataFrame([{

        "patch_count":
            len(water_df),

        "minimum":
            water_df["water_fraction"].min(),

        "p05":
            water_df["water_fraction"].quantile(0.05),

        "median":
            water_df["water_fraction"].median(),

        "mean":
            water_df["water_fraction"].mean(),

        "p95":
            water_df["water_fraction"].quantile(0.95),

        "maximum":
            water_df["water_fraction"].max(),

        "zero_water_fraction_patches":
            (
                water_df["water_fraction"]
                == 0
            ).sum(),

        "full_water_fraction_patches":
            (
                water_df["water_fraction"]
                == 1
            ).sum(),

        "between_0_and_1":
            (
                (
                    water_df["water_fraction"] > 0
                )
                &
                (
                    water_df["water_fraction"] < 1
                )
            ).sum(),

    }])

else:

    water_summary = pd.DataFrame()


water_summary.to_csv(
    WATERMASK_SUMMARY,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("REFLECTANCE SUMMARY")
print("=" * 70)

print(
    reflectance_df.to_string(
        index=False
    )
)


print()
print("=" * 70)
print("PROBLEMATIC PATCHES")
print("=" * 70)

print(
    f"Patches successfully analysed: "
    f"{successful}"
)

print(
    f"Patches failed: {failed}"
)

print(
    f"Patches with at least one "
    f"band outside 0–1: "
    f"{len(problem_df)}"
)

if len(problem_df) > 0:

    print()

    print(
        problem_df[
            [
                "patch_number",
                "patch_filename",
                "turbidity_NTU",
                "days_difference",
                "cloud_percentage",
                "water_fraction",
            ]
        ].head(30).to_string(
            index=False
        )
    )


print()
print("=" * 70)
print("WATER-MASK SUMMARY")
print("=" * 70)

if len(water_summary) > 0:

    print(
        water_summary.to_string(
            index=False
        )
    )


# ============================================================
# TURBIDITY COMPARISON
# ============================================================

print()
print("=" * 70)
print("REFLECTANCE / TURBIDITY RELATIONSHIP")
print("=" * 70)

if len(problem_df) > 0:

    labelled_problem = (
        problem_df[
            problem_df["turbidity_NTU"]
            .notna()
        ]
    )

    if len(labelled_problem) > 0:

        print(
            f"Problematic patches with "
            f"turbidity labels: "
            f"{len(labelled_problem)}"
        )

        print(
            f"Mean turbidity: "
            f"{labelled_problem['turbidity_NTU'].mean():.3f}"
        )

        print(
            f"Median turbidity: "
            f"{labelled_problem['turbidity_NTU'].median():.3f}"
        )

        print(
            f"Maximum turbidity: "
            f"{labelled_problem['turbidity_NTU'].max():.3f}"
        )

else:

    print(
        "No patches were flagged "
        "for reflectance review."
    )


# ============================================================
# SAVE WATER-MASK PATCH DETAILS
# ============================================================

water_patch_file = (
    BASE_DIR
    / "data"
    / "watermask_patch_details.csv"
)

water_df.to_csv(
    water_patch_file,
    index=False
)


print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    REFLECTANCE_SUMMARY
)

print(
    PROBLEM_PATCHES
)

print(
    WATERMASK_SUMMARY
)

print(
    water_patch_file
)

print()
print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)