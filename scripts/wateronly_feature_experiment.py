"""
Water-Only Feature Experiment
Etawah Yamuna Turbidity ML Project

Purpose:
    Compare spectral features calculated from:
        A) the entire Sentinel-2 patch
        B) only pixels classified as water by MNDWI_WATER

Important:
    - Existing TIFF files are NOT modified.
    - Existing ml_ready_turbidity.csv is NOT modified.
    - Features are calculated only for labelled observations.
    - GroupKFold is used by sentinel2_id to prevent leakage.

Inputs:
    data/ml_ready_turbidity.csv
    data/patches/*.tif

Outputs:
    data/ml_ready_turbidity_wateronly.csv
    data/wateronly_feature_cv_results.csv
    data/wateronly_feature_comparison.csv
"""


from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


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

WATERONLY_DATA_FILE = (
    BASE_DIR
    / "data"
    / "ml_ready_turbidity_wateronly.csv"
)

CV_RESULTS_FILE = (
    BASE_DIR
    / "data"
    / "wateronly_feature_cv_results.csv"
)

COMPARISON_FILE = (
    BASE_DIR
    / "data"
    / "wateronly_feature_comparison.csv"
)


# ============================================================
# SENTINEL-2 BANDS
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

NDWI_BAND = 7
MNDWI_BAND = 8
WATER_MASK_BAND = 9


# ============================================================
# WATER-ONLY FEATURES
# ============================================================

WATERONLY_FEATURES = [
    "MNDWI_water_median",
    "NDWI_water_median",
    "MNDWI_water_mean",
    "NDWI_water_mean",
    "water_fraction",
    "B11_water_mean",
    "B12_water_mean",
    "B8_water_mean",
]


# ============================================================
# CURRENT ALL-PIXEL FEATURES
# ============================================================

ALL_PIXEL_FEATURES = [
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
print("ETAWAH WATER-ONLY FEATURE EXPERIMENT")
print("=" * 70)

print()
print(f"Reading:")
print(DATA_FILE)

df = pd.read_csv(DATA_FILE)

print(
    f"Rows loaded: {len(df)}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "patch_number",
    "patch_filename",
    "sentinel2_id",
    TARGET,
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
# KEEP LABELLED OBSERVATIONS
# ============================================================

df = df[
    df[TARGET].notna()
].copy()

print(
    f"Labelled rows: {len(df)}"
)


# ============================================================
# EXTRACT WATER-ONLY FEATURES
# ============================================================

print()
print("=" * 70)
print("EXTRACTING WATER-ONLY FEATURES")
print("=" * 70)

records = []

failed = []

for counter, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    patch_number = int(
        row["patch_number"]
    )

    patch_filename = str(
        row["patch_filename"]
    )

    patch_path = (
        PATCH_DIR
        / patch_filename
    )

    print(
        f"[{counter:03d}/{len(df)}] "
        f"Patch {patch_number}: "
        f"{patch_filename}"
    )

    if not patch_path.exists():

        print(
            "    WARNING: patch not found"
        )

        failed.append({
            "patch_number":
                patch_number,
            "patch_filename":
                patch_filename,
            "reason":
                "file_not_found",
        })

        continue

    try:

        with rasterio.open(
            patch_path
        ) as src:

            data = src.read().astype(
                np.float64
            )

        # ----------------------------------------------------
        # CHECK BAND COUNT
        # ----------------------------------------------------

        if data.shape[0] < 9:

            raise ValueError(
                f"Expected at least 9 bands, "
                f"found {data.shape[0]}"
            )

        # ----------------------------------------------------
        # WATER MASK
        # ----------------------------------------------------

        mask_values = data[
            WATER_MASK_BAND - 1
        ]

        valid_mask = np.isfinite(
            mask_values
        )

        water_mask = (
            valid_mask
            &
            (mask_values > 0.5)
        )

        total_pixels = (
            mask_values.size
        )

        water_pixels = int(
            water_mask.sum()
        )

        if total_pixels == 0:

            raise ValueError(
                "Patch contains no pixels"
            )

        water_fraction = (
            water_pixels
            / total_pixels
        )

        # ----------------------------------------------------
        # REQUIRE AT LEAST ONE WATER PIXEL
        # ----------------------------------------------------

        if water_pixels == 0:

            print(
                "    WARNING: "
                "0 water pixels"
            )

            record = {
                "patch_number":
                    patch_number,
                "patch_filename":
                    patch_filename,
                "sentinel2_id":
                    row["sentinel2_id"],
                TARGET:
                    row[TARGET],
                "water_pixel_count":
                    0,
                "water_fraction":
                    0.0,
            }

            # All water-only statistics are NaN
            for band in BANDS:

                record[
                    f"{band}_water_mean"
                ] = np.nan

                record[
                    f"{band}_water_median"
                ] = np.nan

                record[
                    f"{band}_water_std"
                ] = np.nan

            record[
                "NDWI_water_mean"
            ] = np.nan

            record[
                "NDWI_water_median"
            ] = np.nan

            record[
                "NDWI_water_std"
            ] = np.nan

            record[
                "MNDWI_water_mean"
            ] = np.nan

            record[
                "MNDWI_water_median"
            ] = np.nan

            record[
                "MNDWI_water_std"
            ] = np.nan

            records.append(record)

            continue

        # ----------------------------------------------------
        # INITIAL RECORD
        # ----------------------------------------------------

        record = {
            "patch_number":
                patch_number,

            "patch_filename":
                patch_filename,

            "sentinel2_id":
                row["sentinel2_id"],

            TARGET:
                row[TARGET],

            "water_pixel_count":
                water_pixels,

            "total_pixel_count":
                total_pixels,

            "water_fraction":
                water_fraction,
        }

        # ----------------------------------------------------
        # SPECTRAL BANDS
        # ----------------------------------------------------

        for band in BANDS:

            band_number = (
                BAND_NUMBERS[band]
            )

            values = data[
                band_number - 1
            ][water_mask]

            values = values[
                np.isfinite(values)
            ]

            if len(values) == 0:

                mean_value = np.nan
                median_value = np.nan
                std_value = np.nan

            else:

                mean_value = float(
                    np.mean(values)
                )

                median_value = float(
                    np.median(values)
                )

                std_value = float(
                    np.std(values)
                )

            record[
                f"{band}_water_mean"
            ] = mean_value

            record[
                f"{band}_water_median"
            ] = median_value

            record[
                f"{band}_water_std"
            ] = std_value

        # ----------------------------------------------------
        # NDWI
        # ----------------------------------------------------

        ndwi = data[
            NDWI_BAND - 1
        ][water_mask]

        ndwi = ndwi[
            np.isfinite(ndwi)
        ]

        if len(ndwi) > 0:

            record[
                "NDWI_water_mean"
            ] = float(
                np.mean(ndwi)
            )

            record[
                "NDWI_water_median"
            ] = float(
                np.median(ndwi)
            )

            record[
                "NDWI_water_std"
            ] = float(
                np.std(ndwi)
            )

        else:

            record[
                "NDWI_water_mean"
            ] = np.nan

            record[
                "NDWI_water_median"
            ] = np.nan

            record[
                "NDWI_water_std"
            ] = np.nan

        # ----------------------------------------------------
        # MNDWI
        # ----------------------------------------------------

        mndwi = data[
            MNDWI_BAND - 1
        ][water_mask]

        mndwi = mndwi[
            np.isfinite(mndwi)
        ]

        if len(mndwi) > 0:

            record[
                "MNDWI_water_mean"
            ] = float(
                np.mean(mndwi)
            )

            record[
                "MNDWI_water_median"
            ] = float(
                np.median(mndwi)
            )

            record[
                "MNDWI_water_std"
            ] = float(
                np.std(mndwi)
            )

        else:

            record[
                "MNDWI_water_mean"
            ] = np.nan

            record[
                "MNDWI_water_median"
            ] = np.nan

            record[
                "MNDWI_water_std"
            ] = np.nan

        records.append(record)

    except Exception as exc:

        print(
            f"    ERROR: {exc}"
        )

        failed.append({
            "patch_number":
                patch_number,

            "patch_filename":
                patch_filename,

            "reason":
                str(exc),
        })


# ============================================================
# CREATE WATER-ONLY DATASET
# ============================================================

water_df = pd.DataFrame(
    records
)

print()
print("=" * 70)
print("WATER-ONLY DATASET")
print("=" * 70)

print(
    f"Successful records: "
    f"{len(water_df)}"
)

print(
    f"Failed records: "
    f"{len(failed)}"
)

if len(failed) > 0:

    print()

    print(
        pd.DataFrame(
            failed
        ).to_string(
            index=False
        )
    )


# ============================================================
# SAVE WATER-ONLY DATASET
# ============================================================

water_df.to_csv(
    WATERONLY_DATA_FILE,
    index=False
)

print()
print(
    f"Saved:"
)

print(
    WATERONLY_DATA_FILE
)


# ============================================================
# REMOVE ROWS WITHOUT WATER FEATURES
# ============================================================

water_ml = water_df.dropna(
    subset=WATERONLY_FEATURES + [
        TARGET,
        GROUP,
    ]
).copy()

print()
print(
    f"Rows usable for water-only ML: "
    f"{len(water_ml)}"
)

print(
    f"Sentinel groups: "
    f"{water_ml[GROUP].nunique()}"
)


# ============================================================
# MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    data,
    features,
    model_name,
):

    X = data[features]

    y = data[TARGET]

    groups = data[GROUP]

    n_groups = groups.nunique()

    if n_groups < 5:

        raise ValueError(
            f"Not enough groups for CV: "
            f"{n_groups}"
        )

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
                "Sentinel group leakage "
                "detected!"
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

            "model":
                model_name,

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

    fold_df = pd.DataFrame(
        fold_results
    )

    summary = {

        "model":
            model_name,

        "rows":
            len(data),

        "sentinel_groups":
            n_groups,

        "overall_MAE":
            overall_mae,

        "overall_RMSE":
            overall_rmse,

        "overall_R2":
            overall_r2,

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

    return summary, fold_results


# ============================================================
# COMPARE ALL-PIXEL VS WATER-ONLY
# ============================================================

print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


# ------------------------------------------------------------
# ALL-PIXEL MODEL
# ------------------------------------------------------------

print()
print("ALL-PIXEL COMPACT_8")

all_pixel_df = df.dropna(
    subset=ALL_PIXEL_FEATURES + [
        TARGET,
        GROUP,
    ]
).copy()

print(
    f"Rows: {len(all_pixel_df)}"
)

print(
    f"Sentinel groups: "
    f"{all_pixel_df[GROUP].nunique()}"
)

all_summary, all_folds = (
    evaluate_model(
        all_pixel_df,
        ALL_PIXEL_FEATURES,
        "All_pixel_Compact_8",
    )
)


# ------------------------------------------------------------
# WATER-ONLY MODEL
# ------------------------------------------------------------

print()
print("WATER-ONLY COMPACT_8")

water_summary, water_folds = (
    evaluate_model(
        water_ml,
        WATERONLY_FEATURES,
        "Water_only_Compact_8",
    )
)


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

all_fold_df = pd.DataFrame(
    all_folds
)

water_fold_df = pd.DataFrame(
    water_folds
)

fold_results_df = pd.concat(
    [
        all_fold_df,
        water_fold_df,
    ],
    ignore_index=True
)

fold_results_df.to_csv(
    CV_RESULTS_FILE,
    index=False
)


# ============================================================
# SAVE SUMMARY COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    [
        all_summary,
        water_summary,
    ]
)

comparison_df[
    "MAE_change_vs_all_pixel"
] = (
    comparison_df["overall_MAE"]
    - all_summary["overall_MAE"]
)

comparison_df[
    "RMSE_change_vs_all_pixel"
] = (
    comparison_df["overall_RMSE"]
    - all_summary["overall_RMSE"]
)

comparison_df[
    "R2_change_vs_all_pixel"
] = (
    comparison_df["overall_R2"]
    - all_summary["overall_R2"]
)

comparison_df.to_csv(
    COMPARISON_FILE,
    index=False
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    comparison_df[
        [
            "model",
            "rows",
            "sentinel_groups",
            "overall_MAE",
            "overall_RMSE",
            "overall_R2",
            "fold_MAE_mean",
            "fold_RMSE_mean",
            "fold_R2_mean",
            "MAE_change_vs_all_pixel",
            "RMSE_change_vs_all_pixel",
            "R2_change_vs_all_pixel",
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

mae_change = (
    water_summary["overall_MAE"]
    - all_summary["overall_MAE"]
)

rmse_change = (
    water_summary["overall_RMSE"]
    - all_summary["overall_RMSE"]
)

r2_change = (
    water_summary["overall_R2"]
    - all_summary["overall_R2"]
)


if mae_change < 0:

    print(
        f"Water-only MAE improved by "
        f"{abs(mae_change):.3f} NTU."
    )

else:

    print(
        f"Water-only MAE increased by "
        f"{abs(mae_change):.3f} NTU."
    )


if rmse_change < 0:

    print(
        f"Water-only RMSE improved by "
        f"{abs(rmse_change):.3f} NTU."
    )

else:

    print(
        f"Water-only RMSE increased by "
        f"{abs(rmse_change):.3f} NTU."
    )


if r2_change > 0:

    print(
        f"Water-only R² improved by "
        f"{r2_change:.3f}."
    )

else:

    print(
        f"Water-only R² decreased by "
        f"{abs(r2_change):.3f}."
    )


print()
print(
    "IMPORTANT:"
)

print(
    "Do not replace the existing model yet."
)

print(
    "This is an experimental comparison."
)

print()
print(
    f"Saved:"
)

print(
    WATERONLY_DATA_FILE
)

print(
    CV_RESULTS_FILE
)

print(
    COMPARISON_FILE
)

print()
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)