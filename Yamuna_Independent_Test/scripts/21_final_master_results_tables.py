
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 21
# FINAL MASTER RESULTS TABLES — CORRECTED
#
# Purpose:
#   Assemble the already-validated:
#   1. Etawah development / GroupKFold results
#   2. Independent temporal 2024 test
#   3. Independent spatial 8-station test
#   4. Spatial diagnostic findings
#
# IMPORTANT:
#   - NO model retraining
#   - NO test observations added to training
#   - NO prediction modification
#   - NO tuning
#
# This version automatically detects observed/predicted
# turbidity column names in the temporal prediction file.
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"D:\Etawah_Water_Quality"
)

PROJECT = (
    ROOT /
    "Yamuna_Independent_Test"
)

RESULTS = PROJECT / "results"

OUTPUT = (
    RESULTS /
    "final_master_results"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

INPUTS = {

    # --------------------------------------------------------
    # TEMPORAL TEST
    # --------------------------------------------------------

    "temporal_predictions":
        RESULTS /
        "temporal_2024_frozen_model_predictions.csv",

    "temporal_metrics":
        RESULTS /
        "temporal_2024_frozen_model_metrics.csv",

    "temporal_categories":
        RESULTS /
        "temporal_2024_category_performance.csv",

    "temporal_cloud":
        RESULTS /
        "temporal_2024_cloud_performance.csv",

    "temporal_gap":
        RESULTS /
        "temporal_2024_timegap_performance.csv",

    # --------------------------------------------------------
    # SPATIAL TEST
    # --------------------------------------------------------

    "spatial_predictions":
        RESULTS /
        "expanded_spatial_frozen_model_predictions.csv",

    "spatial_overall":
        RESULTS /
        "final_spatial_validation_overall_summary.csv",

    "spatial_station":
        RESULTS /
        "final_spatial_station_performance.csv",

    "spatial_categories":
        RESULTS /
        "final_spatial_turbidity_category_performance.csv",

    "spatial_extremes":
        RESULTS /
        "final_spatial_extreme_events.csv",

    "spatial_cloud":
        RESULTS /
        "final_spatial_cloud_sensitivity.csv",

    "spatial_gap":
        RESULTS /
        "final_spatial_temporal_gap_sensitivity.csv",

    "spatial_b8":
        RESULTS /
        "final_spatial_B8_quality_summary.csv",

    # --------------------------------------------------------
    # SPATIAL DIAGNOSTICS
    # --------------------------------------------------------

    "spatial_regime":
        RESULTS /
        "spatial_transfer_turbidity_regime_diagnostics.csv",

    "spatial_underprediction":
        RESULTS /
        "spatial_transfer_underprediction_by_turbidity.csv",

    "spatial_feature_shift":
        RESULTS /
        "spatial_feature_distribution_shift.csv",

    "spatial_station_shift":
        RESULTS /
        "spatial_station_feature_shift.csv",

    "spatial_extreme_burden":
        RESULTS /
        "spatial_station_extreme_burden.csv",

    "spatial_high_turbidity_cloud":
        RESULTS /
        "spatial_high_turbidity_cloud_diagnostics.csv",
}


# ============================================================
# HELPERS
# ============================================================

def read_csv(name):

    path = INPUTS[name]

    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    print(
        f"Loading {name}: {path.name}"
    )

    return pd.read_csv(path)


def find_column(
    df,
    candidates,
    required=True
):

    # Exact matches first
    for col in candidates:

        if col in df.columns:
            return col

    # Case-insensitive matches
    lower_map = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for col in candidates:

        key = str(col).strip().lower()

        if key in lower_map:
            return lower_map[key]

    if required:

        raise KeyError(
            "\nCould not find any of these columns:\n"
            + "\n".join(
                f"  - {c}"
                for c in candidates
            )
            + "\n\nAvailable columns:\n"
            + "\n".join(
                f"  - {c}"
                for c in df.columns
            )
        )

    return None


def numeric_series(
    df,
    column
):

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def calculate_basic_metrics(
    observed,
    predicted
):

    observed = np.asarray(
        observed,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    mask = (
        np.isfinite(observed)
        &
        np.isfinite(predicted)
    )

    observed = observed[mask]
    predicted = predicted[mask]

    if len(observed) == 0:

        return {
            "n": 0,
            "MAE_NTU": np.nan,
            "RMSE_NTU": np.nan,
            "R2": np.nan,
            "bias_observed_minus_predicted_NTU": np.nan,
            "Pearson_r": np.nan,
            "observed_mean_NTU": np.nan,
            "predicted_mean_NTU": np.nan,
        }

    residual = (
        observed -
        predicted
    )

    mae = np.mean(
        np.abs(residual)
    )

    rmse = np.sqrt(
        np.mean(
            residual ** 2
        )
    )

    bias = residual.mean()

    if len(observed) > 1:

        ss_res = np.sum(
            residual ** 2
        )

        ss_tot = np.sum(
            (
                observed -
                observed.mean()
            ) ** 2
        )

        if ss_tot > 0:

            r2 = (
                1 -
                ss_res /
                ss_tot
            )

        else:

            r2 = np.nan

        if (
            np.std(observed) > 0
            and
            np.std(predicted) > 0
        ):

            pearson_r = np.corrcoef(
                observed,
                predicted
            )[0, 1]

        else:

            pearson_r = np.nan

    else:

        r2 = np.nan
        pearson_r = np.nan

    return {
        "n": len(observed),
        "MAE_NTU": mae,
        "RMSE_NTU": rmse,
        "R2": r2,
        "bias_observed_minus_predicted_NTU": bias,
        "Pearson_r": pearson_r,
        "observed_mean_NTU":
            observed.mean(),
        "predicted_mean_NTU":
            predicted.mean(),
    }


def get_first_numeric_value(
    df,
    candidates
):

    col = find_column(
        df,
        candidates,
        required=False
    )

    if col is None:
        return np.nan

    values = pd.to_numeric(
        df[col],
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        return np.nan

    return values.iloc[0]


# ============================================================
# START
# ============================================================

print("=" * 75)
print(
    "STEP 21 — FINAL MASTER RESULTS TABLES"
)
print("=" * 75)


# ============================================================
# LOAD FILES
# ============================================================

temporal_pred = read_csv(
    "temporal_predictions"
)

temporal_metrics = read_csv(
    "temporal_metrics"
)

temporal_categories = read_csv(
    "temporal_categories"
)

temporal_cloud = read_csv(
    "temporal_cloud"
)

temporal_gap = read_csv(
    "temporal_gap"
)

spatial_pred = read_csv(
    "spatial_predictions"
)

spatial_overall = read_csv(
    "spatial_overall"
)

spatial_station = read_csv(
    "spatial_station"
)

spatial_categories = read_csv(
    "spatial_categories"
)

spatial_extremes = read_csv(
    "spatial_extremes"
)

spatial_cloud = read_csv(
    "spatial_cloud"
)

spatial_gap = read_csv(
    "spatial_gap"
)

spatial_b8 = read_csv(
    "spatial_b8"
)

spatial_regime = read_csv(
    "spatial_regime"
)

spatial_underprediction = read_csv(
    "spatial_underprediction"
)

spatial_feature_shift = read_csv(
    "spatial_feature_shift"
)

spatial_station_shift = read_csv(
    "spatial_station_shift"
)

spatial_extreme_burden = read_csv(
    "spatial_extreme_burden"
)

spatial_high_turbidity_cloud = read_csv(
    "spatial_high_turbidity_cloud"
)


# ============================================================
# INSPECT TEMPORAL COLUMNS
# ============================================================

print("\n" + "=" * 75)
print(
    "TEMPORAL PREDICTION COLUMN DETECTION"
)
print("=" * 75)

print(
    "\nTemporal prediction columns:"
)

for col in temporal_pred.columns:
    print(
        f"  {col}"
    )


# ============================================================
# DETECT TEMPORAL OBSERVED / PREDICTED COLUMNS
# ============================================================

temporal_observed_col = find_column(
    temporal_pred,
    [
        "Turbidity (NTU)",
        "turbidity_NTU",
        "turbidity",
        "observed_turbidity_NTU",
        "observed_turbidity",
        "observed_NTU",
        "observed",
        "actual_turbidity_NTU",
    ]
)

temporal_predicted_col = find_column(
    temporal_pred,
    [
        "predicted_turbidity_NTU",
        "predicted_turbidity",
        "prediction_NTU",
        "prediction",
        "predicted_NTU",
        "predicted",
    ]
)

print(
    "\nDetected temporal observed column:",
    temporal_observed_col
)

print(
    "Detected temporal predicted column:",
    temporal_predicted_col
)


# ============================================================
# TEMPORAL NUMERIC SERIES
# ============================================================

temporal_observed = numeric_series(
    temporal_pred,
    temporal_observed_col
)

temporal_predicted = numeric_series(
    temporal_pred,
    temporal_predicted_col
)

temporal_valid = pd.DataFrame({
    "observed_NTU":
        temporal_observed,
    "predicted_NTU":
        temporal_predicted,
})

temporal_valid = temporal_valid.dropna()

print(
    "\nValid temporal prediction pairs:",
    len(temporal_valid)
)


# ============================================================
# DETECT SPATIAL OBSERVED / PREDICTED COLUMNS
# ============================================================

spatial_observed_col = find_column(
    spatial_pred,
    [
        "Turbidity (NTU)",
        "turbidity_NTU",
        "observed_turbidity_NTU",
        "observed_turbidity",
    ]
)

spatial_predicted_col = find_column(
    spatial_pred,
    [
        "predicted_turbidity_NTU",
        "predicted_turbidity",
        "prediction_NTU",
        "prediction",
    ]
)

print(
    "\nDetected spatial observed column:",
    spatial_observed_col
)

print(
    "Detected spatial predicted column:",
    spatial_predicted_col
)


# ============================================================
# TABLE 1
# STUDY VALIDATION DESIGN
# ============================================================

print(
    "\nCreating Table 1..."
)

design_df = pd.DataFrame([

    {
        "stage":
            "Development",
        "location":
            "Etawah",
        "role":
            "Model development and grouped cross-validation",
        "observations":
            199,
        "stations":
            1,
        "sentinel2_groups":
            99,
        "independence":
            "5-fold GroupKFold by Sentinel-2 scene",
        "status":
            "Development model frozen",
    },

    {
        "stage":
            "Temporal independent test",
        "location":
            "Etawah",
        "role":
            "Unseen 2024 observations",
        "observations":
            len(temporal_valid),
        "stations":
            1,
        "sentinel2_groups":
            14,
        "independence":
            "2021–2023 training; 2024 testing",
        "status":
            "Independent test",
    },

    {
        "stage":
            "Spatial independent test",
        "location":
            "8 additional Yamuna stations",
        "role":
            "Unseen spatial transfer",
        "observations":
            len(
                spatial_pred
            ),
        "stations":
            spatial_pred[
                "Station"
            ].nunique(),
        "sentinel2_groups":
            spatial_pred[
                "sentinel2_id"
            ].nunique(),
        "independence":
            "Etawah observations excluded from spatial test",
        "status":
            "Independent test",
    },

])

design_df.to_csv(
    OUTPUT /
    "01_study_validation_design.csv",
    index=False
)


# ============================================================
# TABLE 2
# FINAL VALIDATION PERFORMANCE
# ============================================================

print(
    "Creating Table 2..."
)

# ---- Temporal metrics calculated directly from
#      the actual frozen prediction file ----

temporal_calc = calculate_basic_metrics(
    temporal_valid[
        "observed_NTU"
    ],
    temporal_valid[
        "predicted_NTU"
    ]
)

# ---- Spatial metrics calculated directly from
#      frozen spatial prediction file ----

spatial_observed = numeric_series(
    spatial_pred,
    spatial_observed_col
)

spatial_predicted = numeric_series(
    spatial_pred,
    spatial_predicted_col
)

spatial_valid = pd.DataFrame({
    "observed_NTU":
        spatial_observed,
    "predicted_NTU":
        spatial_predicted,
})

spatial_valid = spatial_valid.dropna()

spatial_calc = calculate_basic_metrics(
    spatial_valid[
        "observed_NTU"
    ],
    spatial_valid[
        "predicted_NTU"
    ]
)


final_validation_df = pd.DataFrame([

    {
        "validation":
            "Etawah temporal independent test — 2024",
        **temporal_calc,
    },

    {
        "validation":
            "Yamuna spatial independent test — 8 stations",
        **spatial_calc,
    },

])

final_validation_df.to_csv(
    OUTPUT /
    "02_final_validation_performance.csv",
    index=False
)

print(
    "\nFinal validation performance:"
)

print(
    final_validation_df.round(4)
    .to_string(index=False)
)


# ============================================================
# VERIFY KNOWN FROZEN RESULTS
# ============================================================

print(
    "\nVerifying frozen results..."
)

# Temporal expected values
assert abs(
    temporal_calc["MAE_NTU"]
    - 26.269
) < 0.01, (
    "Temporal MAE does not match "
    "the frozen 2024 result."
)

assert abs(
    temporal_calc["RMSE_NTU"]
    - 35.086
) < 0.01, (
    "Temporal RMSE does not match "
    "the frozen 2024 result."
)

assert abs(
    temporal_calc["R2"]
    - 0.064
) < 0.01, (
    "Temporal R² does not match "
    "the frozen 2024 result."
)

assert abs(
    temporal_calc[
        "bias_observed_minus_predicted_NTU"
    ]
    - (-19.140)
) < 0.02, (
    "Temporal bias does not match "
    "the frozen 2024 result."
)

# Spatial expected values
assert len(
    spatial_valid
) == 812, (
    "Spatial valid row count changed."
)

assert abs(
    spatial_calc["MAE_NTU"]
    - 68.647
) < 0.01, (
    "Spatial MAE does not match "
    "the frozen result."
)

assert abs(
    spatial_calc["RMSE_NTU"]
    - 157.876
) < 0.01, (
    "Spatial RMSE does not match "
    "the frozen result."
)

assert abs(
    spatial_calc["R2"]
    - (-0.107)
) < 0.01, (
    "Spatial R² does not match "
    "the frozen result."
)


# ============================================================
# TABLE 3
# TEMPORAL CATEGORY PERFORMANCE
# ============================================================

print(
    "Creating Table 3..."
)

temporal_categories.to_csv(
    OUTPUT /
    "03_temporal_2024_category_performance.csv",
    index=False
)


# ============================================================
# TABLE 4
# SPATIAL CATEGORY PERFORMANCE
# ============================================================

print(
    "Creating Table 4..."
)

spatial_categories.to_csv(
    OUTPUT /
    "04_spatial_category_performance.csv",
    index=False
)


# ============================================================
# TABLE 5
# SPATIAL STATION PERFORMANCE
# ============================================================

print(
    "Creating Table 5..."
)

spatial_station.sort_values(
    "MAE_NTU"
).to_csv(
    OUTPUT /
    "05_spatial_station_performance.csv",
    index=False
)


# ============================================================
# TABLE 6
# TEMPORAL SENSITIVITY
# ============================================================

print(
    "Creating Table 6..."
)

temporal_cloud_export = (
    temporal_cloud.copy()
)

temporal_cloud_export.insert(
    0,
    "analysis",
    "Cloud sensitivity"
)

temporal_gap_export = (
    temporal_gap.copy()
)

temporal_gap_export.insert(
    0,
    "analysis",
    "Temporal-gap sensitivity"
)

pd.concat(
    [
        temporal_cloud_export,
        temporal_gap_export,
    ],
    ignore_index=True
).to_csv(
    OUTPUT /
    "06_temporal_sensitivity.csv",
    index=False
)


# ============================================================
# TABLE 7
# SPATIAL SENSITIVITY
# ============================================================

print(
    "Creating Table 7..."
)

spatial_cloud_export = (
    spatial_cloud.copy()
)

spatial_cloud_export.insert(
    0,
    "analysis",
    "Cloud sensitivity"
)

spatial_gap_export = (
    spatial_gap.copy()
)

spatial_gap_export.insert(
    0,
    "analysis",
    "Temporal-gap sensitivity"
)

spatial_b8_export = (
    spatial_b8.copy()
)

spatial_b8_export.insert(
    0,
    "analysis",
    "B8 quality sensitivity"
)

pd.concat(
    [
        spatial_cloud_export,
        spatial_gap_export,
        spatial_b8_export,
    ],
    ignore_index=True
).to_csv(
    OUTPUT /
    "07_spatial_sensitivity.csv",
    index=False
)


# ============================================================
# TABLE 8
# EXTREME TURBIDITY SUMMARY
# ============================================================

print(
    "Creating Table 8..."
)

spatial_extreme_mask = (
    spatial_valid[
        "observed_NTU"
    ] >= 100
)

spatial_extreme_valid = (
    spatial_valid.loc[
        spatial_extreme_mask
    ]
)

spatial_extreme_metrics = (
    calculate_basic_metrics(
        spatial_extreme_valid[
            "observed_NTU"
        ],
        spatial_extreme_valid[
            "predicted_NTU"
        ]
    )
)

spatial_ge500 = int(
    (
        spatial_valid[
            "observed_NTU"
        ] >= 500
    ).sum()
)

temporal_ge100 = int(
    (
        temporal_valid[
            "observed_NTU"
        ] >= 100
    ).sum()
)

temporal_ge500 = int(
    (
        temporal_valid[
            "observed_NTU"
        ] >= 500
    ).sum()
)

temporal_extreme_valid = (
    temporal_valid[
        temporal_valid[
            "observed_NTU"
        ] >= 100
    ]
)

temporal_extreme_metrics = (
    calculate_basic_metrics(
        temporal_extreme_valid[
            "observed_NTU"
        ],
        temporal_extreme_valid[
            "predicted_NTU"
        ]
    )
)

extreme_summary = pd.DataFrame([

    {
        "validation":
            "Temporal 2024",
        "n_ge_100":
            temporal_ge100,
        "n_ge_500":
            temporal_ge500,
        "MAE_ge_100_NTU":
            temporal_extreme_metrics[
                "MAE_NTU"
            ],
        "RMSE_ge_100_NTU":
            temporal_extreme_metrics[
                "RMSE_NTU"
            ],
        "bias_ge_100_NTU":
            temporal_extreme_metrics[
                "bias_observed_minus_predicted_NTU"
            ],
    },

    {
        "validation":
            "Spatial 8-station test",
        "n_ge_100":
            int(
                spatial_extreme_mask.sum()
            ),
        "n_ge_500":
            spatial_ge500,
        "MAE_ge_100_NTU":
            spatial_extreme_metrics[
                "MAE_NTU"
            ],
        "RMSE_ge_100_NTU":
            spatial_extreme_metrics[
                "RMSE_NTU"
            ],
        "bias_ge_100_NTU":
            spatial_extreme_metrics[
                "bias_observed_minus_predicted_NTU"
            ],
    },

])

extreme_summary.to_csv(
    OUTPUT /
    "08_extreme_turbidity_summary.csv",
    index=False
)


# ============================================================
# TABLE 9
# SPATIAL EXTREME EVENT DETAILS
# ============================================================

print(
    "Creating Table 9..."
)

spatial_extremes.to_csv(
    OUTPUT /
    "09_spatial_extreme_event_details.csv",
    index=False
)


# ============================================================
# TABLE 10
# STATION EXTREME BURDEN
# ============================================================

print(
    "Creating Table 10..."
)

spatial_extreme_burden.to_csv(
    OUTPUT /
    "10_station_extreme_turbidity_burden.csv",
    index=False
)


# ============================================================
# TABLE 11
# FEATURE DISTRIBUTION SHIFT
# ============================================================

print(
    "Creating Table 11..."
)

spatial_feature_shift.to_csv(
    OUTPUT /
    "11_feature_distribution_shift.csv",
    index=False
)


# ============================================================
# TABLE 12
# STATION FEATURE SHIFT
# ============================================================

print(
    "Creating Table 12..."
)

spatial_station_shift.to_csv(
    OUTPUT /
    "12_station_feature_shift.csv",
    index=False
)


# ============================================================
# TABLE 13
# HIGH TURBIDITY / CLOUD DIAGNOSTIC
# ============================================================

print(
    "Creating Table 13..."
)

spatial_high_turbidity_cloud.to_csv(
    OUTPUT /
    "13_high_turbidity_cloud_diagnostic.csv",
    index=False
)


# ============================================================
# TABLE 14
# SPATIAL UNDERPREDICTION BY TURBIDITY
# ============================================================

print(
    "Creating Table 14..."
)

spatial_underprediction.to_csv(
    OUTPUT /
    "14_spatial_underprediction_by_turbidity.csv",
    index=False
)


# ============================================================
# TABLE 15
# FINAL STUDY FINDINGS
# ============================================================

print(
    "Creating Table 15..."
)

findings = [

    {
        "finding_id": 1,
        "domain":
            "Development",
        "finding":
            "The Compact-8 Gradient Boosting model "
            "provided the frozen Etawah development baseline "
            "under grouped Sentinel-2 cross-validation.",
        "interpretation":
            "The model is suitable as the development-site "
            "baseline but does not establish Yamuna-wide "
            "transferability.",
    },

    {
        "finding_id": 2,
        "domain":
            "Temporal transfer",
        "finding":
            "The frozen model retained a positive association "
            "with independent 2024 Etawah observations.",
        "interpretation":
            "Temporal transfer shows potential, but the "
            "2024 sample is small and absolute calibration "
            "remains limited.",
    },

    {
        "finding_id": 3,
        "domain":
            "Spatial transfer",
        "finding":
            "Performance declined substantially on eight "
            "unseen Yamuna stations.",
        "interpretation":
            "The Etawah-trained model should not be treated "
            "as a Yamuna-wide transferable model.",
    },

    {
        "finding_id": 4,
        "domain":
            "Extreme turbidity",
        "finding":
            "Spatial error increased dramatically above "
            "100 NTU.",
        "interpretation":
            "The current Compact-8 formulation does not "
            "capture the extreme turbidity regime reliably.",
    },

    {
        "finding_id": 5,
        "domain":
            "Cloud sensitivity",
        "finding":
            "Clearer scenes generally reduced absolute error, "
            "but substantial errors remained under low cloud.",
        "interpretation":
            "Cloud contamination contributes to error but "
            "does not explain the spatial transfer failure.",
    },

    {
        "finding_id": 6,
        "domain":
            "Temporal matching",
        "finding":
            "Tightening the CWC–Sentinel-2 temporal gap did "
            "not eliminate spatial prediction error.",
        "interpretation":
            "Temporal mismatch is not the dominant explanation "
            "for poor spatial transfer.",
    },

    {
        "finding_id": 7,
        "domain":
            "Radiometric quality",
        "finding":
            "Only a small fraction of spatial test observations "
            "had B8 mean values above 1.",
        "interpretation":
            "The B8 > 1 issue is not sufficient to explain "
            "the overall spatial transfer failure.",
    },

]

pd.DataFrame(
    findings
).to_csv(
    OUTPUT /
    "15_final_study_findings.csv",
    index=False
)


# ============================================================
# TABLE 16
# FINAL NUMERICAL SUMMARY
# ============================================================

print(
    "Creating Table 16..."
)

spatial_b8_count = np.nan

if "B8_mean" in spatial_pred.columns:

    spatial_b8_count = int(
        (
            pd.to_numeric(
                spatial_pred[
                    "B8_mean"
                ],
                errors="coerce"
            ) > 1
        ).sum()
    )

final_summary = {

    "development_training_rows":
        199,

    "development_sentinel_groups":
        99,

    "temporal_test_rows":
        temporal_calc["n"],

    "temporal_test_year":
        2024,

    "temporal_MAE_NTU":
        temporal_calc["MAE_NTU"],

    "temporal_RMSE_NTU":
        temporal_calc["RMSE_NTU"],

    "temporal_R2":
        temporal_calc["R2"],

    "temporal_bias_NTU":
        temporal_calc[
            "bias_observed_minus_predicted_NTU"
        ],

    "temporal_Pearson_r":
        temporal_calc["Pearson_r"],

    "spatial_test_rows":
        spatial_calc["n"],

    "spatial_stations":
        spatial_pred[
            "Station"
        ].nunique(),

    "spatial_unique_sentinel2_images":
        spatial_pred[
            "sentinel2_id"
        ].nunique(),

    "spatial_MAE_NTU":
        spatial_calc["MAE_NTU"],

    "spatial_RMSE_NTU":
        spatial_calc["RMSE_NTU"],

    "spatial_R2":
        spatial_calc["R2"],

    "spatial_bias_NTU":
        spatial_calc[
            "bias_observed_minus_predicted_NTU"
        ],

    "spatial_Pearson_r":
        spatial_calc["Pearson_r"],

    "spatial_n_ge_100_NTU":
        int(
            (
                spatial_valid[
                    "observed_NTU"
                ] >= 100
            ).sum()
        ),

    "spatial_n_ge_500_NTU":
        int(
            (
                spatial_valid[
                    "observed_NTU"
                ] >= 500
            ).sum()
        ),

    "spatial_B8_gt_1_count":
        spatial_b8_count,

    "model_retrained_during_validation":
        "NO",

    "test_data_added_to_training":
        "NO",

    "predictions_modified_after_freeze":
        "NO",
}

pd.DataFrame(
    [final_summary]
).to_csv(
    OUTPUT /
    "16_final_study_numerical_summary.csv",
    index=False
)


# ============================================================
# TABLE 17
# PUBLICATION-READY METRIC TABLE
# ============================================================

print(
    "Creating Table 17..."
)

publication_df = pd.DataFrame([

    {
        "Validation stage":
            "Etawah development",
        "Dataset role":
            "Grouped cross-validation",
        "N":
            199,
        "MAE (NTU)":
            22.954,
        "RMSE (NTU)":
            41.599,
        "R²":
            0.236,
        "Pearson r":
            np.nan,
        "Bias (NTU)":
            np.nan,
    },

    {
        "Validation stage":
            "Etawah temporal test",
        "Dataset role":
            "Independent 2024",
        "N":
            temporal_calc["n"],
        "MAE (NTU)":
            temporal_calc["MAE_NTU"],
        "RMSE (NTU)":
            temporal_calc["RMSE_NTU"],
        "R²":
            temporal_calc["R2"],
        "Pearson r":
            temporal_calc["Pearson_r"],
        "Bias (NTU)":
            temporal_calc[
                "bias_observed_minus_predicted_NTU"
            ],
    },

    {
        "Validation stage":
            "Yamuna spatial test",
        "Dataset role":
            "8 unseen stations",
        "N":
            spatial_calc["n"],
        "MAE (NTU)":
            spatial_calc["MAE_NTU"],
        "RMSE (NTU)":
            spatial_calc["RMSE_NTU"],
        "R²":
            spatial_calc["R2"],
        "Pearson r":
            spatial_calc["Pearson_r"],
        "Bias (NTU)":
            spatial_calc[
                "bias_observed_minus_predicted_NTU"
            ],
    },

])

publication_df.to_csv(
    OUTPUT /
    "17_publication_ready_validation_metrics.csv",
    index=False
)


# ============================================================
# TABLE 18
# MODEL / VALIDATION INTEGRITY
# ============================================================

print(
    "Creating Table 18..."
)

integrity_df = pd.DataFrame([

    {
        "check":
            "Development model retrained during testing",
        "result":
            "NO",
    },

    {
        "check":
            "2024 temporal observations added to training",
        "result":
            "NO",
    },

    {
        "check":
            "Spatial test observations added to training",
        "result":
            "NO",
    },

    {
        "check":
            "Spatial predictions modified after Step 18",
        "result":
            "NO",
    },

    {
        "check":
            "Temporal matching independently verified",
        "result":
            "YES",
    },

    {
        "check":
            "Spatial matching independently verified",
        "result":
            "YES",
    },

    {
        "check":
            "Spatial test prediction count",
        "result":
            str(
                spatial_calc["n"]
            ),
    },

    {
        "check":
            "Independent spatial stations",
        "result":
            str(
                spatial_pred[
                    "Station"
                ].nunique()
            ),
    },

])

integrity_df.to_csv(
    OUTPUT /
    "18_validation_integrity.csv",
    index=False
)


# ============================================================
# MASTER INDEX
# ============================================================

print(
    "Creating master table index..."
)

table_files = sorted(
    OUTPUT.glob("*.csv")
)

index_df = pd.DataFrame({

    "table_file":
        [
            f.name
            for f in table_files
        ],

    "full_path":
        [
            str(f)
            for f in table_files
        ],

})

index_df.to_csv(
    OUTPUT /
    "00_master_table_index.csv",
    index=False
)


# ============================================================
# FINAL CHECKS
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "FINAL CONSISTENCY CHECKS"
)

print(
    "=" * 75
)


# Temporal
assert temporal_calc["n"] == 14

assert abs(
    temporal_calc["MAE_NTU"]
    - 26.269
) < 0.01

assert abs(
    temporal_calc["RMSE_NTU"]
    - 35.086
) < 0.01

assert abs(
    temporal_calc["R2"]
    - 0.064
) < 0.01


# Spatial
assert spatial_calc["n"] == 812

assert spatial_pred[
    "Station"
].nunique() == 8

assert spatial_pred[
    "sentinel2_id"
].nunique() == 521

assert abs(
    spatial_calc["MAE_NTU"]
    - 68.647
) < 0.01

assert abs(
    spatial_calc["RMSE_NTU"]
    - 157.876
) < 0.01

assert abs(
    spatial_calc["R2"]
    - (-0.107)
) < 0.01

assert (
    spatial_valid[
        "observed_NTU"
    ] >= 100
).sum() == 158

assert (
    spatial_valid[
        "observed_NTU"
    ] >= 500
).sum() == 31


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "STEP 21 COMPLETE"
)

print(
    "=" * 75
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT
)

print(
    "\nTables created:",
    len(
        list(
            OUTPUT.glob("*.csv")
        )
    )
)

print(
    "\nTemporal independent test:"
)

print(
    f"  N = {temporal_calc['n']}"
)

print(
    f"  MAE = "
    f"{temporal_calc['MAE_NTU']:.3f} NTU"
)

print(
    f"  RMSE = "
    f"{temporal_calc['RMSE_NTU']:.3f} NTU"
)

print(
    f"  R² = "
    f"{temporal_calc['R2']:.3f}"
)

print(
    f"  Pearson r = "
    f"{temporal_calc['Pearson_r']:.3f}"
)

print(
    "\nSpatial independent test:"
)

print(
    f"  N = {spatial_calc['n']}"
)

print(
    f"  Stations = "
    f"{spatial_pred['Station'].nunique()}"
)

print(
    f"  Unique Sentinel-2 scenes = "
    f"{spatial_pred['sentinel2_id'].nunique()}"
)

print(
    f"  MAE = "
    f"{spatial_calc['MAE_NTU']:.3f} NTU"
)

print(
    f"  RMSE = "
    f"{spatial_calc['RMSE_NTU']:.3f} NTU"
)

print(
    f"  R² = "
    f"{spatial_calc['R2']:.3f}"
)

print(
    f"  Pearson r = "
    f"{spatial_calc['Pearson_r']:.3f}"
)

print(
    "\nValidation integrity:"
)

print(
    "  Model retrained = NO"
)

print(
    "  Test observations added to training = NO"
)

print(
    "  Predictions modified = NO"
)

print(
    "\nSTEP 21 COMPLETE."
)
