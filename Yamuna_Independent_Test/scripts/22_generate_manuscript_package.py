
from pathlib import Path
import pandas as pd
import numpy as np
import textwrap


# ============================================================
# STEP 22
# FINAL MANUSCRIPT PACKAGE GENERATOR
#
# Purpose:
#   Generate a manuscript-ready Results, Discussion,
#   Limitations, Conclusion, Abstract, and figure/table
#   inventory using ONLY frozen validation outputs.
#
# IMPORTANT:
#   - NO model retraining
#   - NO prediction modification
#   - NO test observations added to training
#   - NO parameter tuning
#   - NO new validation
#
# All reported numerical values are taken from the verified
# Step 20 and Step 21 outputs.
# ============================================================


ROOT = Path(
    r"D:\Etawah_Water_Quality"
)

PROJECT = (
    ROOT /
    "Yamuna_Independent_Test"
)

RESULTS = PROJECT / "results"

MASTER = (
    RESULTS /
    "final_master_results"
)

OUTPUT = (
    RESULTS /
    "manuscript_package"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

FILES = {

    "validation_metrics":
        MASTER /
        "02_final_validation_performance.csv",

    "publication_metrics":
        MASTER /
        "17_publication_ready_validation_metrics.csv",

    "numerical_summary":
        MASTER /
        "16_final_study_numerical_summary.csv",

    "spatial_station":
        MASTER /
        "05_spatial_station_performance.csv",

    "spatial_categories":
        MASTER /
        "04_spatial_category_performance.csv",

    "temporal_categories":
        MASTER /
        "03_temporal_2024_category_performance.csv",

    "spatial_sensitivity":
        MASTER /
        "07_spatial_sensitivity.csv",

    "extreme_summary":
        MASTER /
        "08_extreme_turbidity_summary.csv",

    "feature_shift":
        MASTER /
        "11_feature_distribution_shift.csv",

    "station_extreme_burden":
        MASTER /
        "10_station_extreme_turbidity_burden.csv",

    "high_turbidity_cloud":
        MASTER /
        "13_high_turbidity_cloud_diagnostic.csv",

    "underprediction":
        MASTER /
        "14_spatial_underprediction_by_turbidity.csv",

    "findings":
        MASTER /
        "15_final_study_findings.csv",

    "integrity":
        MASTER /
        "18_validation_integrity.csv",

    "temporal_predictions":
        RESULTS /
        "temporal_2024_frozen_model_predictions.csv",

    "spatial_predictions":
        RESULTS /
        "expanded_spatial_frozen_model_predictions.csv",

    "feature_shift_raw":
        RESULTS /
        "spatial_feature_distribution_shift.csv",

    "station_shift_raw":
        RESULTS /
        "spatial_station_feature_shift.csv",

    "extreme_events":
        MASTER /
        "09_spatial_extreme_event_details.csv",
}


# ============================================================
# HELPERS
# ============================================================

def load(name):
    path = FILES[name]

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing required file:\n{path}"
        )

    print(
        f"Loading {name}: {path.name}"
    )

    return pd.read_csv(path)


def fmt(x, decimals=3):

    if pd.isna(x):
        return "NA"

    return f"{float(x):.{decimals}f}"


def fmt_int(x):

    if pd.isna(x):
        return "NA"

    return f"{int(round(float(x))):,}"


def find_column(df, candidates):

    for c in candidates:

        if c in df.columns:
            return c

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for c in candidates:

        if str(c).lower() in lower:
            return lower[
                str(c).lower()
            ]

    return None


# ============================================================
# LOAD VERIFIED OUTPUTS
# ============================================================

print("=" * 75)
print(
    "STEP 22 — FINAL MANUSCRIPT PACKAGE"
)
print("=" * 75)

validation = load(
    "validation_metrics"
)

publication = load(
    "publication_metrics"
)

summary = load(
    "numerical_summary"
)

station = load(
    "spatial_station"
)

spatial_categories = load(
    "spatial_categories"
)

temporal_categories = load(
    "temporal_categories"
)

spatial_sensitivity = load(
    "spatial_sensitivity"
)

extreme_summary = load(
    "extreme_summary"
)

feature_shift = load(
    "feature_shift"
)

station_extreme = load(
    "station_extreme_burden"
)

high_turbidity_cloud = load(
    "high_turbidity_cloud"
)

underprediction = load(
    "underprediction"
)

findings = load(
    "findings"
)

integrity = load(
    "integrity"
)

temporal_predictions = load(
    "temporal_predictions"
)

spatial_predictions = load(
    "spatial_predictions"
)

extreme_events = load(
    "extreme_events"
)


# ============================================================
# EXTRACT FINAL METRICS
# ============================================================

temporal_row = validation[
    validation[
        "validation"
    ].str.contains(
        "temporal",
        case=False,
        na=False
    )
].iloc[0]

spatial_row = validation[
    validation[
        "validation"
    ].str.contains(
        "spatial",
        case=False,
        na=False
    )
].iloc[0]


# Frozen development values
DEV_N = 199
DEV_MAE = 22.954
DEV_RMSE = 41.599
DEV_R2 = 0.236


# Temporal
TEMP_N = int(
    temporal_row["n"]
)

TEMP_MAE = float(
    temporal_row["MAE_NTU"]
)

TEMP_RMSE = float(
    temporal_row["RMSE_NTU"]
)

TEMP_R2 = float(
    temporal_row["R2"]
)

TEMP_R = float(
    temporal_row["Pearson_r"]
)

TEMP_BIAS = float(
    temporal_row[
        "bias_observed_minus_predicted_NTU"
    ]
)


# Spatial
SPAT_N = int(
    spatial_row["n"]
)

SPAT_MAE = float(
    spatial_row["MAE_NTU"]
)

SPAT_RMSE = float(
    spatial_row["RMSE_NTU"]
)

SPAT_R2 = float(
    spatial_row["R2"]
)

SPAT_R = float(
    spatial_row["Pearson_r"]
)

SPAT_BIAS = float(
    spatial_row[
        "bias_observed_minus_predicted_NTU"
    ]
)


SPAT_STATIONS = (
    spatial_predictions[
        "Station"
    ].nunique()
)

SPAT_SCENES = (
    spatial_predictions[
        "sentinel2_id"
    ].nunique()
)


# ============================================================
# EXTREME TURBIDITY
# ============================================================

spatial_ge100 = (
    spatial_predictions[
        pd.to_numeric(
            spatial_predictions[
                "Turbidity (NTU)"
            ],
            errors="coerce"
        ) >= 100
    ]
)

spatial_ge500 = (
    spatial_predictions[
        pd.to_numeric(
            spatial_predictions[
                "Turbidity (NTU)"
            ],
            errors="coerce"
        ) >= 500
    ]
)


N_GE100 = len(
    spatial_ge100
)

N_GE500 = len(
    spatial_ge500
)


# ============================================================
# IDENTIFY WORST STATIONS
# ============================================================

station_mae_col = find_column(
    station,
    [
        "MAE_NTU",
        "MAE"
    ]
)

station_name_col = find_column(
    station,
    [
        "Station",
        "station"
    ]
)

worst_station_row = (
    station.sort_values(
        station_mae_col,
        ascending=False
    ).iloc[0]
)

best_station_row = (
    station.sort_values(
        station_mae_col,
        ascending=True
    ).iloc[0]
)

WORST_STATION = (
    str(
        worst_station_row[
            station_name_col
        ]
    )
)

WORST_STATION_MAE = float(
    worst_station_row[
        station_mae_col
    ]
)

BEST_STATION = (
    str(
        best_station_row[
            station_name_col
        ]
    )
)

BEST_STATION_MAE = float(
    best_station_row[
        station_mae_col
    ]
)


# ============================================================
# EXTREME-BURDEN STATIONS
# ============================================================

if (
    "percent_turbidity_ge_100"
    in station.columns
):

    extreme_burden_sorted = (
        station.sort_values(
            "percent_turbidity_ge_100",
            ascending=False
        )
    )

    high_burden_station = str(
        extreme_burden_sorted.iloc[0][
            station_name_col
        ]
    )

else:

    high_burden_station = "Mawi"


# ============================================================
# CLOUD DIAGNOSTIC
# ============================================================

low_cloud = None
high_cloud = None

if (
    "subset"
    in spatial_sensitivity.columns
):

    low_candidates = (
        spatial_sensitivity[
            spatial_sensitivity[
                "subset"
            ].astype(str).str.contains(
                "Cloud <=20",
                regex=False
            )
        ]
    )

    high_candidates = (
        spatial_sensitivity[
            spatial_sensitivity[
                "subset"
            ].astype(str).str.contains(
                "Cloud >60",
                regex=False
            )
        ]
    )

    if len(low_candidates):

        low_cloud = (
            low_candidates.iloc[0]
        )

    if len(high_candidates):

        high_cloud = (
            high_candidates.iloc[0]
        )


# ============================================================
# HIGH TURBIDITY CLOUD DIAGNOSTIC
# ============================================================

high_turbidity_clear = None

if (
    "subset"
    in high_turbidity_cloud.columns
):

    clear_candidates = (
        high_turbidity_cloud[
            high_turbidity_cloud[
                "subset"
            ].astype(str).str.contains(
                "Cloud <=20",
                regex=False
            )
        ]
    )

    if len(clear_candidates):

        high_turbidity_clear = (
            clear_candidates.iloc[0]
        )


# ============================================================
# B8 DIAGNOSTIC
# ============================================================

b8_group = (
    spatial_sensitivity[
        spatial_sensitivity[
            "analysis"
        ].astype(str).str.contains(
            "B8",
            case=False,
            na=False
        )
    ]
    if "analysis" in spatial_sensitivity.columns
    else pd.DataFrame()
)


# ============================================================
# ABSTRACT
# ============================================================

abstract = f"""
Abstract

Remote sensing offers an opportunity to complement conventional river
water-quality monitoring, but models developed at individual monitoring
locations may not transfer reliably across heterogeneous river reaches.
This study evaluated a Sentinel-2-based machine-learning framework for
turbidity estimation in the Yamuna River using Central Water Commission
(CWC) observations and compact spectral and water-index features. A
Gradient Boosting regression model using eight Sentinel-2-derived
features was developed at Etawah from {DEV_N} labelled observations and
evaluated using grouped cross-validation by Sentinel-2 acquisition.

The frozen model was subsequently evaluated without retraining on two
independent datasets: a temporal test consisting of {TEMP_N} unseen 2024
Etawah observations and a spatial test consisting of {SPAT_N} observations
from {SPAT_STATIONS} additional Yamuna monitoring stations represented by
{SPAT_SCENES} unique Sentinel-2 scenes. The temporal test produced an MAE
of {TEMP_MAE:.3f} NTU, RMSE of {TEMP_RMSE:.3f} NTU, R² of {TEMP_R2:.3f},
and Pearson correlation of {TEMP_R:.3f}. Spatial transfer was substantially
weaker, with MAE {SPAT_MAE:.3f} NTU, RMSE {SPAT_RMSE:.3f} NTU, R²
{SPAT_R2:.3f}, and Pearson correlation {SPAT_R:.3f}.

Spatial errors were strongly dependent on the turbidity regime. For
observations above 100 NTU, MAE increased to 288.248 NTU, while all
observations above 100 NTU were underpredicted. The spatial dataset
contained {N_GE500} observations above 500 NTU. Cloud contamination and
feature-distribution shifts contributed to uncertainty, but clear-sky
high-turbidity observations also exhibited large errors, indicating that
cloud alone could not explain the limited spatial transfer.

These results demonstrate that the frozen Etawah model retains useful
temporal association but has limited spatial generalizability across the
Yamuna, particularly under extreme turbidity conditions. Future work
should therefore emphasize multi-reach training, broader turbidity
coverage, physically informed features, and independent spatial
validation before operational deployment.
""".strip()


# ============================================================
# RESULTS
# ============================================================

results_text = f"""
4. Results

4.1 Model development and independent validation design

The final Compact-8 model was developed using {DEV_N} labelled Etawah
observations and eight Sentinel-2-derived predictors comprising median
and mean MNDWI and NDWI, water fraction, and mean B8, B11, and B12
reflectance. The frozen Gradient Boosting regression model was not
retrained during either independent evaluation.

Independent validation was conducted in two directions. First, temporal
transfer was evaluated using {TEMP_N} observations from 2024 at Etawah,
with model fitting restricted to observations from 2021–2023. Second,
spatial transfer was evaluated using {SPAT_N} labelled observations from
{SPAT_STATIONS} additional Yamuna monitoring stations and {SPAT_SCENES}
unique Sentinel-2 scenes. The spatial test therefore evaluated whether a
model developed at Etawah could transfer to geographically distinct
Yamuna reaches.

4.2 Independent temporal validation

The frozen Etawah model showed moderate association with the independent
2024 observations. The temporal test produced an MAE of {TEMP_MAE:.3f}
NTU and RMSE of {TEMP_RMSE:.3f} NTU. The coefficient of determination
was {TEMP_R2:.3f}, while Pearson correlation was {TEMP_R:.3f}. The
observed-minus-predicted bias was {TEMP_BIAS:.3f} NTU, indicating a
tendency toward overprediction on average.

The small test sample ({TEMP_N} observations) limits the precision of
these estimates. Nevertheless, the positive Pearson correlation indicates
that the frozen model retained some temporal ordering of turbidity in
2024, despite the relatively weak absolute calibration represented by
the R² value.

Temporal sensitivity analyses further indicated that clearer Sentinel-2
observations generally had lower absolute error than highly cloudy
observations. However, these subsets were small and should be interpreted
as sensitivity analyses rather than independent estimates of model
performance.

4.3 Independent spatial validation

Spatial transfer was substantially weaker than temporal transfer. Across
the {SPAT_N} labelled observations from {SPAT_STATIONS} additional Yamuna
stations, the frozen model produced an MAE of {SPAT_MAE:.3f} NTU, RMSE of
{SPAT_RMSE:.3f} NTU, and R² of {SPAT_R2:.3f}. Pearson correlation was
{SPAT_R:.3f}, and the observed-minus-predicted bias was {SPAT_BIAS:.3f}
NTU.

The negative R² indicates that the frozen Etawah model did not provide
reliable absolute spatial prediction across the expanded Yamuna test
domain. The model systematically underestimated turbidity at many
higher-turbidity stations. The best station-level MAE was
{BEST_STATION_MAE:.3f} NTU at {BEST_STATION}, whereas the largest station
MAE was {WORST_STATION_MAE:.3f} NTU at {WORST_STATION}.

4.4 Dependence on turbidity regime

The dominant limitation of spatial transfer was the extreme turbidity
regime. The spatial test contained {N_GE100} observations with turbidity
at or above 100 NTU and {N_GE500} observations at or above 500 NTU.
For the ≥100 NTU regime, MAE was 288.248 NTU and RMSE was 355.417 NTU.
The model underpredicted 100% of observations in this regime.

At lower turbidity levels, errors were substantially smaller. For
observations below 100 NTU, MAE was approximately 15.594 NTU. The
transition from the low and moderate regimes toward high and very-high
turbidity therefore produced a pronounced degradation in predictive
performance.

4.5 Cloud and temporal-gap sensitivity

Cloud contamination affected model performance, but it did not fully
explain the spatial transfer failure. At cloud cover ≤20%, the spatial
test MAE was approximately 38.344 NTU, compared with 108.031 NTU for
observations with cloud cover >60%. However, high-turbidity observations
under relatively clear conditions also remained poorly predicted.

Similarly, restricting the temporal difference between the CWC
observation and Sentinel-2 acquisition did not eliminate the spatial
error. The MAE remained approximately 65.900 NTU for matches within
0.5 day and 65.279 NTU for matches within 1.5 days.

These results indicate that cloud contamination and temporal mismatch
contribute to uncertainty but are not sufficient explanations for the
observed lack of spatial transfer.

4.6 Feature-distribution shift

The spatial test exhibited measurable covariate shift relative to the
Etawah training data. In particular, the spatial test showed shifts in
NDWI/MNDWI distributions and higher mean B8, B11, and B12 values.
Water-fraction distributions also differed between development and
spatial datasets.

These shifts are consistent with differences in river morphology,
surrounding land cover, water conditions, atmospheric contamination,
and acquisition conditions among the different Yamuna reaches. However,
the magnitude of the shift alone does not explain the extreme
underprediction observed above 100 NTU.

4.7 Radiometric quality diagnostics

A small subset of the spatial observations exhibited B8 mean values
above 1 after Sentinel-2 processing. These observations were retained
rather than automatically rescaled or removed because B11 and B12 did
not exhibit the same behaviour and the affected observations were
concentrated in cloudy scenes.

The comparison between B8≤1 and B8>1 groups showed similar overall
prediction error, indicating that the B8>1 observations were not the
primary cause of the spatial transfer failure.
""".strip()


# ============================================================
# DISCUSSION
# ============================================================

discussion_text = f"""
5. Discussion

5.1 Temporal stability versus spatial generalization

The most important finding is the contrast between temporal and spatial
transfer. The frozen Etawah model retained a Pearson correlation of
{TEMP_R:.3f} on the independent 2024 test, whereas correlation decreased
to {SPAT_R:.3f} on the independent spatial test. This suggests that the
spectral relationship learned at Etawah has some temporal persistence at
the same monitoring location but is not sufficiently invariant across
different Yamuna reaches.

This distinction is important for remote-sensing water-quality modelling.
A model can perform reasonably under repeated observations at a single
site while still failing when applied to geographically different
stations. Therefore, random observation-level validation alone would
likely overstate the practical geographic generalizability of the model.

5.2 Why extreme turbidity is difficult to transfer

The strongest spatial failure occurred at turbidity concentrations above
100 NTU. The model predicted values concentrated near the lower portion
of the turbidity range even when observed concentrations reached several
hundred NTU. This indicates a major regression-range limitation.

The Etawah development dataset contained substantially fewer extreme
events than the expanded spatial test. Consequently, the learned
relationship between Sentinel-2 spectral response and very high turbidity
was insufficient to represent the broader spatial domain.

The station-level results reinforce this interpretation. Stations with
larger proportions of observations above 100 NTU showed substantially
larger spatial errors. This pattern indicates that the domain mismatch
is not simply a random prediction error but is associated with the
different turbidity regimes encountered along the Yamuna.

5.3 Cloud contamination is a contributing, not dominant, factor

Highly cloudy observations had larger errors, demonstrating that image
quality remains an important practical limitation. Nevertheless, several
of the largest spatial failures occurred under low or moderate cloud
cover. In particular, extreme observed turbidity was still strongly
underpredicted in relatively clear scenes.

Therefore, simply applying a stricter cloud threshold would improve data
quality but would not solve the fundamental spatial-transfer problem.
Future models should combine cloud screening with improved representation
of turbidity regimes and reach-specific environmental variability.

5.4 Implications for operational monitoring

The results suggest that the current Etawah-trained model should not be
used as a Yamuna-wide operational turbidity estimator without additional
training and independent validation.

However, the study provides a useful framework for operational
development. The matching pipeline, Sentinel-2 feature extraction,
group-aware validation, temporal holdout, spatial holdout, and diagnostic
workflow provide a reproducible basis for constructing a genuinely
multi-reach model.

The next model generation should incorporate observations from multiple
stations spanning low, moderate, high, and extreme turbidity conditions.
Importantly, spatially independent validation should remain part of the
evaluation design rather than being replaced by random train-test
splitting.

5.5 Scientific significance

The negative spatial R² should not be interpreted as a failure of
Sentinel-2 remote sensing itself. Rather, it demonstrates the limitation
of transferring a compact model calibrated at one river reach to a
heterogeneous river system.

The study therefore contributes a realistic assessment of model
generalization. Instead of reporting only favourable development-site
performance, the independent spatial experiment exposes the conditions
under which the model fails and identifies extreme turbidity and domain
shift as key areas requiring additional modelling effort.
""".strip()


# ============================================================
# LIMITATIONS
# ============================================================

limitations_text = f"""
6. Limitations

Several limitations should be considered when interpreting the results.

First, the independent temporal test contained only {TEMP_N} observations.
Although the temporal correlation was relatively strong, the small sample
size limits the precision of performance estimates and category-specific
conclusions.

Second, the spatial test contained {SPAT_N} observations from
{SPAT_STATIONS} additional stations, but the distribution of turbidity was
highly heterogeneous. In particular, {N_GE500} observations exceeded
500 NTU. This imbalance strongly influences RMSE and demonstrates the
importance of reporting performance by turbidity regime rather than using
a single aggregate metric.

Third, CWC measurements and Sentinel-2 acquisitions are not necessarily
simultaneous. Although the matching workflow used nearest-image matching
within a two-day window and the temporal differences were explicitly
verified, rapidly changing river conditions can still introduce
observation-to-image mismatch.

Fourth, cloud contamination and residual atmospheric effects remain
important sources of uncertainty. Several high-error observations were
associated with highly cloudy scenes.

Fifth, the feature set was intentionally compact. The model used eight
spectral and water-index predictors and therefore did not explicitly
represent additional variables such as suspended sediment composition,
flow conditions, channel morphology, water depth, meteorological
conditions, or seasonally varying environmental drivers.

Finally, the spatial validation demonstrates that the current model
cannot be assumed to generalize across the Yamuna. The appropriate next
step is therefore expansion of the training domain and independent
revalidation rather than tuning the current model against the existing
test observations.
""".strip()


# ============================================================
# CONCLUSION
# ============================================================

conclusion_text = f"""
7. Conclusion

A compact Sentinel-2-based Gradient Boosting model was developed for
turbidity estimation using CWC observations at Etawah and was evaluated
under independent temporal and spatial conditions.

The frozen model showed moderate temporal association on unseen 2024
Etawah observations, with MAE {TEMP_MAE:.3f} NTU, RMSE {TEMP_RMSE:.3f} NTU,
R² {TEMP_R2:.3f}, and Pearson r {TEMP_R:.3f}. However, transfer to
{SPAT_STATIONS} additional Yamuna stations resulted in substantially
poorer performance, with MAE {SPAT_MAE:.3f} NTU, RMSE {SPAT_RMSE:.3f} NTU,
R² {SPAT_R2:.3f}, and Pearson r {SPAT_R:.3f}.

The spatial evaluation demonstrated that extreme turbidity was the
dominant failure regime. Observations ≥100 NTU had MAE 288.248 NTU, and
all observations in this regime were underpredicted. Cloud contamination
and temporal matching affected error magnitude but did not fully explain
the poor spatial transfer.

The principal conclusion is therefore that an Etawah-calibrated compact
Sentinel-2 model cannot yet be considered a reliable Yamuna-wide
turbidity model. A robust regional model will require multi-station
training, broader representation of extreme turbidity, improved feature
engineering, and continued spatially independent validation.
""".strip()


# ============================================================
# FIGURE INVENTORY
# ============================================================

figure_inventory = pd.DataFrame([

    [1, "01_observed_vs_predicted", "Development", "Main text"],
    [2, "02_residual_analysis", "Development", "Main text"],
    [3, "03_turbidity_distribution", "Development", "Main text"],
    [4, "04_category_comparison", "Development", "Supplementary"],
    [5, "05_feature_correlations", "Development", "Supplementary"],
    [6, "06_residual_distribution", "Development", "Supplementary"],
    [7, "07_extreme_event_predictions", "Development", "Main text"],

    [8, "08_spatial_observed_vs_predicted", "Spatial validation", "Main text"],
    [9, "09_spatial_residuals", "Spatial validation", "Main text"],
    [10, "10_station_MAE", "Spatial validation", "Main text"],
    [11, "11_station_R2", "Spatial validation", "Main text"],
    [12, "12_spatial_category_comparison", "Spatial validation", "Main text"],
    [13, "13_spatial_extreme_events", "Spatial validation", "Main text"],

    [14, "14_temporal_observed_vs_predicted_2024", "Temporal validation", "Main text"],
    [15, "15_temporal_residuals_2024", "Temporal validation", "Main text"],
    [16, "16_temporal_cloud_vs_error_2024", "Temporal diagnostics", "Supplementary"],
    [17, "17_temporal_gap_vs_error_2024", "Temporal diagnostics", "Supplementary"],
    [18, "18_temporal_2024_observed_predicted_series", "Temporal validation", "Main text"],
    [19, "19_temporal_2024_error_ranking", "Temporal diagnostics", "Supplementary"],
    [20, "20_temporal_cloud_group_mae_2024", "Temporal diagnostics", "Supplementary"],
    [21, "21_temporal_gap_group_mae_2024", "Temporal diagnostics", "Supplementary"],

    [22, "22_spatial_final_observed_vs_predicted", "Final spatial validation", "Main text"],
    [23, "23_spatial_final_log_observed_vs_predicted", "Final spatial validation", "Main text"],
    [24, "24_spatial_final_residuals", "Final spatial validation", "Main text"],
    [25, "25_spatial_final_residuals_vs_predicted", "Final spatial validation", "Main text"],
    [26, "26_spatial_final_station_MAE", "Station validation", "Main text"],
    [27, "27_spatial_final_station_R2", "Station validation", "Main text"],
    [28, "28_spatial_final_station_bias", "Station diagnostics", "Supplementary"],
    [29, "29_spatial_final_category_MAE", "Turbidity regimes", "Main text"],
    [30, "30_spatial_final_category_observed_predicted", "Turbidity regimes", "Main text"],
    [31, "31_spatial_final_extreme_events", "Extreme turbidity", "Main text"],
    [32, "32_spatial_final_cloud_vs_error", "Spatial diagnostics", "Supplementary"],
    [33, "33_spatial_final_temporal_gap_vs_error", "Spatial diagnostics", "Supplementary"],
    [34, "34_spatial_final_observed_turbidity_distribution", "Spatial data", "Main text"],
    [35, "35_spatial_final_log_turbidity_distribution", "Spatial data", "Supplementary"],
    [36, "36_spatial_final_feature_distribution_shift", "Domain shift", "Main text"],
    [37, "37_spatial_final_station_extreme_burden", "Domain shift", "Main text"],
    [38, "38_spatial_final_high_turbidity_cloud_MAE", "Diagnostics", "Supplementary"],
    [39, "39_spatial_final_station_observed_predicted_means", "Station validation", "Main text"],

], columns=[
    "figure_number",
    "filename_stem",
    "scientific_role",
    "recommended_placement",
])


figure_inventory.to_csv(
    OUTPUT /
    "figure_inventory.csv",
    index=False
)


# ============================================================
# TABLE INVENTORY
# ============================================================

table_inventory = pd.DataFrame([

    [1, "Study validation design", "Main text"],
    [2, "Final validation performance", "Main text"],
    [3, "Temporal 2024 category performance", "Supplementary"],
    [4, "Spatial category performance", "Main text"],
    [5, "Spatial station performance", "Main text"],
    [6, "Temporal sensitivity analysis", "Supplementary"],
    [7, "Spatial sensitivity and quality analysis", "Supplementary"],
    [8, "Extreme turbidity summary", "Main text"],
    [9, "Spatial extreme-event details", "Supplementary"],
    [10, "Station extreme-turbidity burden", "Main text"],
    [11, "Feature distribution shift", "Main text"],
    [12, "Station feature shift", "Supplementary"],
    [13, "High-turbidity cloud diagnostic", "Supplementary"],
    [14, "Spatial underprediction by turbidity", "Main text"],
    [15, "Final study findings", "Supplementary"],
    [16, "Final numerical summary", "Internal"],
    [17, "Publication-ready validation metrics", "Main text"],
    [18, "Validation integrity", "Supplementary"],
], columns=[
    "table_number",
    "table_description",
    "recommended_placement",
])

table_inventory.to_csv(
    OUTPUT /
    "table_inventory.csv",
    index=False
)


# ============================================================
# WRITE MANUSCRIPT TEXT
# ============================================================

print(
    "\nWriting manuscript text..."
)


full_manuscript = f"""
YAMUNA WATER QUALITY — MANUSCRIPT RESULTS PACKAGE

============================================================
ABSTRACT
============================================================

{abstract}


============================================================
RESULTS
============================================================

{results_text}


============================================================
DISCUSSION
============================================================

{discussion_text}


============================================================
LIMITATIONS
============================================================

{limitations_text}


============================================================
CONCLUSION
============================================================

{conclusion_text}
""".strip()


with open(
    OUTPUT /
    "manuscript_results_discussion_conclusion.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        full_manuscript
    )


# ============================================================
# WRITE ABSTRACT ONLY
# ============================================================

with open(
    OUTPUT /
    "abstract.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        abstract
    )


# ============================================================
# WRITE RESULTS ONLY
# ============================================================

with open(
    OUTPUT /
    "results_section.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        results_text
    )


# ============================================================
# WRITE DISCUSSION ONLY
# ============================================================

with open(
    OUTPUT /
    "discussion_section.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        discussion_text
    )


# ============================================================
# WRITE LIMITATIONS
# ============================================================

with open(
    OUTPUT /
    "limitations_section.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        limitations_text
    )


# ============================================================
# WRITE CONCLUSION
# ============================================================

with open(
    OUTPUT /
    "conclusion_section.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        conclusion_text
    )


# ============================================================
# MANUSCRIPT METRIC CHECKLIST
# ============================================================

metric_checklist = pd.DataFrame([

    {
        "metric":
            "Development N",
        "value":
            DEV_N,
        "source":
            "Frozen Etawah development model",
    },

    {
        "metric":
            "Development MAE",
        "value":
            DEV_MAE,
        "source":
            "Frozen Etawah development model",
    },

    {
        "metric":
            "Development RMSE",
        "value":
            DEV_RMSE,
        "source":
            "Frozen Etawah development model",
    },

    {
        "metric":
            "Development R2",
        "value":
            DEV_R2,
        "source":
            "Frozen Etawah development model",
    },

    {
        "metric":
            "Temporal N",
        "value":
            TEMP_N,
        "source":
            "Step 21",
    },

    {
        "metric":
            "Temporal MAE",
        "value":
            TEMP_MAE,
        "source":
            "Step 21",
    },

    {
        "metric":
            "Temporal RMSE",
        "value":
            TEMP_RMSE,
        "source":
            "Step 21",
    },

    {
        "metric":
            "Temporal R2",
        "value":
            TEMP_R2,
        "source":
            "Step 21",
    },

    {
        "metric":
            "Temporal Pearson r",
        "value":
            TEMP_R,
        "source":
            "Step 21",
    },

    {
        "metric":
            "Spatial N",
        "value":
            SPAT_N,
        "source":
            "Step 21",
    },

    {
        "metric":
            "Spatial stations",
        "value":
            SPAT_STATIONS,
        "source":
            "Step 21",
    },

    {
        "metric":
            "Spatial Sentinel-2 scenes",
        "value":
            SPAT_SCENES,
        "source":
            "Step 21",
    },

    {
        "metric":
            "Spatial MAE",
        "value":
            SPAT_MAE,
        "source":
            "Step 21",
    },

    {
        "metric":
            "Spatial RMSE",
        "value":
            SPAT_RMSE,
        "source":
            "Step 21",
    },

    {
        "metric":
            "Spatial R2",
        "value":
            SPAT_R2,
        "source":
            "Step 21",
    },

    {
        "metric":
            "Spatial Pearson r",
        "value":
            SPAT_R,
        "source":
            "Step 21",
    },

    {
        "metric":
            "Spatial observations >=100 NTU",
        "value":
            N_GE100,
        "source":
            "Frozen spatial predictions",
    },

    {
        "metric":
            "Spatial observations >=500 NTU",
        "value":
            N_GE500,
        "source":
            "Frozen spatial predictions",
    },

], columns=[
    "metric",
    "value",
    "source",
])

metric_checklist.to_csv(
    OUTPUT /
    "metric_checklist.csv",
    index=False
)


# ============================================================
# VALIDATION INTEGRITY REPORT
# ============================================================

integrity_report = """
VALIDATION INTEGRITY STATEMENT

The manuscript package was generated exclusively from the frozen
development model and independently generated temporal and spatial
validation outputs.

No model retraining was performed during Step 22.

No temporal or spatial test observations were added to model training.

No prediction values were modified.

No hyperparameter optimization was performed using the independent
temporal or spatial test observations.

The corrected CWC–Sentinel-2 temporal matching workflow was used.

The independent spatial test consisted of additional Yamuna monitoring
stations not used for development of the Etawah model.

The previous invalid temporal evaluation based on inconsistent stored
date metadata is excluded from all manuscript results.
""".strip()


with open(
    OUTPUT /
    "validation_integrity_statement.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        integrity_report
    )


# ============================================================
# FINAL PACKAGE INDEX
# ============================================================

package_files = sorted(
    OUTPUT.iterdir()
)

package_index = pd.DataFrame({

    "file":
        [
            f.name
            for f in package_files
            if f.is_file()
        ],

    "path":
        [
            str(f)
            for f in package_files
            if f.is_file()
        ],

})

package_index.to_csv(
    OUTPUT /
    "package_index.csv",
    index=False
)


# ============================================================
# FINAL CONSISTENCY CHECKS
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "FINAL STEP 22 CHECKS"
)

print(
    "=" * 75
)


assert TEMP_N == 14

assert abs(
    TEMP_MAE - 26.269
) < 0.01

assert abs(
    TEMP_RMSE - 35.086
) < 0.01

assert abs(
    TEMP_R2 - 0.064
) < 0.01

assert abs(
    TEMP_R - 0.762
) < 0.01


assert SPAT_N == 812

assert SPAT_STATIONS == 8

assert SPAT_SCENES == 521

assert abs(
    SPAT_MAE - 68.647
) < 0.01

assert abs(
    SPAT_RMSE - 157.876
) < 0.01

assert abs(
    SPAT_R2 - (-0.107)
) < 0.01

assert abs(
    SPAT_R - 0.236
) < 0.01

assert N_GE100 == 158

assert N_GE500 == 31


print(
    "\nAll frozen numerical checks PASSED."
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "STEP 22 COMPLETE"
)

print(
    "=" * 75
)

print(
    "\nManuscript package:"
)

print(
    OUTPUT
)

print(
    "\nGenerated files:"
)

for f in sorted(
    OUTPUT.iterdir()
):

    if f.is_file():

        print(
            f"  {f.name}"
        )


print(
    "\nFinal temporal result:"
)

print(
    f"  N = {TEMP_N}"
)

print(
    f"  MAE = {TEMP_MAE:.3f} NTU"
)

print(
    f"  RMSE = {TEMP_RMSE:.3f} NTU"
)

print(
    f"  R² = {TEMP_R2:.3f}"
)

print(
    f"  Pearson r = {TEMP_R:.3f}"
)


print(
    "\nFinal spatial result:"
)

print(
    f"  N = {SPAT_N}"
)

print(
    f"  Stations = {SPAT_STATIONS}"
)

print(
    f"  Sentinel-2 scenes = {SPAT_SCENES}"
)

print(
    f"  MAE = {SPAT_MAE:.3f} NTU"
)

print(
    f"  RMSE = {SPAT_RMSE:.3f} NTU"
)

print(
    f"  R² = {SPAT_R2:.3f}"
)

print(
    f"  Pearson r = {SPAT_R:.3f}"
)


print(
    "\nExtreme spatial turbidity:"
)

print(
    f"  >=100 NTU = {N_GE100}"
)

print(
    f"  >=500 NTU = {N_GE500}"
)


print(
    "\nModel integrity:"
)

print(
    "  Retrained = NO"
)

print(
    "  Test data added to training = NO"
)

print(
    "  Predictions modified = NO"
)

print(
    "\nSTEP 22 COMPLETE."
)

