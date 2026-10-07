# ============================================================
# STEP 18
# FINAL FROZEN MODEL — EXPANDED SPATIAL INDEPENDENT TEST
# ============================================================
#
# PURPOSE:
# Evaluate the frozen Etawah Compact-8 turbidity model on
# independent Yamuna stations.
#
# TRAINING:
#   Etawah only
#   199 labelled observations
#
# TEST:
#   8 unseen Yamuna stations
#   820 verified CWC-Sentinel matches
#   819 usable Compact-8 feature rows
#
# IMPORTANT:
#   - NO retraining
#   - NO hyperparameter tuning
#   - NO test-based scaling
#   - NO global B8 rescaling
#   - NO clipping of B8 > 1
#   - NO test observations added to training
#
# One observation is excluded because B2/B3 were completely
# unavailable in the Sentinel-2 image:
#
#   Vrindavan...
#   10-11-2023 11:33
#   20231110T053039_20231110T053039_T43RGL
#
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

ROOT = Path(r"D:\Etawah_Water_Quality")

TRAIN_FILE = (
    ROOT
    / "data"
    / "ml_ready_turbidity.csv"
)

MATCH_FILE = (
    ROOT
    / "Yamuna_Independent_Test"
    / "data"
    / "Yamuna_Expanded_Independent_Spatial_Match_2021_2024.csv"
)

FEATURE_FILE = (
    ROOT
    / "Yamuna_Independent_Test"
    / "data"
    / "Yamuna_Expanded_Compact8_Features_820.csv"
)

RESULTS = (
    ROOT
    / "Yamuna_Independent_Test"
    / "results"
)

RESULTS.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. FROZEN FEATURE SET
# ------------------------------------------------------------

FEATURES = [
    "MNDWI_median",
    "NDWI_median",
    "MNDWI_mean",
    "NDWI_mean",
    "water_fraction",
    "B11_mean",
    "B12_mean",
    "B8_mean",
]


# ------------------------------------------------------------
# 3. FROZEN MODEL
# ------------------------------------------------------------

MODEL = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=2,
    loss="huber",
    random_state=42,
)


# ------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------

def normalize_datetime(series):
    """
    Normalize CWC datetime strings consistently.
    """
    return (
        pd.to_datetime(
            series,
            dayfirst=True,
            errors="coerce",
        )
        .dt.floor("s")
    )


def normalize_string(series):
    return (
        series
        .astype(str)
        .str.strip()
    )


def safe_pearson(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan, np.nan

    if (
        np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):
        return np.nan, np.nan

    r, p = pearsonr(
        y_true,
        y_pred,
    )

    return float(r), float(p)


def metrics_dict(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residual = y_true - y_pred

    r, p = safe_pearson(
        y_true,
        y_pred,
    )

    return {
        "n": len(y_true),
        "MAE_NTU": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "RMSE_NTU": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),
        "R2": r2_score(
            y_true,
            y_pred,
        ),
        "bias_observed_minus_predicted_NTU":
            np.mean(residual),
        "Pearson_r": r,
        "Pearson_p": p,
        "observed_mean_NTU":
            np.mean(y_true),
        "predicted_mean_NTU":
            np.mean(y_pred),
        "observed_median_NTU":
            np.median(y_true),
        "predicted_median_NTU":
            np.median(y_pred),
        "underpredicted_count":
            int(
                np.sum(
                    y_pred < y_true
                )
            ),
        "underpredicted_percent":
            100
            * np.mean(
                y_pred < y_true
            ),
    }


# ------------------------------------------------------------
# 5. LOAD TRAINING DATA
# ------------------------------------------------------------

print("=" * 70)
print("LOADING FROZEN ETawah TRAINING DATA")
print("=" * 70)

train = pd.read_csv(
    TRAIN_FILE
)

print(
    "Training rows:",
    len(train)
)

missing_training_features = [
    c for c in FEATURES
    if c not in train.columns
]

if missing_training_features:
    raise ValueError(
        "Missing training features: "
        + str(missing_training_features)
    )


# ------------------------------------------------------------
# 6. LOAD VERIFIED MATCHING TABLE
# ------------------------------------------------------------

print()
print("=" * 70)
print("LOADING VERIFIED SPATIAL MATCHES")
print("=" * 70)

matches = pd.read_csv(
    MATCH_FILE
)

print(
    "Verified matching rows:",
    len(matches)
)

print(
    "Stations:",
    matches["Station"]
    .nunique()
)


# ------------------------------------------------------------
# 7. LOAD COMPACT-8 FEATURES
# ------------------------------------------------------------

print()
print("=" * 70)
print("LOADING COMPACT-8 FEATURES")
print("=" * 70)

features = pd.read_csv(
    FEATURE_FILE
)

print(
    "Feature rows:",
    len(features)
)


missing_features = [
    c for c in FEATURES
    if c not in features.columns
]

if missing_features:
    raise ValueError(
        "Missing Compact-8 columns: "
        + str(missing_features)
    )


# ------------------------------------------------------------
# 8. NORMALIZE IDENTIFIERS
# ------------------------------------------------------------

matches["_station_key"] = normalize_string(
    matches["Station"]
).str.upper()

features["_station_key"] = normalize_string(
    features["Station"]
).str.upper()


matches["_datetime_key"] = normalize_datetime(
    matches["Data Acquisition Time"]
)

features["_datetime_key"] = normalize_datetime(
    features["Data Acquisition Time"]
)


matches["_sentinel_key"] = normalize_string(
    matches["sentinel2_id"]
)

features["_sentinel_key"] = normalize_string(
    features["sentinel2_id"]
)


# ------------------------------------------------------------
# 9. VALIDATE DATETIME PARSING
# ------------------------------------------------------------

if matches["_datetime_key"].isna().any():
    raise ValueError(
        "Some matching-table CWC datetimes could not be parsed."
    )

if features["_datetime_key"].isna().any():
    raise ValueError(
        "Some feature-table CWC datetimes could not be parsed."
    )


# ------------------------------------------------------------
# 10. CREATE ONE-TO-ONE MATCH KEYS
# ------------------------------------------------------------

matches["_match_key"] = (
    matches["_station_key"]
    + "|"
    + matches["_datetime_key"]
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    + "|"
    + matches["_sentinel_key"]
)

features["_match_key"] = (
    features["_station_key"]
    + "|"
    + features["_datetime_key"]
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    + "|"
    + features["_sentinel_key"]
)


# ------------------------------------------------------------
# 11. KEY VALIDATION
# ------------------------------------------------------------

print()
print("=" * 70)
print("KEY VALIDATION")
print("=" * 70)

matching_duplicates = (
    matches["_match_key"]
    .duplicated()
    .sum()
)

feature_duplicates = (
    features["_match_key"]
    .duplicated()
    .sum()
)

print(
    "Matching duplicate keys:",
    matching_duplicates
)

print(
    "Feature duplicate keys:",
    feature_duplicates
)

if matching_duplicates:
    raise ValueError(
        "Duplicate matching keys detected."
    )

if feature_duplicates:
    raise ValueError(
        "Duplicate feature keys detected."
    )


missing_from_features = (
    set(matches["_match_key"])
    - set(features["_match_key"])
)

extra_features = (
    set(features["_match_key"])
    - set(matches["_match_key"])
)

print(
    "Matching keys missing from feature table:",
    len(missing_from_features)
)

print(
    "Feature keys not present in matching table:",
    len(extra_features)
)

if missing_from_features:
    print(
        "\nFirst missing keys:"
    )

    for key in list(
        missing_from_features
    )[:10]:
        print(key)

    raise ValueError(
        "Verified matching rows are missing "
        "from the feature table."
    )


# ------------------------------------------------------------
# 12. ONE-TO-ONE JOIN
# ------------------------------------------------------------

test = matches.merge(
    features[
        [
            "_match_key"
        ]
        + FEATURES
        + [
            "valid_B8_pixels",
            "feature_status",
        ]
    ],
    on="_match_key",
    how="left",
    suffixes=(
        "",
        "_feature",
    ),
    validate="one_to_one",
)

print()
print(
    "Joined rows:",
    len(test)
)


# ------------------------------------------------------------
# 13. IDENTIFY INCOMPLETE FEATURE ROWS
# ------------------------------------------------------------

test["missing_feature_count"] = (
    test[FEATURES]
    .isna()
    .sum(axis=1)
)

invalid_feature_mask = (
    test["missing_feature_count"] > 0
)

invalid_feature_rows = test[
    invalid_feature_mask
].copy()

valid_test = test[
    ~invalid_feature_mask
].copy()


# ------------------------------------------------------------
# 14. REPORT FEATURE EXCLUSIONS
# ------------------------------------------------------------

print()
print("=" * 70)
print("FEATURE QC")
print("=" * 70)

print(
    "Total verified spatial matches:",
    len(test)
)

print(
    "Rows with complete Compact-8:",
    len(valid_test)
)

print(
    "Rows excluded due to missing Compact-8:",
    len(invalid_feature_rows)
)


if len(invalid_feature_rows) > 0:

    print()
    print(
        "Excluded observations:"
    )

    display_cols = [
        "Station",
        "Data Acquisition Time",
        "Turbidity (NTU)",
        "sentinel2_id",
        "cloud_percentage",
        "days_difference",
        "feature_status",
    ]

    print(
        invalid_feature_rows[
            display_cols
        ].to_string(
            index=False
        )
    )

    invalid_feature_rows[
        display_cols
    ].to_csv(
        RESULTS
        / "spatial_excluded_invalid_features.csv",
        index=False,
    )


# ------------------------------------------------------------
# 15. ASSERT EXPECTED QC RESULT
# ------------------------------------------------------------

if len(test) != 820:
    raise ValueError(
        f"Expected 820 verified matches, got {len(test)}"
    )

if len(valid_test) != 819:
    raise ValueError(
        "Expected exactly 819 usable spatial "
        f"observations, got {len(valid_test)}"
    )


# ------------------------------------------------------------
# 16. TRAIN FROZEN MODEL
# ------------------------------------------------------------

print()
print("=" * 70)
print("FITTING FROZEN ETawah MODEL")
print("=" * 70)

train_model = train.dropna(
    subset=FEATURES + [
        "turbidity_NTU"
    ]
).copy()

X_train = train_model[
    FEATURES
]

y_train = train_model[
    "turbidity_NTU"
]

print(
    "Training rows used:",
    len(train_model)
)

if len(train_model) != 199:
    raise ValueError(
        "Frozen training dataset should contain "
        "exactly 199 rows."
    )


MODEL.fit(
    X_train,
    y_train,
)

print(
    "Model:",
    MODEL
)


# ------------------------------------------------------------
# 17. IMPORTANT: NO TEST RESCALING
# ------------------------------------------------------------

print()
print("=" * 70)
print("FEATURE SCALE POLICY")
print("=" * 70)

print(
    "No automatic test scaling applied."
)

print(
    "B8 > 1 observations retained:",
    int(
        (
            valid_test["B8_mean"]
            > 1
        ).sum()
    )
)

print(
    "B11 > 1 observations:",
    int(
        (
            valid_test["B11_mean"]
            > 1
        ).sum()
    )
)

print(
    "B12 > 1 observations:",
    int(
        (
            valid_test["B12_mean"]
            > 1
        ).sum()
    )
)


# ------------------------------------------------------------
# 18. PREDICT
# ------------------------------------------------------------

X_test = valid_test[
    FEATURES
]

y_test = pd.to_numeric(
    valid_test[
        "Turbidity (NTU)"
    ],
    errors="coerce",
)

labelled_mask = (
    y_test.notna()
)

labelled_test = valid_test[
    labelled_mask
].copy()

y_test = y_test[
    labelled_mask
].to_numpy()

X_test = labelled_test[
    FEATURES
]

print()
print(
    "Usable feature rows:",
    len(valid_test)
)

print(
    "Usable labelled test rows:",
    len(labelled_test)
)


predictions = MODEL.predict(
    X_test
)

labelled_test[
    "predicted_turbidity_NTU"
] = predictions

labelled_test[
    "residual_observed_minus_predicted_NTU"
] = (
    labelled_test[
        "Turbidity (NTU)"
    ]
    - labelled_test[
        "predicted_turbidity_NTU"
    ]
)

labelled_test[
    "absolute_error_NTU"
] = (
    labelled_test[
        "residual_observed_minus_predicted_NTU"
    ]
    .abs()
)


# ------------------------------------------------------------
# 19. OVERALL METRICS
# ------------------------------------------------------------

overall = metrics_dict(
    labelled_test[
        "Turbidity (NTU)"
    ].to_numpy(),
    labelled_test[
        "predicted_turbidity_NTU"
    ].to_numpy(),
)

overall_df = pd.DataFrame(
    [overall]
)

overall_df.insert(
    0,
    "test_type",
    "Expanded independent spatial test",
)

overall_df.insert(
    1,
    "training_rows",
    len(train_model),
)

overall_df.insert(
    2,
    "verified_matches",
    len(test),
)

overall_df.insert(
    3,
    "usable_feature_rows",
    len(valid_test),
)

overall_df.insert(
    4,
    "usable_labelled_rows",
    len(labelled_test),
)

overall_df.insert(
    5,
    "excluded_feature_rows",
    len(invalid_feature_rows),
)


# ------------------------------------------------------------
# 20. PRINT OVERALL RESULTS
# ------------------------------------------------------------

print()
print("=" * 70)
print("FINAL SPATIAL INDEPENDENT TEST RESULTS")
print("=" * 70)

print(
    f"N = {overall['n']}"
)

print(
    f"MAE = {overall['MAE_NTU']:.3f} NTU"
)

print(
    f"RMSE = {overall['RMSE_NTU']:.3f} NTU"
)

print(
    f"R² = {overall['R2']:.3f}"
)

print(
    "Bias observed - predicted = "
    f"{overall['bias_observed_minus_predicted_NTU']:.3f} NTU"
)

print(
    f"Pearson r = {overall['Pearson_r']:.3f}"
)

print(
    f"Pearson p = {overall['Pearson_p']:.6g}"
)

print(
    f"Observed mean = "
    f"{overall['observed_mean_NTU']:.3f}"
)

print(
    f"Predicted mean = "
    f"{overall['predicted_mean_NTU']:.3f}"
)


# ------------------------------------------------------------
# 21. STATION PERFORMANCE
# ------------------------------------------------------------

station_rows = []

for station, group in (
    labelled_test
    .groupby(
        "Station",
        sort=True
    )
):

    m = metrics_dict(
        group[
            "Turbidity (NTU)"
        ].to_numpy(),
        group[
            "predicted_turbidity_NTU"
        ].to_numpy(),
    )

    m["Station"] = station

    station_rows.append(
        m
    )


station_performance = (
    pd.DataFrame(
        station_rows
    )
    [
        [
            "Station",
            "n",
            "MAE_NTU",
            "RMSE_NTU",
            "R2",
            "bias_observed_minus_predicted_NTU",
            "Pearson_r",
            "Pearson_p",
            "observed_mean_NTU",
            "predicted_mean_NTU",
            "underpredicted_percent",
        ]
    ]
    .sort_values(
        "Station"
    )
)


# ------------------------------------------------------------
# 22. TURBIDITY CATEGORIES
# ------------------------------------------------------------

def turbidity_category(value):

    if value < 10:
        return "Low"

    if value < 20:
        return "Moderate"

    if value < 50:
        return "Elevated"

    if value < 100:
        return "High"

    return "Very high"


labelled_test[
    "turbidity_category"
] = labelled_test[
    "Turbidity (NTU)"
].apply(
    turbidity_category
)


category_order = [
    "Low",
    "Moderate",
    "Elevated",
    "High",
    "Very high",
]


category_rows = []

for category in category_order:

    group = labelled_test[
        labelled_test[
            "turbidity_category"
        ]
        == category
    ]

    if len(group) == 0:
        continue

    m = metrics_dict(
        group[
            "Turbidity (NTU)"
        ].to_numpy(),
        group[
            "predicted_turbidity_NTU"
        ].to_numpy(),
    )

    m["turbidity_category"] = category

    category_rows.append(
        m
    )


category_performance = pd.DataFrame(
    category_rows
)


# ------------------------------------------------------------
# 23. EXTREME EVENTS
# ------------------------------------------------------------

extreme_events = (
    labelled_test[
        labelled_test[
            "Turbidity (NTU)"
        ] >= 100
    ]
    .copy()
    .sort_values(
        "Turbidity (NTU)",
        ascending=False,
    )
)


# ------------------------------------------------------------
# 24. CLOUD PERFORMANCE
# ------------------------------------------------------------

cloud_rows = []

for threshold in [
    20,
    40,
    60,
]:

    subset = labelled_test[
        pd.to_numeric(
            labelled_test[
                "cloud_percentage"
            ],
            errors="coerce",
        )
        <= threshold
    ]

    if len(subset) == 0:
        continue

    m = metrics_dict(
        subset[
            "Turbidity (NTU)"
        ].to_numpy(),
        subset[
            "predicted_turbidity_NTU"
        ].to_numpy(),
    )

    m[
        "cloud_threshold_percent"
    ] = threshold

    cloud_rows.append(
        m
    )


cloud_performance = pd.DataFrame(
    cloud_rows
)


# ------------------------------------------------------------
# 25. TEMPORAL-GAP PERFORMANCE
# ------------------------------------------------------------

gap_rows = []

for threshold in [
    0.5,
    1.0,
    1.5,
    2.0,
]:

    subset = labelled_test[
        pd.to_numeric(
            labelled_test[
                "days_difference"
            ],
            errors="coerce",
        )
        <= threshold
    ]

    if len(subset) == 0:
        continue

    m = metrics_dict(
        subset[
            "Turbidity (NTU)"
        ].to_numpy(),
        subset[
            "predicted_turbidity_NTU"
        ].to_numpy(),
    )

    m[
        "days_difference_threshold"
    ] = threshold

    gap_rows.append(
        m
    )


gap_performance = pd.DataFrame(
    gap_rows
)


# ------------------------------------------------------------
# 26. ERROR RANKING
# ------------------------------------------------------------

error_ranking = (
    labelled_test[
        [
            "Station",
            "Data Acquisition Time",
            "Turbidity (NTU)",
            "predicted_turbidity_NTU",
            "residual_observed_minus_predicted_NTU",
            "absolute_error_NTU",
            "cloud_percentage",
            "days_difference",
            "sentinel2_id",
        ]
    ]
    .sort_values(
        "absolute_error_NTU",
        ascending=False,
    )
)


# ------------------------------------------------------------
# 27. SAVE PREDICTIONS
# ------------------------------------------------------------

prediction_columns = [
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
    "sentinel2_date",
    "valid_B8_pixels",
    "water_fraction",
] + FEATURES


labelled_test[
    prediction_columns
].to_csv(
    RESULTS
    / "expanded_spatial_frozen_model_predictions.csv",
    index=False,
)


# ------------------------------------------------------------
# 28. SAVE ALL OUTPUTS
# ------------------------------------------------------------

overall_df.to_csv(
    RESULTS
    / "expanded_spatial_frozen_model_metrics.csv",
    index=False,
)

station_performance.to_csv(
    RESULTS
    / "expanded_spatial_station_performance.csv",
    index=False,
)

category_performance.to_csv(
    RESULTS
    / "expanded_spatial_category_performance.csv",
    index=False,
)

extreme_events.to_csv(
    RESULTS
    / "expanded_spatial_extreme_events.csv",
    index=False,
)

cloud_performance.to_csv(
    RESULTS
    / "expanded_spatial_cloud_performance.csv",
    index=False,
)

gap_performance.to_csv(
    RESULTS
    / "expanded_spatial_timegap_performance.csv",
    index=False,
)

error_ranking.to_csv(
    RESULTS
    / "expanded_spatial_error_ranking.csv",
    index=False,
)


# ------------------------------------------------------------
# 29. PRINT STATION RESULTS
# ------------------------------------------------------------

print()
print("=" * 70)
print("STATION PERFORMANCE")
print("=" * 70)

print(
    station_performance.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# ------------------------------------------------------------
# 30. PRINT CATEGORY RESULTS
# ------------------------------------------------------------

print()
print("=" * 70)
print("CATEGORY PERFORMANCE")
print("=" * 70)

print(
    category_performance.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# ------------------------------------------------------------
# 31. PRINT EXTREME EVENTS
# ------------------------------------------------------------

print()
print("=" * 70)
print("EXTREME EVENTS >= 100 NTU")
print("=" * 70)

if len(extreme_events) > 0:

    print(
        extreme_events[
            [
                "Station",
                "Data Acquisition Time",
                "Turbidity (NTU)",
                "predicted_turbidity_NTU",
                "absolute_error_NTU",
                "cloud_percentage",
                "sentinel2_id",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

else:

    print(
        "No observations >= 100 NTU."
    )


# ------------------------------------------------------------
# 32. PRINT CLOUD PERFORMANCE
# ------------------------------------------------------------

print()
print("=" * 70)
print("CLOUD SENSITIVITY")
print("=" * 70)

if len(cloud_performance) > 0:

    print(
        cloud_performance.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


# ------------------------------------------------------------
# 33. PRINT TIME-GAP PERFORMANCE
# ------------------------------------------------------------

print()
print("=" * 70)
print("TEMPORAL-GAP SENSITIVITY")
print("=" * 70)

if len(gap_performance) > 0:

    print(
        gap_performance.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


# ------------------------------------------------------------
# 34. PRINT TOP ERRORS
# ------------------------------------------------------------

print()
print("=" * 70)
print("TOP 20 ABSOLUTE ERRORS")
print("=" * 70)

print(
    error_ranking.head(20).to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# ------------------------------------------------------------
# 35. FINAL QC
# ------------------------------------------------------------

print()
print("=" * 70)
print("FINAL QC")
print("=" * 70)

print(
    "Verified matches:",
    len(test)
)

print(
    "Complete Compact-8 rows:",
    len(valid_test)
)

print(
    "Excluded incomplete rows:",
    len(invalid_feature_rows)
)

print(
    "Labelled predictions:",
    len(labelled_test)
)

print(
    "B8 > 1 retained:",
    int(
        (
            labelled_test[
                "B8_mean"
            ] > 1
        ).sum()
    )
)

print(
    "Global scale correction:",
    "NO"
)

print(
    "Model retraining on test data:",
    "NO"
)

print(
    "Test observations added to training:",
    "NO"
)

print()
print(
    "Step 18 COMPLETE."
)

print(
    "Results written to:"
)

print(
    RESULTS
)