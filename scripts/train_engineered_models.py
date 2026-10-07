from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

INPUT_FILE = DATA_DIR / "ml_ready_turbidity.csv"
OUTPUT_FILE = DATA_DIR / "engineered_model_cv_results.csv"

TARGET = "turbidity_NTU"
GROUP = "sentinel2_id"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ETAWAH ENGINEERED FEATURE MODEL TEST")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df.dropna(
    subset=[TARGET, GROUP]
).copy()

print(f"Rows: {len(df)}")
print(f"Unique Sentinel groups: {df[GROUP].nunique()}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Existing patch statistics
BASE_FEATURES = [
    "B2_mean",
    "B3_mean",
    "B4_mean",
    "B8_mean",
    "B11_mean",
    "B12_mean",

    "B2_median",
    "B3_median",
    "B4_median",
    "B8_median",
    "B11_median",
    "B12_median",

    "B2_std",
    "B3_std",
    "B4_std",
    "B8_std",
    "B11_std",
    "B12_std",

    "NDWI_mean",
    "NDWI_median",
    "NDWI_std",

    "MNDWI_mean",
    "MNDWI_median",
    "MNDWI_std",

    "water_fraction",
]


# Check required columns
missing = [
    c for c in BASE_FEATURES
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing feature columns: {missing}"
    )


# ------------------------------------------------------------
# Spectral ratios and differences
# ------------------------------------------------------------

EPS = 1e-6

df["B2_B3_ratio"] = (
    df["B2_mean"] /
    (df["B3_mean"] + EPS)
)

df["B4_B3_ratio"] = (
    df["B4_mean"] /
    (df["B3_mean"] + EPS)
)

df["B8_B4_ratio"] = (
    df["B8_mean"] /
    (df["B4_mean"] + EPS)
)

df["B11_B8_ratio"] = (
    df["B11_mean"] /
    (df["B8_mean"] + EPS)
)

df["B12_B8_ratio"] = (
    df["B12_mean"] /
    (df["B8_mean"] + EPS)
)

df["B11_B12_ratio"] = (
    df["B11_mean"] /
    (df["B12_mean"] + EPS)
)

df["B8_minus_B4"] = (
    df["B8_mean"] -
    df["B4_mean"]
)

df["B11_minus_B8"] = (
    df["B11_mean"] -
    df["B8_mean"]
)

df["B12_minus_B8"] = (
    df["B12_mean"] -
    df["B8_mean"]
)


RATIO_FEATURES = [
    "B2_B3_ratio",
    "B4_B3_ratio",
    "B8_B4_ratio",
    "B11_B8_ratio",
    "B12_B8_ratio",
    "B11_B12_ratio",
    "B8_minus_B4",
    "B11_minus_B8",
    "B12_minus_B8",
]


# ============================================================
# TWO FEATURE SETS
# ============================================================

# Original baseline
BASELINE_FEATURES = [
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


# Engineered feature set
ENGINEERED_FEATURES = (
    BASE_FEATURES +
    RATIO_FEATURES
)


# ============================================================
# CLEAN DATA
# ============================================================

all_features = list(
    dict.fromkeys(
        BASELINE_FEATURES +
        ENGINEERED_FEATURES
    )
)

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=all_features
).copy()

print(
    f"Rows after feature cleaning: {len(df)}"
)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {

    "Ridge": Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                Ridge(alpha=1.0)
            ),
        ]
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        loss="huber",
        random_state=42,
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=500,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    ),
}


# ============================================================
# GROUPED CV
# ============================================================

X_groups = df[GROUP].astype(str).to_numpy()
y = df[TARGET].astype(float)

gkf = GroupKFold(
    n_splits=5
)

results = []


for feature_set_name, feature_list in [
    ("Baseline", BASELINE_FEATURES),
    ("Engineered", ENGINEERED_FEATURES),
]:

    print()
    print("=" * 70)
    print(
        f"{feature_set_name.upper()} FEATURES"
    )
    print("=" * 70)

    X = df[feature_list].astype(float)

    print(
        f"Number of features: {X.shape[1]}"
    )

    for model_name, model in models.items():

        fold_mae = []
        fold_rmse = []
        fold_r2 = []

        print()
        print(
            f"--- {model_name} ---"
        )

        for fold, (train_idx, test_idx) in enumerate(
            gkf.split(
                X,
                y,
                groups=X_groups
            ),
            start=1
        ):

            train_groups = set(
                X_groups[train_idx]
            )

            test_groups = set(
                X_groups[test_idx]
            )

            leakage = (
                train_groups &
                test_groups
            )

            if leakage:
                raise RuntimeError(
                    f"Sentinel leakage in fold {fold}"
                )

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model.fit(
                X_train,
                y_train
            )

            prediction = model.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                prediction
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    prediction
                )
            )

            r2 = r2_score(
                y_test,
                prediction
            )

            fold_mae.append(mae)
            fold_rmse.append(rmse)
            fold_r2.append(r2)

            print(
                f"Fold {fold}: "
                f"MAE={mae:.3f} | "
                f"RMSE={rmse:.3f} | "
                f"R²={r2:.3f}"
            )

        results.append(
            {
                "feature_set": feature_set_name,
                "model": model_name,
                "MAE_mean": np.mean(fold_mae),
                "MAE_std": np.std(
                    fold_mae,
                    ddof=1
                ),
                "RMSE_mean": np.mean(
                    fold_rmse
                ),
                "RMSE_std": np.std(
                    fold_rmse,
                    ddof=1
                ),
                "R2_mean": np.mean(
                    fold_r2
                ),
                "R2_std": np.std(
                    fold_r2,
                    ddof=1
                ),
            }
        )

        print(
            f"Mean: "
            f"MAE={np.mean(fold_mae):.3f} ± "
            f"{np.std(fold_mae, ddof=1):.3f} | "
            f"RMSE={np.mean(fold_rmse):.3f} ± "
            f"{np.std(fold_rmse, ddof=1):.3f} | "
            f"R²={np.mean(fold_r2):.3f} ± "
            f"{np.std(fold_r2, ddof=1):.3f}"
        )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    [
        "RMSE_mean",
        "MAE_mean"
    ]
)

print()
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# FEATURE IMPROVEMENT
# ============================================================

print()
print("=" * 70)
print("ENGINEERED FEATURE IMPROVEMENT")
print("=" * 70)

for model_name in models.keys():

    base = results_df[
        (
            results_df["model"]
            == model_name
        )
        &
        (
            results_df["feature_set"]
            == "Baseline"
        )
    ]

    engineered = results_df[
        (
            results_df["model"]
            == model_name
        )
        &
        (
            results_df["feature_set"]
            == "Engineered"
        )
    ]

    if len(base) == 1 and len(engineered) == 1:

        base_rmse = base.iloc[0][
            "RMSE_mean"
        ]

        eng_rmse = engineered.iloc[0][
            "RMSE_mean"
        ]

        base_mae = base.iloc[0][
            "MAE_mean"
        ]

        eng_mae = engineered.iloc[0][
            "MAE_mean"
        ]

        print()
        print(model_name)

        print(
            f"RMSE: "
            f"{base_rmse:.3f} -> "
            f"{eng_rmse:.3f}"
        )

        print(
            f"MAE:  "
            f"{base_mae:.3f} -> "
            f"{eng_mae:.3f}"
        )

        print(
            f"RMSE change: "
            f"{eng_rmse - base_rmse:+.3f}"
        )

        print(
            f"MAE change: "
            f"{eng_mae - base_mae:+.3f}"
        )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_FILE)

print()
print("DONE.")