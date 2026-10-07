
# Sentinel-2-Based Turbidity Estimation in the Yamuna River

## Independent Temporal and Spatial Validation of an Etawah-Calibrated Model

This repository contains the Google Earth Engine (GEE) scripts, Python analysis scripts, validation workflows, diagnostic analyses, and publication figures developed for a study of Sentinel-2-based turbidity estimation in the Yamuna River.

The study develops a turbidity model using Central Water Commission (CWC) observations from **Etawah** and Sentinel-2 Surface Reflectance imagery, and then evaluates the frozen model using two independent validation strategies:

1. **Independent temporal validation at Etawah**
2. **Independent spatial validation across additional Yamuna monitoring stations**

The primary objective is to evaluate how well a model calibrated at one river location and time period transfers to later observations and to geographically distinct locations.

---

## Study Title

**Sentinel-2-Based Turbidity Estimation in the Yamuna River: Independent Temporal and Spatial Validation of an Etawah-Calibrated Model**

---

## Overview

River turbidity is an important indicator of water quality and suspended particulate conditions. Satellite remote sensing provides the possibility of estimating turbidity over larger spatial and temporal scales than conventional field sampling alone.

This study combines:

- Central Water Commission (CWC) surface-water-quality observations
- Sentinel-2 Surface Reflectance imagery
- Water indices including NDWI and MNDWI
- Local spectral reflectance features
- Gradient Boosting regression
- Grouped cross-validation
- Independent temporal validation
- Independent spatial validation

A model was first developed using observations from **Etawah**, on the Yamuna River. The model was then frozen and evaluated without retraining or tuning on the independent validation datasets.

This design is intended to provide a more realistic assessment of model transferability than random train-test splitting alone.

---

# 1. Data Sources

## 1.1 Central Water Commission

The study uses surface-water-quality observations from the **Central Water Commission (CWC)**.

The CWC observations contain station information, acquisition date/time, geographic coordinates, and measured turbidity.

The primary CWC asset used in Google Earth Engine is:

```text
projects/sentinel2-research-508107/assets/swq_physical_parameter_manual_cwc_up_2021_2025
````

Relevant fields include:

```text
Local River
Station
Data Acquisition Time
Latitude
Longitude
Turbidity (NTU)
```

The Yamuna observations were selected using:

```text
Local River == "Yamuna"
```

The development model was calibrated using observations from:

```text
Station == "ETAWAH"
```

### Data access note

The original CWC observations are subject to applicable data-access and redistribution conditions. Raw CWC data are therefore not redistributed through this repository unless redistribution is permitted.

---

## 1.2 Sentinel-2

Sentinel-2 Surface Reflectance Harmonized imagery was obtained from the Google Earth Engine collection:

```text
COPERNICUS/S2_SR_HARMONIZED
```

Reference:

Drusch, M., Del Bello, U., Carlier, S., Colin, O., Fernandez, V., Gascon, F., Hoersch, B., Isola, C., Laberinti, P., Martimort, P., Meygret, A., Spoto, F., Sy, O., Marchese, F., & Bargellini, P. (2012). Sentinel-2: ESA's optical high-resolution mission for GMES operational services. *Remote Sensing of Environment, 120*, 25–36.

---

# 2. Study Design

The study follows a frozen-model validation framework.

```text
CWC observations
       │
       ▼
Etawah observations
       │
       ▼
Sentinel-2 temporal matching
       │
       ▼
Compact-8 feature extraction
       │
       ▼
Model development
       │
       ▼
Frozen Etawah model
       │
       ├──────────────► Independent temporal validation
       │                at Etawah
       │
       └──────────────► Independent spatial validation
                        across additional Yamuna stations
```

The validation datasets were not used for model tuning.

---

# 3. Model Development

The development dataset contains:

* **199 labelled observations**
* **99 Sentinel-2 groups**

The model uses Sentinel-2 spectral and water-index predictors.

## Final predictor set

The frozen model uses eight predictors:

```text
MNDWI_median
NDWI_median
MNDWI_mean
NDWI_mean
water_fraction
B11_mean
B12_mean
B8_mean
```

NDWI follows the formulation introduced by McFeeters (1996).

MNDWI follows the modification proposed by Xu (2006).

References:

McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. *International Journal of Remote Sensing, 17*(7), 1425–1432.

Xu, H. (2006). Modification of normalized difference water index (NDWI) to enhance open water features in remotely sensed imagery. *International Journal of Remote Sensing, 27*(14), 3025–3033.

---

# 4. Satellite Feature Extraction

Features were extracted from Sentinel-2 imagery around the CWC monitoring location.

The final feature extraction workflow uses a **256-m image patch** centered on the CWC station coordinates.

The extracted predictors include:

* Median MNDWI
* Median NDWI
* Mean MNDWI
* Mean NDWI
* Water fraction
* Mean B8 reflectance
* Mean B11 reflectance
* Mean B12 reflectance

Reflectance bands were scaled appropriately from the Sentinel-2 surface reflectance product.

A water mask based on MNDWI was used for water-related feature extraction.

---

# 5. Sentinel-2 Matching

CWC observations were matched with Sentinel-2 acquisitions using the actual acquisition timestamps.

The matching procedure:

1. Parse the CWC observation date and time.
2. Search Sentinel-2 imagery within ±2 days.
3. Calculate the absolute temporal difference between the CWC observation and Sentinel-2 acquisition.
4. Identify candidate Sentinel-2 scenes.
5. Select the nearest usable scene with valid pixels for feature extraction.

The final matching procedure prioritizes the nearest image that contains usable B8 pixels rather than simply selecting the nearest image regardless of data validity.

---

# 6. Machine-Learning Model

The final estimator is a Gradient Boosting regression model:

```python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=2,
    loss="huber",
    random_state=42
)
```

The Gradient Boosting approach follows the framework described by:

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232.

---

# 7. Development Validation

Model development used **5-fold GroupKFold cross-validation**.

The grouping variable was:

```text
sentinel2_id
```

This prevents observations associated with the same Sentinel-2 acquisition from being distributed across both training and validation folds.

Development performance:

| Metric       |      Value |
| ------------ | ---------: |
| Observations |        199 |
| MAE          | 22.954 NTU |
| RMSE         | 41.599 NTU |
| R²           |      0.236 |

These values describe the model-development stage and should not be interpreted as independent generalization performance.

---

# 8. Independent Temporal Validation

The independent temporal test evaluates later Etawah observations using a model developed from earlier observations.

Training period:

```text
2021–2023
```

Independent test period:

```text
2024
```

The model was not retrained or tuned using the 2024 observations.

## Temporal validation results

| Metric                      |        2024 |
| --------------------------- | ----------: |
| N                           |          14 |
| MAE                         |  26.269 NTU |
| RMSE                        |  35.086 NTU |
| R²                          |       0.064 |
| Bias (Observed − Predicted) | -19.140 NTU |
| Pearson r                   |       0.762 |

Observed mean turbidity:

```text
33.543 NTU
```

Predicted mean turbidity:

```text
52.683 NTU
```

The temporal validation indicates that the frozen Etawah model retains some correspondence with later observations but does not reproduce the 2024 turbidity distribution accurately.

---

# 9. Independent Spatial Validation

The spatial validation evaluates the frozen Etawah model at additional Yamuna monitoring stations.

The expanded spatial validation includes:

```text
Baghpat
Mawi
Kalanaur
Mathura (Gokul Barrage)
Vrindavan-Yamuna ExpresswayLink Road Bridge U/S of Mathura
Yamuna Highway Road Bridge
AGRA(POIYAGHAT)
Kailas Mandir, Agra
```

After quality control and removal of one observation with unusable satellite features:

```text
812 labelled observations
```

were used for the final spatial evaluation.

The spatial test was performed without automatic test-set scaling and without model retraining.

## Spatial validation results

| Metric                      | Spatial Test |
| --------------------------- | -----------: |
| N                           |          812 |
| MAE                         |   68.647 NTU |
| RMSE                        |  157.876 NTU |
| R²                          |       -0.107 |
| Bias (Observed − Predicted) |   54.998 NTU |
| Pearson r                   |        0.236 |

Observed mean turbidity:

```text
79.353 NTU
```

Predicted mean turbidity:

```text
24.354 NTU
```

The spatial validation demonstrates substantial deterioration in predictive performance when the Etawah-calibrated model is transferred to geographically distinct Yamuna stations.

---

# 10. Extreme Turbidity

A major limitation of the frozen model is its performance during high-turbidity conditions.

For observations with:

```text
Turbidity >= 100 NTU
```

the spatial test contains:

```text
158 observations
```

Performance in this category:

| Metric          |       Value |
| --------------- | ----------: |
| MAE             | 288.248 NTU |
| RMSE            | 355.417 NTU |
| Bias            | 288.248 NTU |
| Underprediction |        100% |

The model substantially underestimates high-turbidity observations.

There are:

```text
31 observations >= 500 NTU
```

in the spatial validation dataset.

These extreme observations have a strong influence on the overall RMSE and demonstrate that the model has limited ability to extrapolate beyond the turbidity regime represented during model development.

---

# 11. Cloud and Temporal-Gap Diagnostics

Cloud contamination was investigated as a potential contributor to spatial transfer failure.

Spatial validation MAE by cloud category:

| Cloud condition |   N | MAE (NTU) |
| --------------- | --: | --------: |
| ≤20%            | 428 |    38.344 |
| ≤40%            | 488 |    45.668 |
| ≤60%            | 557 |    50.617 |
| >60%            | 255 |   108.031 |

Higher cloud contamination is associated with larger errors.

However, cloud contamination does not fully explain the spatial validation failure.

Temporal-gap sensitivity was also evaluated:

| Maximum temporal gap |   N | MAE (NTU) |
| -------------------- | --: | --------: |
| ≤0.5 day             | 281 |    65.900 |
| ≤1 day               | 486 |    71.382 |
| ≤1.5 days            | 703 |    65.279 |
| ≤2 days              | 812 |    68.647 |

The relatively similar errors across temporal-gap thresholds indicate that temporal mismatch alone is unlikely to explain the observed spatial transfer failure.

---

# 12. Feature Distribution Shift

Feature distributions were compared between the Etawah development dataset and the independent spatial test dataset.

Mean values:

| Feature        | Development | Spatial Test |
| -------------- | ----------: | -----------: |
| MNDWI median   |      0.0598 |      -0.0489 |
| NDWI median    |     -0.0132 |      -0.1155 |
| MNDWI mean     |      0.0601 |      -0.0286 |
| NDWI mean      |     -0.0405 |      -0.1254 |
| Water fraction |      0.4873 |       0.4367 |
| B11 mean       |      0.1929 |       0.2588 |
| B12 mean       |      0.1641 |       0.2162 |
| B8 mean        |      0.2341 |       0.3512 |

The results indicate moderate covariate shift between the model-development environment and the independent spatial validation environment.

The shift is particularly evident in the reflectance variables and water-index distributions.

---

# 13. Main Findings

The study supports the following conclusions:

1. The Etawah-calibrated model shows limited but measurable temporal transferability to later Etawah observations.

2. Spatial transferability across the Yamuna River is substantially weaker than temporal transferability at the calibration station.

3. The model strongly underpredicts high-turbidity observations.

4. Cloud contamination contributes to prediction error but does not fully explain the spatial transfer failure.

5. Temporal matching uncertainty within the ±2-day matching window does not appear to be the dominant cause of the spatial validation failure.

6. Feature distribution differences between development and spatial-test datasets indicate covariate shift.

7. The results demonstrate the importance of geographically independent validation for remote-sensing water-quality models.

---

# 14. Repository Structure

```text
Yamuna-Turbidity-Sentinel2-ML/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── gee/
│   ├── 10_rebuild_etawah_temporal_matching.js
│   ├── 12_etawah_temporal_compact8_extraction.js
│   ├── 16_yamuna_expanded_spatial_matching.js
│   └── 17_yamuna_expanded_compact8_extraction.js
│
├── scripts/
│   ├── 01_independent_spatial_validation.py
│   ├── 02_frozen_model_spatial_test.py
│   ├── 11_verify_correct_temporal_match.py
│   ├── 12_verify_temporal_compact8.py
│   ├── 13_frozen_model_temporal_test.py
│   ├── 14_temporal_test_diagnostics.py
│   ├── 15_temporal_validation_figures.py
│   ├── 18_frozen_model_expanded_spatial_test.py
│   ├── 19_spatial_transfer_diagnostics.py
│   ├── 20_final_spatial_validation_figures.py
│   ├── 21_final_master_results_tables.py
│   └── 22_generate_manuscript_package.py
│
├── docs/
│   ├── methodology.md
│   ├── validation.md
│   └── reproducibility.md
│
├── figures/
│   └── manuscript/
│
└── data/
    └── README.md
```

---

# 15. Reproducibility Workflow

The recommended workflow is:

```text
1. Run GEE temporal matching
        ↓
2. Verify temporal matching
        ↓
3. Extract Sentinel-2 Compact-8 features
        ↓
4. Run frozen temporal validation
        ↓
5. Run GEE spatial matching
        ↓
6. Extract spatial Compact-8 features
        ↓
7. Run frozen spatial validation
        ↓
8. Run diagnostic analyses
        ↓
9. Generate final tables
        ↓
10. Generate publication figures
        ↓
11. Generate manuscript package
```

The validation datasets should not be used to tune model hyperparameters.

---

# 16. Important Validation Principle

The central methodological principle of this repository is:

> **The validation data are independent of model development.**

The Etawah model is frozen before independent temporal and spatial testing.

No hyperparameter tuning, feature selection, model retraining, or test-set scaling is performed using the independent validation observations.

This separation is important for evaluating true model transferability.

---



# 17. Data Availability

The repository contains analysis code, processing scripts, derived results, and publication figures.

Original Central Water Commission observations are not redistributed unless permitted under the applicable data-access and redistribution conditions.

Sentinel-2 imagery is accessed through Google Earth Engine and is not redistributed as part of this repository.

Users wishing to reproduce the analysis should obtain the required CWC observations through the appropriate data-access process and access Sentinel-2 imagery through Google Earth Engine.

---

# 18. Code Availability

The repository contains:

* Google Earth Engine temporal matching scripts
* Google Earth Engine spatial matching scripts
* Sentinel-2 feature extraction scripts
* Python model-validation scripts
* Temporal validation diagnostics
* Spatial transfer diagnostics
* Master result-table generation
* Publication figure generation
* Manuscript-supporting analysis

The Google Earth Engine scripts are intended to reproduce the CWC–Sentinel-2 matching and satellite feature extraction stages.

---

# 19. Software

The analysis was developed using Python and Google Earth Engine.

The Python workflow uses scientific and machine-learning libraries including:

```text
Python
pandas
numpy
scikit-learn
matplotlib
```

The machine-learning implementation uses `GradientBoostingRegressor` from scikit-learn.

Reference:

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

---

# 20. Citation

If you use this repository or build upon the methodology, please cite the associated research manuscript:

```text
[Manuscript citation to be added after publication]
```

---

# 21. License

The source code in this repository is intended to be released under an appropriate open-source license.

The license applies to code contributed to this repository and does not necessarily apply to third-party datasets or satellite products.

See:

```text
LICENSE
```

for the applicable terms.

---

# 22. Contact

For questions regarding the methodology, reproducibility, or code:

```text
Ajay B V
ajaybvachar21@gmail.com
```

---

## Project Status

**Status:** Research / manuscript preparation

The model-development and independent validation analyses reported in the associated manuscript have been completed.

The repository is intended to provide a reproducible record of the analysis workflow and supporting materials.

````
