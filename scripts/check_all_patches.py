import rasterio
import numpy as np
import pandas as pd
from pathlib import Path


PATCH_DIR = Path("data/patches")
OUTPUT = Path("data/patch_quality_control.csv")

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


rows = []

patches = sorted(PATCH_DIR.glob("*.tif"))

print(f"Found {len(patches)} patch files.")
print()


for i, patch in enumerate(patches, start=1):

    try:
        with rasterio.open(patch) as src:

            # ----------------------------------------
            # Basic metadata
            # ----------------------------------------

            width = src.width
            height = src.height
            band_count = src.count

            band_names = list(src.descriptions)

            # ----------------------------------------
            # Read data
            # ----------------------------------------

            data = src.read().astype(np.float32)

            # ----------------------------------------
            # Check expected bands
            # ----------------------------------------

            bands_ok = band_names == EXPECTED_BANDS

            # ----------------------------------------
            # Numerical validity
            # ----------------------------------------

            finite = np.isfinite(data).all()

            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()

            # ----------------------------------------
            # MNDWI
            # ----------------------------------------

            mndwi = data[7]

            mndwi_min = np.nanmin(mndwi)
            mndwi_max = np.nanmax(mndwi)
            mndwi_mean = np.nanmean(mndwi)

            # ----------------------------------------
            # Water mask
            # ----------------------------------------

            water_mask = data[8]

            unique_water = np.unique(water_mask)

            water_pixels = np.sum(water_mask == 1)
            total_pixels = water_mask.size

            water_fraction = water_pixels / total_pixels

            water_mask_binary = np.all(
                np.isin(unique_water, [0, 1])
            )

            # ----------------------------------------
            # Overall quality status
            # ----------------------------------------

            valid_dimensions = (
                (width == 13 and height == 14)
                or
                (width == 14 and height == 14)
            )

            quality_ok = (
                valid_dimensions
                and band_count == 9
                and bands_ok
                and finite
                and water_mask_binary
            )

            # ----------------------------------------
            # Store results
            # ----------------------------------------

            rows.append({
                "patch": patch.name,
                "width": width,
                "height": height,
                "bands": band_count,
                "bands_ok": bands_ok,
                "finite": finite,
                "nan_count": int(nan_count),
                "inf_count": int(inf_count),
                "mndwi_min": mndwi_min,
                "mndwi_max": mndwi_max,
                "mndwi_mean": mndwi_mean,
                "water_fraction": water_fraction,
                "water_mask_binary": water_mask_binary,
                "quality_ok": quality_ok,
            })

    except Exception as e:

        rows.append({
            "patch": patch.name,
            "width": None,
            "height": None,
            "bands": None,
            "bands_ok": False,
            "finite": False,
            "nan_count": None,
            "inf_count": None,
            "mndwi_min": None,
            "mndwi_max": None,
            "mndwi_mean": None,
            "water_fraction": None,
            "water_mask_binary": False,
            "quality_ok": False,
            "error": str(e),
        })


# --------------------------------------------------
# Create dataframe
# --------------------------------------------------

df = pd.DataFrame(rows)


# --------------------------------------------------
# Save report
# --------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(
    OUTPUT,
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("=" * 60)
print("PATCH QUALITY CONTROL")
print("=" * 60)

print(f"Total patches: {len(df)}")
print(f"Quality OK:    {df['quality_ok'].sum()}")
print(f"Quality FAIL:  {(~df['quality_ok']).sum()}")

print()

print("NaN values:", df["nan_count"].sum())
print("Inf values:", df["inf_count"].sum())

print()

print("Water fraction:")
print(
    f"Minimum: {df['water_fraction'].min():.3f}"
)
print(
    f"Maximum: {df['water_fraction'].max():.3f}"
)
print(
    f"Mean:    {df['water_fraction'].mean():.3f}"
)

print()


# --------------------------------------------------
# Show actual QC failures
# --------------------------------------------------

failed = df[
    ~df["quality_ok"]
]

print(
    f"Actual QC failures: {len(failed)}"
)

if len(failed) > 0:

    print()

    print(
        failed[
            [
                "patch",
                "width",
                "height",
                "bands_ok",
                "finite",
                "nan_count",
                "inf_count",
                "water_mask_binary",
                "quality_ok",
            ]
        ].to_string(index=False)
    )


# --------------------------------------------------
# Water-fraction extremes
# --------------------------------------------------

extreme_water = df[
    (df["water_fraction"] == 0)
    |
    (df["water_fraction"] == 1)
]

print()

print(
    f"Patches with 0% or 100% water mask: "
    f"{len(extreme_water)}"
)

print()

print(
    "These are NOT automatically failures."
)

print(
    "They are flagged only for scientific review."
)

print()

print(f"Report saved to: {OUTPUT}")
