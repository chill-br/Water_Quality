import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PATCH = Path(
    "data/patches/Etawah_1_20210101T052219_20210101T052222_T44RKQ.tif"
)

with rasterio.open(PATCH) as src:
    print("File:", PATCH.name)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Number of bands:", src.count)
    print("CRS:", src.crs)
    print("Transform:", src.transform)

    print("\nBands:")
    for i in range(1, src.count + 1):
        print(i, src.descriptions[i - 1])

    data = src.read().astype(np.float32)

# --------------------------------------------------
# Extract bands
# --------------------------------------------------

B2 = data[0]   # Blue
B3 = data[1]   # Green
B4 = data[2]   # Red
B8 = data[3]   # NIR
NDWI = data[6]
MNDWI = data[7]
MNDWI_WATER = data[8]

# --------------------------------------------------
# Function for percentile stretch
# --------------------------------------------------

def stretch(image, lower=2, upper=98):
    valid = image[np.isfinite(image)]

    if valid.size == 0:
        return np.zeros_like(image)

    vmin, vmax = np.percentile(valid, [lower, upper])

    if vmax == vmin:
        return np.zeros_like(image)

    result = (image - vmin) / (vmax - vmin)
    return np.clip(result, 0, 1)


# --------------------------------------------------
# RGB composites
# --------------------------------------------------

true_color = np.dstack([
    stretch(B4),
    stretch(B3),
    stretch(B2)
])

false_color = np.dstack([
    stretch(B8),
    stretch(B4),
    stretch(B3)
])

# --------------------------------------------------
# Print statistics
# --------------------------------------------------

print("\nBand statistics:")

for name, band in [
    ("B2", B2),
    ("B3", B3),
    ("B4", B4),
    ("B8", B8),
    ("NDWI", NDWI),
    ("MNDWI", MNDWI),
    ("MNDWI_WATER", MNDWI_WATER),
]:
    valid = band[np.isfinite(band)]

    print(
        f"{name:12s} "
        f"min={valid.min():.4f} "
        f"max={valid.max():.4f} "
        f"mean={valid.mean():.4f}"
    )

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# True color
axes[0, 0].imshow(true_color)
axes[0, 0].set_title("True Color (B4-B3-B2)")
axes[0, 0].axis("off")

# False color
axes[0, 1].imshow(false_color)
axes[0, 1].set_title("False Color (B8-B4-B3)")
axes[0, 1].axis("off")

# MNDWI
im = axes[1, 0].imshow(MNDWI, cmap="RdBu", vmin=-1, vmax=1)
axes[1, 0].set_title("MNDWI")
axes[1, 0].axis("off")
fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

# Water mask
im2 = axes[1, 1].imshow(MNDWI_WATER, cmap="gray", vmin=0, vmax=1)
axes[1, 1].set_title("MNDWI Water Mask")
axes[1, 1].axis("off")
fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

fig.suptitle(
    f"Etawah Sentinel-2 Patch\n{PATCH.name}",
    fontsize=14
)

plt.tight_layout()
plt.show()