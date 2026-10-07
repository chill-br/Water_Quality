import rasterio
from pathlib import Path
from collections import Counter


PATCH_DIR = Path("data/patches")

patches = sorted(PATCH_DIR.glob("*.tif"))

print("=" * 70)
print("PATCH GROUP DIAGNOSTIC")
print("=" * 70)

print(f"Total TIFF files: {len(patches)}")
print()


# --------------------------------------------------
# 1. Numbered patch files
# --------------------------------------------------

numbers = []

for patch in patches:
    try:
        number = int(patch.stem.split("_")[1])
        numbers.append(number)
    except Exception:
        pass

numbers = sorted(numbers)

print("Patch number range:")
print(f"Minimum: {min(numbers)}")
print(f"Maximum: {max(numbers)}")

missing = sorted(set(range(1, 209)) - set(numbers))

print()
print("Missing patch numbers:")

if missing:
    print(missing)
else:
    print("None")

print()


# --------------------------------------------------
# 2. Examine band structures
# --------------------------------------------------

band_structures = Counter()

for patch in patches:

    try:
        with rasterio.open(patch) as src:

            structure = (
                src.count,
                tuple(src.descriptions),
                src.width,
                src.height,
                str(src.crs),
            )

            band_structures[structure] += 1

    except Exception as e:
        print("ERROR:", patch.name, e)


print("=" * 70)
print("UNIQUE PATCH STRUCTURES")
print("=" * 70)

for i, (structure, count) in enumerate(
    band_structures.items(), start=1
):

    count_bands, descriptions, width, height, crs = structure

    print()
    print(f"STRUCTURE {i}")
    print(f"Number of files: {count}")
    print(f"Dimensions: {width} x {height}")
    print(f"Band count: {count_bands}")
    print(f"CRS: {crs}")

    print("Bands:")

    for band_number, description in enumerate(
        descriptions,
        start=1
    ):
        print(f"  {band_number}: {description}")


# --------------------------------------------------
# 3. Show representative files from each group
# --------------------------------------------------

print()
print("=" * 70)
print("REPRESENTATIVE FILES")
print("=" * 70)

seen = set()

for patch in patches:

    try:
        with rasterio.open(patch) as src:

            structure = (
                src.count,
                tuple(src.descriptions),
                src.width,
                src.height,
                str(src.crs),
            )

            if structure not in seen:

                seen.add(structure)

                print()
                print("File:")
                print(patch.name)

                print("Dimensions:")
                print(src.width, "x", src.height)

                print("Bands:")
                print(src.descriptions)

                print("CRS:")
                print(src.crs)

    except Exception:
        pass