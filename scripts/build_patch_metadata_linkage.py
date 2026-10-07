import pandas as pd
import re
from pathlib import Path

DATA_DIR = Path("data")
PATCH_DIR = DATA_DIR / "patches"

MATCHED_FILE = DATA_DIR / "Etawah_CWC_Sentinel2_Matched_Observations (1).csv"
SPECTRAL_FILE = DATA_DIR / "Etawah_CWC_Sentinel2_Spectral_Indices_2021_2025 (1).csv"
OUTPUT_FILE = DATA_DIR / "Etawah_patch_metadata_linkage.csv"


# =========================================================
# 1. Load source tables
# =========================================================

matched = pd.read_csv(MATCHED_FILE)
spectral = pd.read_csv(SPECTRAL_FILE)

print("Original matched table:", matched.shape)
print("Original spectral table:", spectral.shape)


# =========================================================
# 2. Keep ETAWAH only
# =========================================================

matched = matched[
    matched["station"].astype(str).str.strip().str.upper() == "ETAWAH"
].copy()

spectral = spectral[
    spectral["station"].astype(str).str.strip().str.upper() == "ETAWAH"
].copy()

print("\nAfter ETAWAH filtering:")
print("Matched ETAWAH:", matched.shape)
print("Spectral ETAWAH:", spectral.shape)


# =========================================================
# 3. Normalize fields
# =========================================================

matched["sentinel2_id"] = matched["sentinel2_id"].astype(str).str.strip()
spectral["satellite_id"] = spectral["satellite_id"].astype(str).str.strip()

matched["cwc_datetime"] = matched["cwc_datetime"].astype(str).str.strip()
spectral["cwc_datetime"] = spectral["cwc_datetime"].astype(str).str.strip()


# =========================================================
# 4. Read the 208 patch filenames
# =========================================================

records = []

pattern = re.compile(r"^Etawah_(\d+)_(.+)\.tif$")

for tif in sorted(PATCH_DIR.glob("Etawah_*.tif")):

    m = pattern.match(tif.name)

    if not m:
        print("WARNING: Could not parse:", tif.name)
        continue

    records.append({
        "patch_number": int(m.group(1)),
        "patch_filename": tif.name,
        "sentinel2_id": m.group(2),
    })

patches = pd.DataFrame(records)

print("\nPatch files:", len(patches))


# =========================================================
# 5. Verify patch numbering
# =========================================================

expected = set(range(1, 209))
actual = set(patches["patch_number"])

missing = sorted(expected - actual)
extra = sorted(actual - expected)

print("Missing patch numbers:", missing if missing else "None")
print("Unexpected patch numbers:", extra if extra else "None")


# =========================================================
# 6. Identify repeated Sentinel IDs among patches
# =========================================================

patch_counts = (
    patches.groupby("sentinel2_id")
    .size()
    .reset_index(name="sentinel_patch_count")
)

patches = patches.merge(
    patch_counts,
    on="sentinel2_id",
    how="left"
)

patches["duplicate_sentinel_id"] = (
    patches["sentinel_patch_count"] > 1
)


# =========================================================
# 7. Prepare spectral table
# =========================================================

spectral = spectral[
    [
        "satellite_id",
        "station",
        "latitude",
        "longitude",
        "cwc_datetime",
        "turbidity_NTU",
        "satellite_date",
        "days_difference",
        "B2",
        "B3",
        "B4",
        "B8",
        "B11",
        "B12",
        "NDWI",
        "MNDWI",
    ]
].copy()

spectral = spectral.rename(
    columns={"satellite_id": "sentinel2_id"}
)


# =========================================================
# 8. Create an exact observation key
# =========================================================

spectral["link_key"] = (
    spectral["sentinel2_id"]
    + "|"
    + spectral["cwc_datetime"]
)

matched["link_key"] = (
    matched["sentinel2_id"]
    + "|"
    + matched["cwc_datetime"]
)


# =========================================================
# 9. Remove exact duplicate spectral observations
# =========================================================

before = len(spectral)

spectral = spectral.drop_duplicates(
    subset=["link_key"],
    keep="first"
)

print(
    "Exact duplicate spectral observations removed:",
    before - len(spectral)
)


# =========================================================
# 10. Check whether a Sentinel ID maps to multiple
#     ETAWAH observations
# =========================================================

id_counts = (
    spectral.groupby("sentinel2_id")
    .size()
    .reset_index(name="spectral_records_per_sentinel")
)

print(
    "\nSentinel IDs with >1 ETAWAH spectral records:",
    (id_counts["spectral_records_per_sentinel"] > 1).sum()
)


# =========================================================
# 11. IMPORTANT:
#     Match patches to observations using Sentinel ID
#     AND the patch's intended ETAWAH observation.
#
#     The patch filename contains only Sentinel ID.
#     Therefore, if a Sentinel ID has multiple ETAWAH
#     observations, we need to resolve which CWC record
#     generated that patch.
#
#     The 244-row spectral table contains those observations.
#     We therefore inspect ambiguity rather than silently
#     duplicating rows.
# =========================================================

spectral_per_id = (
    spectral.groupby("sentinel2_id")
    .size()
    .rename("spectral_records_per_sentinel")
)

patches["spectral_records_per_sentinel"] = (
    patches["sentinel2_id"]
    .map(spectral_per_id)
)


# =========================================================
# 12. For each patch, determine candidate ETAWAH records
# =========================================================

candidate_counts = (
    spectral.groupby("sentinel2_id")
    .size()
)

patches["candidate_cwc_records"] = (
    patches["sentinel2_id"]
    .map(candidate_counts)
    .fillna(0)
    .astype(int)
)


# =========================================================
# 13. The original GEE workflow produced 208 matched
#     observations. We need to identify those observations.
#
#     The safest source is the 123-row matched table,
#     supplemented by the spectral table.
# =========================================================

matched_lookup = matched[
    [
        "link_key",
        "sentinel2_id",
        "cwc_datetime",
        "cloud_percentage",
    ]
].copy()


# Create dictionary of cloud values by exact observation
cloud_lookup = (
    matched_lookup
    .drop_duplicates("link_key")
    .set_index("link_key")["cloud_percentage"]
    .to_dict()
)


# =========================================================
# 14. Because the filename only identifies Sentinel ID,
#     choose the ETAWAH spectral observation with the
#     closest CWC date to the Sentinel acquisition date.
#
#     This gives one row per patch while retaining the
#     original CWC observation metadata.
# =========================================================

spectral["satellite_date_parsed"] = pd.to_datetime(
    spectral["satellite_date"],
    errors="coerce"
)

spectral["cwc_datetime_parsed"] = pd.to_datetime(
    spectral["cwc_datetime"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

spectral["date_distance_hours"] = (
    (
        spectral["cwc_datetime_parsed"]
        - spectral["satellite_date_parsed"]
    )
    .abs()
    .dt.total_seconds()
    / 3600
)


# =========================================================
# 15. Select one record for each Sentinel ID
#
#     If multiple CWC observations share the same Sentinel
#     image, retain the closest CWC observation in time.
# =========================================================

spectral_sorted = spectral.sort_values(
    [
        "sentinel2_id",
        "date_distance_hours",
        "cwc_datetime_parsed",
    ]
)

best_spectral = (
    spectral_sorted
    .drop_duplicates(
        subset=["sentinel2_id"],
        keep="first"
    )
    .copy()
)


# =========================================================
# 16. Merge one spectral/CWC record onto each patch
# =========================================================

linkage = patches.merge(
    best_spectral,
    on="sentinel2_id",
    how="left",
    suffixes=("", "_spectral")
)


# =========================================================
# 17. Add cloud percentage from matched observations
# =========================================================

linkage["link_key"] = (
    linkage["sentinel2_id"]
    + "|"
    + linkage["cwc_datetime"].astype(str).str.strip()
)

linkage["cloud_percentage"] = (
    linkage["link_key"]
    .map(cloud_lookup)
)


# =========================================================
# 18. Linkage status
# =========================================================

def status(row):

    if pd.isna(row["station"]):
        return "NO_CWC_METADATA"

    if row["candidate_cwc_records"] > 1:
        return "DUPLICATE_SENTINEL_ID_REVIEW"

    return "OK"


linkage["linkage_status"] = linkage.apply(
    status,
    axis=1
)


# =========================================================
# 19. Final columns
# =========================================================

final_columns = [
    "patch_number",
    "patch_filename",
    "sentinel2_id",
    "satellite_date",
    "station",
    "cwc_datetime",
    "latitude",
    "longitude",
    "turbidity_NTU",
    "days_difference",
    "cloud_percentage",
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
    "NDWI",
    "MNDWI",
    "sentinel_patch_count",
    "candidate_cwc_records",
    "duplicate_sentinel_id",
    "linkage_status",
]

linkage = linkage[final_columns]

linkage = linkage.sort_values(
    "patch_number"
).reset_index(drop=True)


# =========================================================
# 20. Save
# =========================================================

linkage.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# 21. Final report
# =========================================================

print("\n" + "=" * 60)
print("ETAWAH PATCH LINKAGE")
print("=" * 60)

print("Rows:", len(linkage))
print("Columns:", len(linkage.columns))

print(
    "Unique patch numbers:",
    linkage["patch_number"].nunique()
)

print(
    "Unique Sentinel-2 IDs:",
    linkage["sentinel2_id"].nunique()
)

print("\nLinkage status:")
print(
    linkage["linkage_status"]
    .value_counts(dropna=False)
)

print(
    "\nMissing station:",
    linkage["station"].isna().sum()
)

print(
    "Missing turbidity:",
    linkage["turbidity_NTU"].isna().sum()
)

print(
    "\nDuplicate Sentinel IDs:",
    linkage["duplicate_sentinel_id"].sum()
)

print("\nFirst 10 rows:")
print(
    linkage.head(10).to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)