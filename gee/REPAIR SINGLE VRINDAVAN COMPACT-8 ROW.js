// ============================================================
// STEP 17B — REPAIR SINGLE VRINDAVAN COMPACT-8 ROW
// ============================================================
//
// PURPOSE:
// Repair the missing water-derived Compact-8 features for:
//
// Station:
// Vrindavan-Yamuna ExpresswayLink Road Bridge U/S of Mathura
//
// CWC datetime:
// 10-11-2023 11:33
//
// Sentinel-2:
// 20231110T053039_20231110T053039_T43RGL
//
// IMPORTANT:
// - Does NOT redo Step 16 matching.
// - Does NOT retrain the model.
// - Uses the exact Sentinel-2 image already assigned.
// - Keeps the existing /10000 reflectance scaling.
// - Only calculates the five missing water-derived fields.
//
// ============================================================


// ------------------------------------------------------------
// 1. SETTINGS
// ------------------------------------------------------------

var S2_COLLECTION =
    'COPERNICUS/S2_SR_HARMONIZED';

var stationName =
    'Vrindavan-Yamuna ExpresswayLink Road Bridge U/S of Mathura';

var latitude =
    27.565833329999997;

var longitude =
    77.70777778000001;

var sentinelId =
    '20231110T053039_20231110T053039_T43RGL';


// ------------------------------------------------------------
// 2. STATION POINT
// ------------------------------------------------------------

var point =
    ee.Geometry.Point([
      longitude,
      latitude
    ]);


// ------------------------------------------------------------
// 3. GET EXACT SENTINEL-2 IMAGE
// ------------------------------------------------------------

var s2 =
    ee.ImageCollection(
      S2_COLLECTION
    )
    .filter(
      ee.Filter.eq(
        'system:index',
        sentinelId
      )
    );

print(
  'Sentinel-2 image count:',
  s2.size()
);


// ------------------------------------------------------------
// 4. GET IMAGE
// ------------------------------------------------------------

var image =
    ee.Image(
      s2.first()
    );

print(
  'Image:',
  image
);

print(
  'Image ID:',
  image.get('system:index')
);

print(
  'Image date:',
  ee.Date(
    image.get('system:time_start')
  ).format(
    'YYYY-MM-dd HH:mm:ss'
  )
);


// ------------------------------------------------------------
// 5. SCALE SENTINEL-2 REFLECTANCE
// ------------------------------------------------------------

var B2 =
    image.select('B2')
    .divide(10000);

var B3 =
    image.select('B3')
    .divide(10000);

var B4 =
    image.select('B4')
    .divide(10000);

var B8 =
    image.select('B8')
    .divide(10000);

var B11 =
    image.select('B11')
    .divide(10000);

var B12 =
    image.select('B12')
    .divide(10000);


// ------------------------------------------------------------
// 6. CALCULATE INDICES
// ------------------------------------------------------------

var NDWI =
    B3.subtract(B8)
      .divide(
        B3.add(B8)
      )
      .rename('NDWI');


var MNDWI =
    B3.subtract(B11)
      .divide(
        B3.add(B11)
      )
      .rename('MNDWI');


// ------------------------------------------------------------
// 7. WATER MASK
// ------------------------------------------------------------
//
// Same definition used in Step 17:
// MNDWI > 0
//
// ------------------------------------------------------------

var waterMask =
    MNDWI.gt(0)
    .rename('water_mask');


// ------------------------------------------------------------
// 8. 256 m PATCH
// ------------------------------------------------------------

var patch =
    point
      .buffer(128)
      .bounds();


// ------------------------------------------------------------
// 9. WATER-ONLY IMAGE
// ------------------------------------------------------------

var waterImage =
    ee.Image.cat([
      NDWI,
      MNDWI
    ])
    .updateMask(
      waterMask
    );


// ------------------------------------------------------------
// 10. WATER STATISTICS
// ------------------------------------------------------------

var waterStats =
    waterImage.reduceRegion({

      reducer:
        ee.Reducer.mean()
        .combine({
          reducer2:
            ee.Reducer.median(),
          sharedInputs:
            true
        }),

      geometry:
        patch,

      scale:
        10,

      maxPixels:
        100000

    });


// ------------------------------------------------------------
// 11. WATER FRACTION
// ------------------------------------------------------------

var waterFraction =
    waterMask
      .reduceRegion({

        reducer:
          ee.Reducer.mean(),

        geometry:
          patch,

        scale:
          10,

        maxPixels:
          100000

      })
      .get('water_mask');


// ------------------------------------------------------------
// 12. PRINT ACTUAL VALUES
// ------------------------------------------------------------
//
// IMPORTANT:
// These are the five values needed to repair the CSV.
//
// ------------------------------------------------------------

print('');
print('================================================');
print('REPAIRED VRINDAVAN COMPACT-8 WATER FEATURES');
print('================================================');

print(
  'Station:',
  stationName
);

print(
  'Latitude:',
  latitude
);

print(
  'Longitude:',
  longitude
);

print(
  'Sentinel-2:',
  sentinelId
);

print('');


// MNDWI mean

print(
  'MNDWI_mean:',
  waterStats.get(
    'MNDWI_mean'
  )
);


// MNDWI median

print(
  'MNDWI_median:',
  waterStats.get(
    'MNDWI_median'
  )
);


// NDWI mean

print(
  'NDWI_mean:',
  waterStats.get(
    'NDWI_mean'
  )
);


// NDWI median

print(
  'NDWI_median:',
  waterStats.get(
    'NDWI_median'
  )
);


// Water fraction

print(
  'water_fraction:',
  waterFraction
);


print('');
print('================================================');
print('END OF VRINDAVAN REPAIR');
print('================================================');


// ------------------------------------------------------------
// 13. MAP
// ------------------------------------------------------------

Map.centerObject(
  point,
  13
);

Map.addLayer(
  image,
  {
    bands: [
      'B4',
      'B3',
      'B2'
    ],
    min: 0,
    max: 3000
  },
  'Sentinel-2 RGB'
);

Map.addLayer(
  MNDWI,
  {
    min: -1,
    max: 1
  },
  'MNDWI'
);

Map.addLayer(
  waterMask,
  {
    min: 0,
    max: 1
  },
  'Water mask'
);

Map.addLayer(
  point,
  {
    color: 'red'
  },
  'Vrindavan CWC station'
);