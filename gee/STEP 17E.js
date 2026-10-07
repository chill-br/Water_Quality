// ============================================================
// STEP 17E
// COMPLETE VRINDAVAN SENTINEL-2 DIAGNOSTIC
// ============================================================
//
// Purpose:
// Diagnose the exact reason why B2/B3 are completely masked
// for the Vrindavan observation.
//
// Station:
// Vrindavan-Yamuna ExpresswayLink Road Bridge U/S of Mathura
//
// Sentinel-2:
// 20231110T053039_20231110T053039_T43RGL
//
// IMPORTANT:
// This program does NOT modify any CSV or export anything.
// It is diagnostic only.
// ============================================================


// ------------------------------------------------------------
// 1. SETTINGS
// ------------------------------------------------------------

var COLLECTION =
    'COPERNICUS/S2_SR_HARMONIZED';

var SENTINEL_ID =
    '20231110T053039_20231110T053039_T43RGL';

var LAT =
    27.565833329999997;

var LON =
    77.70777778000001;

var POINT =
    ee.Geometry.Point([
      LON,
      LAT
    ]);

var PATCH =
    POINT
      .buffer(128)
      .bounds();


// ------------------------------------------------------------
// 2. GET EXACT SENTINEL-2 IMAGE
// ------------------------------------------------------------

var IMAGE =
    ee.Image(
      ee.ImageCollection(COLLECTION)
        .filter(
          ee.Filter.eq(
            'system:index',
            SENTINEL_ID
          )
        )
        .first()
    );


// ------------------------------------------------------------
// 3. BASIC IMAGE INFORMATION
// ------------------------------------------------------------

print('================================================');
print('1. IMAGE INFORMATION');
print('================================================');

print(
  'Sentinel-2 ID:',
  IMAGE.get('system:index')
);

print(
  'Image date:',
  ee.Date(
    IMAGE.get('system:time_start')
  ).format(
    'YYYY-MM-dd HH:mm:ss'
  )
);

print(
  'Cloud percentage:',
  IMAGE.get('CLOUDY_PIXEL_PERCENTAGE')
);

print(
  'Tile:',
  IMAGE.get('MGRS_TILE')
);

print(
  'Processing baseline:',
  IMAGE.get('PROCESSING_BASELINE')
);

print(
  'Product type:',
  IMAGE.get('PRODUCT_TYPE')
);

print(
  'Image footprint:',
  IMAGE.geometry()
);


// ------------------------------------------------------------
// 4. CHECK AVAILABLE BANDS
// ------------------------------------------------------------

print('================================================');
print('2. AVAILABLE IMAGE BANDS');
print('================================================');

print(
  'Band names:',
  IMAGE.bandNames()
);


// ------------------------------------------------------------
// 5. SELECT RAW BANDS
// ------------------------------------------------------------

var B2 =
    IMAGE.select('B2');

var B3 =
    IMAGE.select('B3');

var B4 =
    IMAGE.select('B4');

var B8 =
    IMAGE.select('B8');

var B11 =
    IMAGE.select('B11');

var B12 =
    IMAGE.select('B12');

var SCL =
    IMAGE.select('SCL');


// ------------------------------------------------------------
// 6. FUNCTION: VALID PIXEL COUNT
// ------------------------------------------------------------

function validCount(
  band,
  name
) {

  var result =
      band.reduceRegion({

        reducer:
          ee.Reducer.count(),

        geometry:
          PATCH,

        scale:
          10,

        maxPixels:
          100000,

        bestEffort:
          true

      });

  print(
    name + ' valid pixels:',
    result.get(
      name
    )
  );

}


// ------------------------------------------------------------
// 7. VALID PIXEL COUNTS
// ------------------------------------------------------------

print('================================================');
print('3. VALID PIXEL COUNTS');
print('================================================');

validCount(B2, 'B2');
validCount(B3, 'B3');
validCount(B4, 'B4');
validCount(B8, 'B8');
validCount(B11, 'B11');
validCount(B12, 'B12');
validCount(SCL, 'SCL');


// ------------------------------------------------------------
// 8. FUNCTION: RAW BAND STATISTICS
// ------------------------------------------------------------

function bandStats(
  band,
  name
) {

  var stats =
      band.reduceRegion({

        reducer:
          ee.Reducer.mean()
            .combine({
              reducer2:
                ee.Reducer.median(),
              sharedInputs:
                true
            })
            .combine({
              reducer2:
                ee.Reducer.minMax(),
              sharedInputs:
                true
            }),

        geometry:
          PATCH,

        scale:
          10,

        maxPixels:
          100000,

        bestEffort:
          true

      });

  print(
    name + ' mean:',
    stats.get(
      name + '_mean'
    )
  );

  print(
    name + ' median:',
    stats.get(
      name + '_median'
    )
  );

  print(
    name + ' min:',
    stats.get(
      name + '_min'
    )
  );

  print(
    name + ' max:',
    stats.get(
      name + '_max'
    )
  );

}


// ------------------------------------------------------------
// 9. RAW BAND STATISTICS
// ------------------------------------------------------------

print('================================================');
print('4. RAW BAND STATISTICS');
print('================================================');

bandStats(B2, 'B2');
bandStats(B3, 'B3');
bandStats(B4, 'B4');
bandStats(B8, 'B8');
bandStats(B11, 'B11');
bandStats(B12, 'B12');


// ------------------------------------------------------------
// 10. CHECK MASK DIRECTLY
// ------------------------------------------------------------
//
// IMAGE.mask() tells us which pixels are actually valid.
//
// A value of 0 = masked.
// A value of 1 = valid.
//
// ------------------------------------------------------------

print('================================================');
print('5. BAND MASK COUNTS');
print('================================================');

var B2_MASK =
    B2.mask()
      .rename('B2_mask');

var B3_MASK =
    B3.mask()
      .rename('B3_mask');

var B4_MASK =
    B4.mask()
      .rename('B4_mask');

var B8_MASK =
    B8.mask()
      .rename('B8_mask');

var B11_MASK =
    B11.mask()
      .rename('B11_mask');

var B12_MASK =
    B12.mask()
      .rename('B12_mask');


print(
  'B2 mask mean:',
  B2_MASK.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: PATCH,
    scale: 10,
    maxPixels: 100000,
    bestEffort: true
  }).get('B2_mask')
);

print(
  'B3 mask mean:',
  B3_MASK.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: PATCH,
    scale: 10,
    maxPixels: 100000,
    bestEffort: true
  }).get('B3_mask')
);

print(
  'B4 mask mean:',
  B4_MASK.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: PATCH,
    scale: 10,
    maxPixels: 100000,
    bestEffort: true
  }).get('B4_mask')
);

print(
  'B8 mask mean:',
  B8_MASK.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: PATCH,
    scale: 10,
    maxPixels: 100000,
    bestEffort: true
  }).get('B8_mask')
);

print(
  'B11 mask mean:',
  B11_MASK.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: PATCH,
    scale: 10,
    maxPixels: 100000,
    bestEffort: true
  }).get('B11_mask')
);

print(
  'B12 mask mean:',
  B12_MASK.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: PATCH,
    scale: 10,
    maxPixels: 100000,
    bestEffort: true
  }).get('B12_mask')
);


// ------------------------------------------------------------
// 11. SCL CLASS HISTOGRAM — CORRECT METHOD
// ------------------------------------------------------------
//
// Instead of trying to retrieve nonexistent keys,
// convert the histogram into a FeatureCollection.
//
// This safely handles classes that are absent.
//
// ------------------------------------------------------------

print('================================================');
print('6. SCL CLASS COUNTS');
print('================================================');

var SCL_HIST =
    ee.Dictionary(
      SCL.reduceRegion({

        reducer:
          ee.Reducer.frequencyHistogram(),

        geometry:
          PATCH,

        scale:
          10,

        maxPixels:
          100000,

        bestEffort:
          true

      }).get('SCL')
    );

print(
  'Complete SCL histogram:',
  SCL_HIST
);


// ------------------------------------------------------------
// 12. PRINT EACH POSSIBLE SCL CLASS SAFELY
// ------------------------------------------------------------
//
// Default value = 0 if class is absent.
//
// Sentinel-2 SCL:
//
// 0 = No data
// 1 = Saturated / defective
// 2 = Dark area / shadows
// 3 = Cloud shadows
// 4 = Vegetation
// 5 = Bare soil
// 6 = Water
// 7 = Unclassified
// 8 = Cloud medium probability
// 9 = Cloud high probability
// 10 = Thin cirrus
// 11 = Snow / ice
//
// ------------------------------------------------------------

var SCL_CLASSES =
    ee.List.sequence(
      0,
      11
    );

var SCL_TABLE =
    ee.FeatureCollection(
      SCL_CLASSES.map(
        function(classNumber) {

          classNumber =
              ee.Number(
                classNumber
              );

          var key =
              classNumber.format();

          var count =
              ee.Number(
                SCL_HIST.get(
                  key,
                  0
                )
              );

          return ee.Feature(
            null,
            {
              SCL_class:
                classNumber,

              pixel_count:
                count
            }
          );

        }
      )
    );

print(
  'SCL class table:',
  SCL_TABLE
);


// ------------------------------------------------------------
// 13. SCL CLASS PERCENTAGES
// ------------------------------------------------------------

var TOTAL_SCL_PIXELS =
    ee.Number(
      SCL_TABLE.aggregate_sum(
        'pixel_count'
      )
    );

var SCL_PERCENT_TABLE =
    SCL_TABLE.map(
      function(feature) {

        var count =
            ee.Number(
              feature.get(
                'pixel_count'
              )
            );

        return feature.set(
          'percentage',
          count
            .divide(
              TOTAL_SCL_PIXELS
            )
            .multiply(100)
        );

      }
    );

print(
  'SCL class percentages:',
  SCL_PERCENT_TABLE
);


// ------------------------------------------------------------
// 14. EXPLICIT CLOUD COUNTS
// ------------------------------------------------------------

var CLOUD_MASK =
    SCL.eq(8)
      .or(
        SCL.eq(9)
      )
      .or(
        SCL.eq(10)
      )
      .rename(
        'cloud_mask'
      );

var CLOUD_COUNT =
    CLOUD_MASK
      .reduceRegion({

        reducer:
          ee.Reducer.sum(),

        geometry:
          PATCH,

        scale:
          10,

        maxPixels:
          100000,

        bestEffort:
          true

      })
      .get(
        'cloud_mask'
      );


var CLOUD_FRACTION =
    CLOUD_MASK
      .reduceRegion({

        reducer:
          ee.Reducer.mean(),

        geometry:
          PATCH,

        scale:
          10,

        maxPixels:
          100000,

        bestEffort:
          true

      })
      .get(
        'cloud_mask'
      );


print(
  'Cloud/cirrus pixel count:',
  CLOUD_COUNT
);

print(
  'Cloud/cirrus fraction:',
  CLOUD_FRACTION
);

print(
  'Cloud/cirrus percentage:',
  ee.Number(
    CLOUD_FRACTION
  ).multiply(100)
);


// ------------------------------------------------------------
// 15. WATER CLASS COUNT
// ------------------------------------------------------------

var WATER_CLASS =
    SCL.eq(6)
      .rename(
        'water_class'
      );

print(
  'SCL water-class pixel count:',
  WATER_CLASS.reduceRegion({

    reducer:
      ee.Reducer.sum(),

    geometry:
      PATCH,

    scale:
      10,

    maxPixels:
      100000,

    bestEffort:
      true

  }).get(
    'water_class'
  )
);


// ------------------------------------------------------------
// 16. UNDERLYING B3 CHECK
// ------------------------------------------------------------
//
// If B3 has no data at all, unmask() will give -9999.
// This confirms whether there is any recoverable B3 value.
//
// ------------------------------------------------------------

print('================================================');
print('7. UNDERLYING B3 / B2 CHECK');
print('================================================');

var B2_UNMASKED =
    B2.unmask(
      -9999
    );

var B3_UNMASKED =
    B3.unmask(
      -9999
    );


var B2_UNMASKED_STATS =
    B2_UNMASKED.reduceRegion({

      reducer:
        ee.Reducer.mean()
          .combine({
            reducer2:
              ee.Reducer.minMax(),
            sharedInputs:
              true
          }),

      geometry:
        PATCH,

      scale:
        10,

      maxPixels:
        100000,

      bestEffort:
        true

    });


var B3_UNMASKED_STATS =
    B3_UNMASKED.reduceRegion({

      reducer:
        ee.Reducer.mean()
          .combine({
            reducer2:
              ee.Reducer.minMax(),
            sharedInputs:
              true
          }),

      geometry:
        PATCH,

      scale:
        10,

      maxPixels:
        100000,

      bestEffort:
        true

    });


print(
  'B2 unmasked mean:',
  B2_UNMASKED_STATS.get(
    'B2_mean'
  )
);

print(
  'B2 unmasked min:',
  B2_UNMASKED_STATS.get(
    'B2_min'
  )
);

print(
  'B2 unmasked max:',
  B2_UNMASKED_STATS.get(
    'B2_max'
  )
);

print(
  'B3 unmasked mean:',
  B3_UNMASKED_STATS.get(
    'B3_mean'
  )
);

print(
  'B3 unmasked min:',
  B3_UNMASKED_STATS.get(
    'B3_min'
  )
);

print(
  'B3 unmasked max:',
  B3_UNMASKED_STATS.get(
    'B3_max'
  )
);


// ------------------------------------------------------------
// 17. TEST INDEX USING RAW AVAILABLE BANDS
// ------------------------------------------------------------
//
// This will remain null if B3 is unavailable.
// It is included only to document the result.
// ------------------------------------------------------------

var B3_SCALED =
    B3.divide(10000);

var B8_SCALED =
    B8.divide(10000);

var B11_SCALED =
    B11.divide(10000);


var NDWI =
    B3_SCALED
      .subtract(
        B8_SCALED
      )
      .divide(
        B3_SCALED
          .add(
            B8_SCALED
          )
      )
      .rename(
        'NDWI'
      );


var MNDWI =
    B3_SCALED
      .subtract(
        B11_SCALED
      )
      .divide(
        B3_SCALED
          .add(
            B11_SCALED
          )
      )
      .rename(
        'MNDWI'
      );


print('================================================');
print('8. INDEX AVAILABILITY');
print('================================================');

print(
  'NDWI valid pixels:',
  NDWI.reduceRegion({

    reducer:
      ee.Reducer.count(),

    geometry:
      PATCH,

    scale:
      10,

    maxPixels:
      100000,

    bestEffort:
      true

  }).get(
    'NDWI'
  )
);

print(
  'MNDWI valid pixels:',
  MNDWI.reduceRegion({

    reducer:
      ee.Reducer.count(),

    geometry:
      PATCH,

    scale:
      10,

    maxPixels:
      100000,

    bestEffort:
      true

  }).get(
    'MNDWI'
  )
);


// ------------------------------------------------------------
// 18. MAP VISUALIZATION
// ------------------------------------------------------------

Map.centerObject(
  POINT,
  13
);

Map.addLayer(
  IMAGE,
  {
    bands: [
      'B4',
      'B3',
      'B2'
    ],
    min: 0,
    max: 3000
  },
  'RGB'
);

Map.addLayer(
  B3,
  {
    min: 0,
    max: 3000
  },
  'B3'
);

Map.addLayer(
  B8,
  {
    min: 0,
    max: 5000
  },
  'B8'
);

Map.addLayer(
  SCL,
  {
    min: 0,
    max: 11
  },
  'SCL'
);

Map.addLayer(
  CLOUD_MASK,
  {
    min: 0,
    max: 1
  },
  'Cloud / Cirrus'
);

Map.addLayer(
  POINT,
  {
    color: 'red'
  },
  'CWC point'
);

Map.addLayer(
  PATCH,
  {
    color: 'yellow'
  },
  '256 m patch'
);


// ------------------------------------------------------------
// 19. FINAL SUMMARY
// ------------------------------------------------------------

print('================================================');
print('9. FINAL DIAGNOSTIC SUMMARY');
print('================================================');

print(
  'Station:',
  'Vrindavan-Yamuna ExpresswayLink Road Bridge U/S of Mathura'
);

print(
  'Sentinel-2:',
  SENTINEL_ID
);

print(
  'Patch size:',
  '256 m'
);

print(
  'Expected diagnosis:',
  'B2/B3 masking, SCL distribution, and index availability'
);

print('================================================');
print('END OF STEP 17E');
print('================================================');