// ============================================================
// STEP 17C — VRINDAVAN B3 MASK DIAGNOSTIC
// ============================================================

var S2_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED';

var sentinelId =
    '20231110T053039_20231110T053039_T43RGL';

var latitude = 27.565833329999997;
var longitude = 77.70777778000001;

var point = ee.Geometry.Point([
  longitude,
  latitude
]);

var patch = point
  .buffer(128)
  .bounds();

var image = ee.Image(
  ee.ImageCollection(S2_COLLECTION)
    .filter(
      ee.Filter.eq(
        'system:index',
        sentinelId
      )
    )
    .first()
);

print('Image:', image.get('system:index'));

print(
  'Image date:',
  ee.Date(
    image.get('system:time_start')
  ).format('YYYY-MM-dd HH:mm:ss')
);

// ------------------------------------------------------------
// RAW BANDS
// ------------------------------------------------------------

var B2 = image.select('B2');
var B3 = image.select('B3');
var B4 = image.select('B4');
var B8 = image.select('B8');
var B11 = image.select('B11');
var B12 = image.select('B12');

var SCL = image.select('SCL');

// ------------------------------------------------------------
// PIXEL COUNTS
// ------------------------------------------------------------

function countBand(band) {
  return band.reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: patch,
    scale: 10,
    maxPixels: 100000,
    bestEffort: true
  });
}

print('B2 valid pixels:', countBand(B2).get('B2'));
print('B3 valid pixels:', countBand(B3).get('B3'));
print('B4 valid pixels:', countBand(B4).get('B4'));
print('B8 valid pixels:', countBand(B8).get('B8'));
print('B11 valid pixels:', countBand(B11).get('B11'));
print('B12 valid pixels:', countBand(B12).get('B12'));
print('SCL valid pixels:', countBand(SCL).get('SCL'));

// ------------------------------------------------------------
// RAW B3 STATISTICS
// ------------------------------------------------------------

var B3stats = B3.reduceRegion({
  reducer:
    ee.Reducer.mean()
      .combine({
        reducer2: ee.Reducer.median(),
        sharedInputs: true
      })
      .combine({
        reducer2: ee.Reducer.minMax(),
        sharedInputs: true
      }),
  geometry: patch,
  scale: 10,
  maxPixels: 100000,
  bestEffort: true
});

print('B3 statistics:', B3stats);

// ------------------------------------------------------------
// SCL CLASS COUNTS
// ------------------------------------------------------------

var sclHistogram = SCL.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(),
  geometry: patch,
  scale: 10,
  maxPixels: 100000,
  bestEffort: true
});

print('SCL class histogram:', sclHistogram);

// ------------------------------------------------------------
// TEST WHETHER B3 CAN BE RECOVERED BY UNMASKING
// ------------------------------------------------------------

var B3_unmasked = B3.unmask(-9999);

var B3unmaskedStats = B3_unmasked.reduceRegion({
  reducer:
    ee.Reducer.mean()
      .combine({
        reducer2: ee.Reducer.median(),
        sharedInputs: true
      })
      .combine({
        reducer2: ee.Reducer.minMax(),
        sharedInputs: true
      }),
  geometry: patch,
  scale: 10,
  maxPixels: 100000,
  bestEffort: true
});

print(
  'B3 statistics after unmask(-9999):',
  B3unmaskedStats
);

// ------------------------------------------------------------
// MAP
// ------------------------------------------------------------

Map.centerObject(point, 13);

Map.addLayer(
  image,
  {
    bands: ['B4', 'B3', 'B2'],
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
  SCL,
  {
    min: 0,
    max: 11
  },
  'SCL'
);

Map.addLayer(
  point,
  {
    color: 'red'
  },
  'CWC point'
);

Map.addLayer(
  patch,
  {
    color: 'yellow'
  },
  '256 m patch'
);