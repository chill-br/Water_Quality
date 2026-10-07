// ============================================================
// REPAIR ONE MISSING COMPACT-8 WATER FEATURE ROW
// Vrindavan-Yamuna ExpresswayLink Road Bridge U/S of Mathura
// ============================================================

var stationName =
  'Vrindavan-Yamuna ExpresswayLink Road Bridge U/S of Mathura';

var lat = 27.565833329999997;
var lon = 77.70777778000001;

var sentinelId =
  '20231110T053039_20231110T053039_T43RGL';

var point = ee.Geometry.Point([lon, lat]);

var patch = point.buffer(128).bounds();


// ============================================================
// FIND EXACT SENTINEL-2 IMAGE
// ============================================================

var s2 = ee.ImageCollection(
  'COPERNICUS/S2_SR_HARMONIZED'
)
.filter(
  ee.Filter.eq(
    'system:index',
    sentinelId
  )
);

print('Sentinel-2 image count:', s2.size());

var image = ee.Image(
  s2.first()
);

print('Image:', image);
print('Image ID:', image.get('system:index'));
print(
  'Image date:',
  ee.Date(
    image.get('system:time_start')
  ).format('YYYY-MM-dd HH:mm:ss')
);


// ============================================================
// SCALE SENTINEL-2 REFLECTANCE
// ============================================================

var B2 = image.select('B2').divide(10000);
var B3 = image.select('B3').divide(10000);
var B4 = image.select('B4').divide(10000);
var B8 = image.select('B8').divide(10000);
var B11 = image.select('B11').divide(10000);
var B12 = image.select('B12').divide(10000);


// ============================================================
// INDICES
// ============================================================

var NDWI = B3.subtract(B8)
  .divide(
    B3.add(B8)
  )
  .rename('NDWI');

var MNDWI = B3.subtract(B11)
  .divide(
    B3.add(B11)
  )
  .rename('MNDWI');


// ============================================================
// WATER MASK
// SAME DEFINITION AS STEP 17
// ============================================================

var waterMask = MNDWI.gt(0)
  .rename('water_mask');


// ============================================================
// COMBINED IMAGE
// ============================================================

var combined = ee.Image.cat([
  B2.rename('B2'),
  B3.rename('B3'),
  B4.rename('B4'),
  B8.rename('B8'),
  B11.rename('B11'),
  B12.rename('B12'),
  NDWI,
  MNDWI,
  waterMask
]);


// ============================================================
// WATER-ONLY INDICES
// ============================================================

var MNDWI_water = MNDWI.updateMask(
  waterMask
);

var NDWI_water = NDWI.updateMask(
  waterMask
);


// ============================================================
// REDUCE PATCH
// ============================================================

var spectralStats = combined.reduceRegion({
  reducer: ee.Reducer.mean()
    .combine({
      reducer2: ee.Reducer.median(),
      sharedInputs: true
    })
    .combine({
      reducer2: ee.Reducer.count(),
      sharedInputs: true
    }),
  geometry: patch,
  scale: 10,
  maxPixels: 100000,
  bestEffort: false
});


// ============================================================
// WATER-ONLY STATISTICS
// ============================================================

var waterStats = ee.Dictionary({

  MNDWI_mean:
    MNDWI_water.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: patch,
      scale: 10,
      maxPixels: 100000,
      bestEffort: false
    }).get('MNDWI'),

  MNDWI_median:
    MNDWI_water.reduceRegion({
      reducer: ee.Reducer.median(),
      geometry: patch,
      scale: 10,
      maxPixels: 100000,
      bestEffort: false
    }).get('MNDWI'),

  NDWI_mean:
    NDWI_water.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: patch,
      scale: 10,
      maxPixels: 100000,
      bestEffort: false
    }).get('NDWI'),

  NDWI_median:
    NDWI_water.reduceRegion({
      reducer: ee.Reducer.median(),
      geometry: patch,
      scale: 10,
      maxPixels: 100000,
      bestEffort: false
    }).get('NDWI'),

  water_fraction:
    waterMask.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: patch,
      scale: 10,
      maxPixels: 100000,
      bestEffort: false
    }).get('water_mask')
});


// ============================================================
// PRINT RESULTS
// ============================================================

print('');
print('================================================');
print('REPAIRED VRINDAVAN COMPACT-8 WATER FEATURES');
print('================================================');

print('Station:', stationName);
print('Latitude:', lat);
print('Longitude:', lon);
print('Sentinel-2:', sentinelId);

print('');
print('Water statistics:');
print(waterStats);

print('');
print('All spectral statistics:');
print(spectralStats);


// ============================================================
// MAP
// ============================================================

Map.centerObject(
  point,
  15
);

Map.addLayer(
  MNDWI,
  {
    min: -1,
    max: 1,
    palette: [
      'brown',
      'white',
      'blue'
    ]
  },
  'MNDWI'
);

Map.addLayer(
  waterMask.selfMask(),
  {
    palette: ['blue']
  },
  'MNDWI > 0 water mask'
);

Map.addLayer(
  patch,
  {
    color: 'red'
  },
  '256 m patch'
);

Map.addLayer(
  point,
  {
    color: 'yellow'
  },
  'CWC station'
);