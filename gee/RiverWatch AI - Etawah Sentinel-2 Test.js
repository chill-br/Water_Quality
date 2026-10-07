// ============================================================
// RiverWatch AI - Etawah Sentinel-2 Test
// Version 1: Extract Sentinel-2 spectral features
// ============================================================


// ------------------------------------------------------------
// 1. ETawah station
// ------------------------------------------------------------
// We will replace these coordinates with the exact CWC
// coordinates from our dataset if necessary.

var etawah = ee.Geometry.Point([
  78.983333,  // longitude
  26.749722   // latitude
]);

Map.centerObject(etawah, 12);
Map.addLayer(etawah, {color: 'red'}, 'Etawah station');


// ------------------------------------------------------------
// 2. Date range
// ------------------------------------------------------------
// Start with a small test period.
// We will later replace this with the CWC observation dates.

var startDate = '2023-01-01';
var endDate   = '2023-03-31';


// ------------------------------------------------------------
// 3. Sentinel-2 Surface Reflectance
// ------------------------------------------------------------

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(etawah)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30));

print('Sentinel-2 images found:', s2.size());


// ------------------------------------------------------------
// 4. Cloud masking
// ------------------------------------------------------------

function maskClouds(image) {

  var scl = image.select('SCL');

  // Keep:
  // 4 = vegetation
  // 5 = bare soil
  // 6 = water
  // 7 = unclassified

  var mask = scl.eq(4)
    .or(scl.eq(5))
    .or(scl.eq(6))
    .or(scl.eq(7));

  return image
    .updateMask(mask)
    .divide(10000)
    .copyProperties(image, ['system:time_start']);
}


var cleanS2 = s2.map(maskClouds);


// ------------------------------------------------------------
// 5. Select spectral bands
// ------------------------------------------------------------

function addIndices(image) {

  var ndwi = image.normalizedDifference([
    'B3',
    'B8'
  ]).rename('NDWI');

  var mndwi = image.normalizedDifference([
    'B3',
    'B11'
  ]).rename('MNDWI');

  return image.addBands([
    ndwi,
    mndwi
  ]);
}


var processed = cleanS2.map(addIndices);


// ------------------------------------------------------------
// 6. Extract values at Etawah
// ------------------------------------------------------------

var samples = processed.map(function(image) {

  var values = image.select([
    'B2',
    'B3',
    'B4',
    'B8',
    'B11',
    'B12',
    'NDWI',
    'MNDWI'
  ]).reduceRegion({

    reducer: ee.Reducer.mean(),

    geometry: etawah.buffer(100),

    scale: 10,

    maxPixels: 1e9

  });

  return ee.Feature(null, values)
    .set('date',
      image.date().format('YYYY-MM-dd'))
    .set('cloud_percentage',
      image.get('CLOUDY_PIXEL_PERCENTAGE'));

});


// ------------------------------------------------------------
// 7. Display results
// ------------------------------------------------------------

print('Extracted Sentinel-2 observations:', samples);


// ------------------------------------------------------------
// 8. Export CSV
// ------------------------------------------------------------

Export.table.toDrive({

  collection: samples,

  description: 'Etawah_Sentinel2_test',

  fileFormat: 'CSV'

});