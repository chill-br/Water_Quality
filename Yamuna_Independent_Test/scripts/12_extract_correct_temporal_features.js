// ============================================================
// STEP 12 — CORRECT TEMPORAL COMPACT-8 FEATURE EXTRACTION
// ETAWAH, 2021–2024
// ============================================================

var CWC = ee.FeatureCollection(
  'projects/sentinel2-research-508107/assets/swq_physical_parameter_manual_cwc_up_2021_2025'
);

var S2 = ee.ImageCollection(
  'COPERNICUS/S2_SR_HARMONIZED'
);


// ------------------------------------------------------------
// STEP 1 — GET ALL ETAWAH YAMUNA CWC RECORDS
// ------------------------------------------------------------

var etawah = CWC
  .filter(ee.Filter.eq('Local River', 'Yamuna'))
  .filter(ee.Filter.eq('Station', 'ETAWAH'));

print('ETAWAH CWC:', etawah.size());


// ------------------------------------------------------------
// STEP 2 — SENTINEL-2 COLLECTION
// ------------------------------------------------------------

var etawahPoint = ee.Geometry.Point([
  78.98333333,
  26.74972222
]);

var s2 = S2
  .filterDate('2021-01-01', '2025-01-01')
  .filterBounds(etawahPoint.buffer(5000));

print('Sentinel-2 images:', s2.size());


// ------------------------------------------------------------
// STEP 3 — CREATE FEATURES
// ------------------------------------------------------------

var makeFeatures = function(cwcFeature) {

  // Parse CWC datetime
  var cwcDate = ee.Date.parse(
    'dd-MM-yyyy HH:mm',
    ee.String(
      cwcFeature.get('Data Acquisition Time')
    )
  );

  var turbidity =
    cwcFeature.get('Turbidity (NTU)');


  // ----------------------------------------------------------
  // FIND NEAREST SENTINEL-2 IMAGE
  // ----------------------------------------------------------

  var candidates = s2
    .filterDate(
      cwcDate.advance(-2, 'day'),
      cwcDate.advance(2, 'day')
    )
    .map(function(img) {

      var difference = img.date()
        .difference(cwcDate, 'day')
        .abs();

      return img.set(
        'date_difference',
        difference
      );
    })
    .sort('date_difference');


  var candidateCount = candidates.size();


  // ----------------------------------------------------------
  // NO MATCH
  // ----------------------------------------------------------

  return ee.Algorithms.If(
    candidateCount.eq(0),

    ee.Feature(null),

    (function() {

      var img = ee.Image(
        candidates.first()
      );

      var satelliteDate =
        img.date();

      var daysDifference =
        satelliteDate
          .difference(
            cwcDate,
            'day'
          )
          .abs();

      var cloud =
        img.get(
          'CLOUDY_PIXEL_PERCENTAGE'
        );


      // ------------------------------------------------------
      // INDICES
      // ------------------------------------------------------

      var ndwi =
        img.normalizedDifference([
          'B3',
          'B8'
        ]).rename('NDWI');

      var mndwi =
        img.normalizedDifference([
          'B3',
          'B11'
        ]).rename('MNDWI');

      var waterMask =
        mndwi
          .gt(0)
          .rename('water_mask');


      // ------------------------------------------------------
      // 256 m PATCH
      // ------------------------------------------------------

      var patch =
        etawahPoint.buffer(128);


      // ------------------------------------------------------
      // MEANS
      // ------------------------------------------------------

      var meanImage =
        img.select([
          'B8',
          'B11',
          'B12'
        ])
        .addBands([
          ndwi,
          mndwi
        ]);

      var meanValues =
        meanImage.reduceRegion({
          reducer:
            ee.Reducer.mean(),

          geometry:
            patch,

          scale: 20,

          maxPixels: 50000,

          bestEffort: true
        });


      // ------------------------------------------------------
      // MEDIANS
      // ------------------------------------------------------

      var medianImage =
        img.select([
          'B8',
          'B11',
          'B12'
        ])
        .addBands([
          ndwi,
          mndwi
        ]);

      var medianValues =
        medianImage.reduceRegion({
          reducer:
            ee.Reducer.median(),

          geometry:
            patch,

          scale: 20,

          maxPixels: 50000,

          bestEffort: true
        });


      // ------------------------------------------------------
      // WATER FRACTION
      // ------------------------------------------------------

      var waterValues =
        waterMask.reduceRegion({
          reducer:
            ee.Reducer.mean(),

          geometry:
            patch,

          scale: 20,

          maxPixels: 50000,

          bestEffort: true
        });


      // ------------------------------------------------------
      // RAW VALUES
      // ------------------------------------------------------

      var b8 =
        meanValues.get('B8');

      var b11 =
        meanValues.get('B11');

      var b12 =
        meanValues.get('B12');

      var ndwiMean =
        meanValues.get('NDWI');

      var mndwiMean =
        meanValues.get('MNDWI');

      var ndwiMedian =
        medianValues.get('NDWI');

      var mndwiMedian =
        medianValues.get('MNDWI');

      var waterFraction =
        waterValues.get(
          'water_mask'
        );


      // ------------------------------------------------------
      // NULL-SAFE REFLECTANCE SCALING
      // ------------------------------------------------------

      var b8Scaled =
        ee.Algorithms.If(
          ee.Algorithms.IsEqual(
            b8,
            null
          ),
          null,
          ee.Number(b8)
            .divide(10000)
        );

      var b11Scaled =
        ee.Algorithms.If(
          ee.Algorithms.IsEqual(
            b11,
            null
          ),
          null,
          ee.Number(b11)
            .divide(10000)
        );

      var b12Scaled =
        ee.Algorithms.If(
          ee.Algorithms.IsEqual(
            b12,
            null
          ),
          null,
          ee.Number(b12)
            .divide(10000)
        );


      // ------------------------------------------------------
      // OUTPUT
      // ------------------------------------------------------

      return ee.Feature(
        etawahPoint,
        {

          station:
            'ETAWAH',

          latitude:
            cwcFeature.get(
              'Latitude'
            ),

          longitude:
            cwcFeature.get(
              'Longitude'
            ),

          cwc_datetime:
            cwcFeature.get(
              'Data Acquisition Time'
            ),

          turbidity_NTU:
            turbidity,

          satellite_id:
            img.get(
              'system:index'
            ),

          satellite_date:
            satelliteDate.format(
              'YYYY-MM-dd'
            ),

          satellite_datetime:
            satelliteDate.format(
              'YYYY-MM-dd HH:mm:ss'
            ),

          days_difference:
            daysDifference,

          cloud_percentage:
            cloud,

          // Compact-8
          MNDWI_median:
            mndwiMedian,

          NDWI_median:
            ndwiMedian,

          MNDWI_mean:
            mndwiMean,

          NDWI_mean:
            ndwiMean,

          water_fraction:
            waterFraction,

          B11_mean:
            b11Scaled,

          B12_mean:
            b12Scaled,

          B8_mean:
            b8Scaled
        }
      );

    })()
  );
};


// ------------------------------------------------------------
// STEP 4 — MAP
// ------------------------------------------------------------

var featureCollection =
  etawah.map(
    makeFeatures
  );


// ------------------------------------------------------------
// STEP 5 — REMOVE INCOMPLETE ROWS
// ------------------------------------------------------------

var complete =
  featureCollection;

var requiredFields = [

  'MNDWI_median',
  'NDWI_median',

  'MNDWI_mean',
  'NDWI_mean',

  'water_fraction',

  'B11_mean',
  'B12_mean',
  'B8_mean',

  'turbidity_NTU',

  'satellite_id',
  'satellite_date'
];

requiredFields.forEach(
  function(field) {

    complete =
      complete.filter(
        ee.Filter.notNull([
          field
        ])
      );

  }
);


// ------------------------------------------------------------
// STEP 6 — KEEP ONLY 2021–2024 MATCHES
// ------------------------------------------------------------

complete =
  complete.filter(
    ee.Filter.gte(
      'satellite_date',
      '2021-01-01'
    )
  );

complete =
  complete.filter(
    ee.Filter.lt(
      'satellite_date',
      '2025-01-01'
    )
  );


// ------------------------------------------------------------
// STEP 7 — PRINT RESULTS
// ------------------------------------------------------------

print(
  'Correct temporal feature rows:',
  complete.size()
);

print(
  'Correct temporal feature sample:',
  complete.limit(10)
);


// ------------------------------------------------------------
// STEP 8 — EXPORT
// ------------------------------------------------------------

Export.table.toDrive({

  collection:
    complete,

  description:
    'ETAWAH_CORRECT_TEMPORAL_COMPACT8_2021_2024',

  folder:
    'Yamuna_Independent_Test',

  fileNamePrefix:
    'ETAWAH_CORRECT_TEMPORAL_COMPACT8_2021_2024',

  fileFormat:
    'CSV'

});

print(
  'STEP 12 EXPORT READY'
);