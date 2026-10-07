// ============================================================
// STEP 12 DIAGNOSTIC — SENTINEL-2 VALID PIXELS
// ============================================================

var CWC = ee.FeatureCollection(
  'projects/sentinel2-research-508107/assets/swq_physical_parameter_manual_cwc_up_2021_2025'
);

var S2 = ee.ImageCollection(
  'COPERNICUS/S2_SR_HARMONIZED'
);


// ------------------------------------------------------------
// ETAWAH CWC
// ------------------------------------------------------------

var etawah = CWC
  .filter(ee.Filter.eq('Local River', 'Yamuna'))
  .filter(ee.Filter.eq('Station', 'ETAWAH'));

print('ETAWAH CWC:', etawah.size());


// ------------------------------------------------------------
// SENTINEL-2
// ------------------------------------------------------------

var etawahPoint = ee.Geometry.Point([
  78.98333333,
  26.74972222
]);

var s2 = S2
  .filterDate(
    '2021-01-01',
    '2025-01-01'
  )
  .filterBounds(
    etawahPoint.buffer(5000)
  );

print(
  'Sentinel-2 images:',
  s2.size()
);


// ------------------------------------------------------------
// MATCH + VALID PIXEL DIAGNOSTIC
// ------------------------------------------------------------

var diagnose = function(cwcFeature) {

  var cwcDate = ee.Date.parse(
    'dd-MM-yyyy HH:mm',
    ee.String(
      cwcFeature.get(
        'Data Acquisition Time'
      )
    )
  );


  // Find nearest Sentinel image
  var candidates = s2
    .filterDate(
      cwcDate.advance(-2, 'day'),
      cwcDate.advance(2, 'day')
    )
    .map(function(img) {

      var diff =
        img.date()
          .difference(
            cwcDate,
            'day'
          )
          .abs();

      return img.set(
        'date_difference',
        diff
      );

    })
    .sort(
      'date_difference'
    );


  var count =
    candidates.size();


  return ee.Algorithms.If(

    count.eq(0),

    ee.Feature(
      etawahPoint,
      {
        station: 'ETAWAH',

        cwc_datetime:
          cwcFeature.get(
            'Data Acquisition Time'
          ),

        turbidity_NTU:
          cwcFeature.get(
            'Turbidity (NTU)'
          ),

        match_status:
          'NO_SENTINEL_MATCH'
      }
    ),

    (function() {

      var img =
        ee.Image(
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


      // ------------------------------------------------------
      // PATCH
      // ------------------------------------------------------

      var patch =
        etawahPoint.buffer(128);


      // ------------------------------------------------------
      // VALID PIXEL COUNT
      // ------------------------------------------------------

      var validImage =
        img.select([
          'B2',
          'B3',
          'B4',
          'B8',
          'B11',
          'B12'
        ]);


      var validCount =
        validImage
          .select('B8')
          .reduceRegion({
            reducer:
              ee.Reducer.count(),

            geometry:
              patch,

            scale: 20,

            maxPixels:
              50000,

            bestEffort:
              true
          })
          .get('B8');


      // ------------------------------------------------------
      // B8 MEAN
      // ------------------------------------------------------

      var b8Mean =
        validImage
          .select('B8')
          .reduceRegion({
            reducer:
              ee.Reducer.mean(),

            geometry:
              patch,

            scale: 20,

            maxPixels:
              50000,

            bestEffort:
              true
          })
          .get('B8');


      // ------------------------------------------------------
      // CLOUD
      // ------------------------------------------------------

      var cloud =
        img.get(
          'CLOUDY_PIXEL_PERCENTAGE'
        );


      return ee.Feature(
        etawahPoint,
        {

          station:
            'ETAWAH',

          cwc_datetime:
            cwcFeature.get(
              'Data Acquisition Time'
            ),

          turbidity_NTU:
            cwcFeature.get(
              'Turbidity (NTU)'
            ),

          satellite_id:
            img.get(
              'system:index'
            ),

          satellite_date:
            satelliteDate.format(
              'YYYY-MM-dd'
            ),

          days_difference:
            daysDifference,

          cloud_percentage:
            cloud,

          valid_B8_pixels:
            validCount,

          B8_mean_raw:
            b8Mean,

          match_status:
            'MATCHED'

        }
      );

    })()

  );
};


// ------------------------------------------------------------
// RUN
// ------------------------------------------------------------

var diagnostic =
  etawah.map(
    diagnose
  );


// ------------------------------------------------------------
// MATCHED ONLY
// ------------------------------------------------------------

var matched =
  diagnostic.filter(
    ee.Filter.eq(
      'match_status',
      'MATCHED'
    )
  );

print(
  'Matched rows:',
  matched.size()
);


// ------------------------------------------------------------
// VALID PIXEL SUMMARY
// ------------------------------------------------------------

var withPixels =
  matched.filter(
    ee.Filter.gt(
      'valid_B8_pixels',
      0
    )
  );

var noPixels =
  matched.filter(
    ee.Filter.eq(
      'valid_B8_pixels',
      0
    )
  );

print(
  'Matched with valid B8 pixels:',
  withPixels.size()
);

print(
  'Matched with ZERO valid B8 pixels:',
  noPixels.size()
);


// ------------------------------------------------------------
// CLOUD SUMMARY
// ------------------------------------------------------------

print(
  'ZERO-PIXEL MATCHES:',
  noPixels
    .select([
      'cwc_datetime',
      'satellite_id',
      'satellite_date',
      'days_difference',
      'cloud_percentage',
      'valid_B8_pixels'
    ])
    .limit(100)
);


// ------------------------------------------------------------
// ALL MATCHED SUMMARY
// ------------------------------------------------------------

print(
  'ALL MATCHED DIAGNOSTIC:',
  matched
    .select([
      'cwc_datetime',
      'turbidity_NTU',
      'satellite_id',
      'satellite_date',
      'days_difference',
      'cloud_percentage',
      'valid_B8_pixels',
      'B8_mean_raw'
    ])
    .limit(20)
);

print(
  'STEP 12 PIXEL DIAGNOSTIC READY'
);

print(
  'ZERO-PIXEL DETAILS:',
  noPixels
    .aggregate_array('cwc_datetime')
);

print(
  'ZERO-PIXEL SENTINEL IDS:',
  noPixels
    .aggregate_array('satellite_id')
);

print(
  'ZERO-PIXEL SENTINEL DATES:',
  noPixels
    .aggregate_array('satellite_date')
);

print(
  'ZERO-PIXEL CLOUD PERCENTAGES:',
  noPixels
    .aggregate_array('cloud_percentage')
);

print(
  'ZERO-PIXEL CWC TURBIDITY:',
  noPixels
    .aggregate_array('turbidity_NTU')
);

print(
  'ZERO-PIXEL DAYS DIFFERENCE:',
  noPixels
    .aggregate_array('days_difference')
);