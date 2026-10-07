// ============================================================
// ETAWAH CWC → SENTINEL-2 MATCHED OBSERVATIONS
// TABLE EXPORT ONLY
// ============================================================

// ------------------------------------------------------------
// 1. ASSETS
// ------------------------------------------------------------

var cwcAsset =
  'projects/sentinel2-research-508107/assets/swq_physical_parameter_manual_cwc_up_2021_2025';

var cwc = ee.FeatureCollection(cwcAsset);


// ------------------------------------------------------------
// 2. ETAWAH CWC OBSERVATIONS
// ------------------------------------------------------------

var etawah = cwc
  .filter(ee.Filter.eq('Station', 'ETAWAH'))
  .filter(ee.Filter.notNull([
    'Latitude',
    'Longitude',
    'Data Acquisition Time'
  ]));

print('Total CWC observations:', cwc.size());
print('Etawah observations:', etawah.size());
print('First Etawah record:', etawah.first());


// ------------------------------------------------------------
// 3. SENTINEL-2 COLLECTION
// ------------------------------------------------------------

var s2 = ee.ImageCollection(
  'COPERNICUS/S2_SR_HARMONIZED'
)
.filterDate(
  '2020-12-25',
  '2026-01-01'
)
.filterBounds(
  ee.Geometry.Point([
    78.98333333,
    26.74972222
  ])
);


// ------------------------------------------------------------
// 4. CLOUD PROBABILITY COLLECTION
// ------------------------------------------------------------

var cloudProbability = ee.ImageCollection(
  'COPERNICUS/S2_CLOUD_PROBABILITY'
)
.filterDate(
  '2020-12-25',
  '2026-01-01'
)
.filterBounds(
  ee.Geometry.Point([
    78.98333333,
    26.74972222
  ])
);


// ------------------------------------------------------------
// 5. JOIN SENTINEL-2 + CLOUD PROBABILITY
// ------------------------------------------------------------

var joined = ee.Join.saveFirst(
  'cloud_probability'
).apply({

  primary: s2,

  secondary: cloudProbability,

  condition: ee.Filter.equals({
    leftField: 'system:index',
    rightField: 'system:index'
  })

});


// ------------------------------------------------------------
// 6. CREATE CLOUD-PROBABILITY COLLECTION
// ------------------------------------------------------------

var joinedS2 = ee.ImageCollection(joined);


// ------------------------------------------------------------
// 7. MATCH EACH CWC OBSERVATION TO CLOSEST IMAGE
// ------------------------------------------------------------

// Maximum allowed difference = 3 days.
//
// We deliberately do NOT calculate pixel statistics here.
// This script is only creating the matched observation table.

var MAX_DAYS = 3;


var matched = etawah.map(function(cwcFeature) {

  // ----------------------------------------------------------
  // CWC DATE
  // ----------------------------------------------------------

  var cwcDate = ee.Date.parse(
    'dd-MM-yyyy HH:mm',
    ee.String(
      cwcFeature.get('Data Acquisition Time')
    )
  );


  // ----------------------------------------------------------
  // CWC COORDINATES
  // ----------------------------------------------------------

  var latitude = ee.Number(
    cwcFeature.get('Latitude')
  );

  var longitude = ee.Number(
    cwcFeature.get('Longitude')
  );


  // ----------------------------------------------------------
  // FIND SENTINEL-2 IMAGES WITHIN ±3 DAYS
  // ----------------------------------------------------------

  var candidates = joinedS2
    .filterBounds(
      ee.Geometry.Point([
        longitude,
        latitude
      ])
    )
    .filterDate(
      cwcDate.advance(-MAX_DAYS, 'day'),
      cwcDate.advance(MAX_DAYS + 1, 'day')
    );


  // ----------------------------------------------------------
  // SORT BY ABSOLUTE DATE DIFFERENCE
  // ----------------------------------------------------------

  var withDifference = candidates.map(
    function(image) {

      var imageDate = ee.Date(
        image.get('system:time_start')
      );

      var differenceHours = ee.Number(
        imageDate.difference(
          cwcDate,
          'hour'
        )
      ).abs();

      return image.set(
        'match_difference_hours',
        differenceHours
      );
    }
  );


  var sorted = withDifference.sort(
    'match_difference_hours'
  );


  var bestImage = ee.Image(
    sorted.first()
  );


  // ----------------------------------------------------------
  // CHECK WHETHER MATCH EXISTS
  // ----------------------------------------------------------

  var hasMatch = sorted.size().gt(0);


  // ----------------------------------------------------------
  // SAFE IMAGE PROPERTIES
  // ----------------------------------------------------------

  var satelliteDate = ee.Algorithms.If(
    hasMatch,

    ee.Date(
      bestImage.get('system:time_start')
    ).format('yyyy-MM-dd'),

    null
  );


  var imageId = ee.Algorithms.If(
    hasMatch,

    bestImage.get('system:index'),

    null
  );


  var differenceDays = ee.Algorithms.If(
    hasMatch,

    ee.Number(
      bestImage.get(
        'match_difference_hours'
      )
    ).divide(24),

    null
  );


  var cloudPercentage = ee.Algorithms.If(
    hasMatch,

    bestImage.get(
      'CLOUDY_PIXEL_PERCENTAGE'
    ),

    null
  );


  // ----------------------------------------------------------
  // RETURN TABLE RECORD
  // ----------------------------------------------------------
  //
  // IMPORTANT:
  // We intentionally create ee.Feature(null, ...)
  // so no invalid ".geo" property is copied.
  //

  return ee.Feature(null, {

    station:
      cwcFeature.get('Station'),

    cwc_datetime:
      cwcFeature.get(
        'Data Acquisition Time'
      ),

    latitude:
      latitude,

    longitude:
      longitude,

    turbidity_NTU:
      cwcFeature.get(
        'Turbidity (NTU)'
      ),

    satellite_date:
      satelliteDate,

    sentinel2_id:
      imageId,

    days_difference:
      differenceDays,

    cloud_percentage:
      cloudPercentage

  });

});


// ------------------------------------------------------------
// 8. REMOVE RECORDS WITHOUT SENTINEL-2 MATCH
// ------------------------------------------------------------

var matchedOnly = matched.filter(
  ee.Filter.notNull([
    'sentinel2_id'
  ])
);


// ------------------------------------------------------------
// 9. PRINT RESULTS
// ------------------------------------------------------------

print(
  'Sentinel-2 images available:',
  s2.size()
);

print(
  'Matched CWC observations:',
  matchedOnly.size()
);

print(
  'First matched observation:',
  matchedOnly.first()
);

print(
  'Matched observation table:',
  matchedOnly.limit(20)
);


// ------------------------------------------------------------
// 10. PRINT MATCHED IMAGE IDS
// ------------------------------------------------------------

print(
  'Matched Sentinel-2 IDs:',
  matchedOnly.aggregate_array(
    'sentinel2_id'
  ).distinct()
);


// ------------------------------------------------------------
// 11. BASIC MATCH STATISTICS
// ------------------------------------------------------------

print(
  'Days difference statistics:',
  matchedOnly.reduceColumns({
    reducer: ee.Reducer.minMax()
      .combine({
        reducer2: ee.Reducer.mean(),
        sharedInputs: true
      }),

    selectors: [
      'days_difference'
    ]
  })
);


print(
  'Cloud percentage statistics:',
  matchedOnly.reduceColumns({
    reducer: ee.Reducer.minMax()
      .combine({
        reducer2: ee.Reducer.mean(),
        sharedInputs: true
      }),

    selectors: [
      'cloud_percentage'
    ]
  })
);


// ------------------------------------------------------------
// 12. EXPORT CSV
// ------------------------------------------------------------

Export.table.toDrive({

  collection: matchedOnly,

  description:
    'Etawah_CWC_Sentinel2_Matched_Observations',

  folder:
    'Etawah_Sentinel2',

  fileNamePrefix:
    'Etawah_CWC_Sentinel2_Matched_Observations',

  fileFormat:
    'CSV',

  selectors: [
    'station',
    'cwc_datetime',
    'latitude',
    'longitude',
    'turbidity_NTU',
    'satellite_date',
    'sentinel2_id',
    'days_difference',
    'cloud_percentage'
  ]

});


// ============================================================
// END
// ============================================================