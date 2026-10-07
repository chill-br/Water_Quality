// ============================================================
// STEP 10 — REBUILD ETAWAH CWC–SENTINEL TEMPORAL MATCHING
// ============================================================
//
// Purpose:
// Rebuild the ETAWAH CWC–Sentinel-2 temporal linkage directly
// from the original CWC asset and Sentinel-2 collection.
//
// IMPORTANT:
// We do NOT use the existing ml_ready_turbidity.csv linkage.
// Sentinel-2 acquisition time is taken directly from:
// system:time_start
//
// Matching window: ±2 days
// Station: ETAWAH
// Local River: Yamuna
//
// ============================================================


// ------------------------------------------------------------
// 1. INPUTS
// ------------------------------------------------------------

var CWC_ASSET =
  'projects/sentinel2-research-508107/assets/swq_physical_parameter_manual_cwc_up_2021_2025';

var S2_COLLECTION =
  'COPERNICUS/S2_SR_HARMONIZED';

var START_DATE = '2021-01-01';
var END_DATE   = '2025-01-01';

var MAX_DAYS = 2;


// ------------------------------------------------------------
// 2. CWC DATA
// ------------------------------------------------------------

var cwc = ee.FeatureCollection(CWC_ASSET)
  .filter(ee.Filter.eq('Local River', 'Yamuna'))
  .filter(ee.Filter.eq('Station', 'ETAWAH'));

print('ETAWAH CWC observations:', cwc.size());


// ------------------------------------------------------------
// 3. SENTINEL-2 COLLECTION
// ------------------------------------------------------------

var s2 = ee.ImageCollection(S2_COLLECTION)
  .filterDate(START_DATE, END_DATE)
  .filterBounds(
    ee.Geometry.Point([78.98333333, 26.74972222])
      .buffer(5000)
  );

print('Sentinel-2 images:', s2.size());


// ------------------------------------------------------------
// 4. CREATE CWC FEATURES WITH REAL TIMESTAMPS
// ------------------------------------------------------------
//
// CWC "Data Acquisition Time" is converted to milliseconds.
// We retain the original fields.
//
// ------------------------------------------------------------

var cwcPrepared = cwc.map(function(f) {

  var dtString = ee.String(
    f.get('Data Acquisition Time')
  );

  var dt = ee.Date.parse(
    'dd-MM-yyyy HH:mm',
    dtString
  );

  return f.set({
    'cwc_datetime_ms': dt.millis(),
    'cwc_datetime_iso': dt.format(
      'YYYY-MM-dd HH:mm:ss'
    )
  });
});


// ------------------------------------------------------------
// 5. MATCH EACH CWC OBSERVATION TO NEAREST S2 IMAGE
// ------------------------------------------------------------
//
// IMPORTANT:
// The match is performed using:
//
// CWC timestamp
//       |
//       v
// Sentinel system:time_start
//
// NOT satellite_date strings.
//
// ------------------------------------------------------------

var matched = cwcPrepared.map(function(cwcFeature) {

  var cwcTime = ee.Date(
    cwcFeature.get('cwc_datetime_ms')
  );

  var before = cwcTime.advance(
    -MAX_DAYS,
    'day'
  );

  var after = cwcTime.advance(
    MAX_DAYS,
    'day'
  );

  var candidates = s2
    .filterDate(before, after)
    .map(function(img) {

      var s2Time = ee.Date(
        img.get('system:time_start')
      );

      var diffHours = cwcTime
        .difference(s2Time, 'hour')
        .abs();

      var diffDays = diffHours
        .divide(24);

      return img.set({
        'temporal_difference_days': diffDays,
        'temporal_difference_hours': diffHours
      });
    })
    .sort('temporal_difference_hours');

  var nearest = ee.Image(
    candidates.first()
  );

  var count = candidates.size();

  return ee.Algorithms.If(
    count.gt(0),

    cwcFeature.set({
      'sentinel2_id': nearest.get(
        'system:index'
      ),

      'sentinel2_datetime': ee.Date(
        nearest.get('system:time_start')
      ).format(
        'YYYY-MM-dd HH:mm:ss'
      ),

      'sentinel2_date': ee.Date(
        nearest.get('system:time_start')
      ).format(
        'YYYY-MM-dd'
      ),

      'days_difference': nearest.get(
        'temporal_difference_days'
      ),

      'hours_difference': nearest.get(
        'temporal_difference_hours'
      ),

      'cloud_percentage': nearest.get(
        'CLOUDY_PIXEL_PERCENTAGE'
      ),

      'match_status': 'MATCHED'
    }),

    cwcFeature.set({
      'sentinel2_id': null,
      'sentinel2_datetime': null,
      'sentinel2_date': null,
      'days_difference': null,
      'hours_difference': null,
      'cloud_percentage': null,
      'match_status': 'NO_MATCH'
    })
  );
});


// ------------------------------------------------------------
// 6. REMOVE UNMATCHED RECORDS
// ------------------------------------------------------------

var matchedOnly = ee.FeatureCollection(
  matched
).filter(
  ee.Filter.eq(
    'match_status',
    'MATCHED'
  )
);

print(
  'Matched CWC observations:',
  matchedOnly.size()
);


// ------------------------------------------------------------
// 7. TEMPORAL QUALITY SUMMARY
// ------------------------------------------------------------

var within1day = matchedOnly.filter(
  ee.Filter.lte(
    'days_difference',
    1
  )
);

var within1_5day = matchedOnly.filter(
  ee.Filter.lte(
    'days_difference',
    1.5
  )
);

var within2day = matchedOnly.filter(
  ee.Filter.lte(
    'days_difference',
    2
  )
);

print(
  'Matches <= 1 day:',
  within1day.size()
);

print(
  'Matches <= 1.5 days:',
  within1_5day.size()
);

print(
  'Matches <= 2 days:',
  within2day.size()
);


// ------------------------------------------------------------
// 8. PRINT SAMPLE RECORDS
// ------------------------------------------------------------

print(
  'Sample matched records:',
  matchedOnly
    .sort('cwc_datetime_ms')
    .limit(30)
);


// ------------------------------------------------------------
// 9. EXPORT MATCHED TABLE
// ------------------------------------------------------------

Export.table.toDrive({
  collection: matchedOnly,
  description:
    'ETAWAH_CWC_Sentinel2_CORRECT_TEMPORAL_MATCH_2021_2024',

  folder:
    'Yamuna_Independent_Test',

  fileNamePrefix:
    'ETAWAH_CWC_Sentinel2_CORRECT_TEMPORAL_MATCH_2021_2024',

  fileFormat:
    'CSV'
});


// ------------------------------------------------------------
// 10. DATE CONSISTENCY CHECK
// ------------------------------------------------------------
//
// This should now show:
//
// CWC date ≈ Sentinel-2 date
//
// for every matched record.
//
// ------------------------------------------------------------

var dateCheck = matchedOnly
  .limit(20)
  .map(function(f) {

    return ee.Feature(null, {
      cwc_datetime:
        f.get('cwc_datetime_iso'),

      sentinel2_datetime:
        f.get('sentinel2_datetime'),

      sentinel2_id:
        f.get('sentinel2_id'),

      days_difference:
        f.get('days_difference'),

      hours_difference:
        f.get('hours_difference'),

      cloud_percentage:
        f.get('cloud_percentage'),

      turbidity_NTU:
        f.get(
          'Turbidity (NTU)'
        )
    });
  });

print(
  'DATE CONSISTENCY CHECK:',
  dateCheck
);


// ============================================================
// END
// ============================================================