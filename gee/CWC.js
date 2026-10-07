// ============================================================
// STEP 2B — CWC / SENTINEL-2 MATCHING
// ONE STATION AT A TIME
// ============================================================

var STATION_NAME = 'AURAIYA';

var CWC = ee.FeatureCollection(
  'projects/sentinel2-research-508107/assets/swq_physical_parameter_manual_cwc_up_2021_2025'
);

var station = CWC
  .filter(ee.Filter.eq('Local River', 'Yamuna'))
  .filter(ee.Filter.eq('Station', STATION_NAME));

print('==============================================');
print('STATION:', STATION_NAME);
print('CWC observations:', station.size());
print('==============================================');


// ------------------------------------------------------------
// Sentinel-2
// ------------------------------------------------------------

var S2 = ee.ImageCollection(
  'COPERNICUS/S2_SR_HARMONIZED'
);


// ------------------------------------------------------------
// Representative station location
// ------------------------------------------------------------

var lat = ee.Number(
  station.aggregate_mean('Latitude')
);

var lon = ee.Number(
  station.aggregate_mean('Longitude')
);

var point = ee.Geometry.Point([lon, lat]);


// ------------------------------------------------------------
// Sentinel-2 images around the station
// ------------------------------------------------------------

var s2 = S2
  .filterBounds(point)
  .filterDate(
    '2021-01-01',
    '2026-01-01'
  );

print('Sentinel-2 images at station:', s2.size());


// ------------------------------------------------------------
// Match every CWC observation to closest S2 image
// within ±2 days
// ------------------------------------------------------------

var matches = station.map(function(cwc) {

  var cwcDate = ee.Date.parse(
    'dd-MM-yyyy HH:mm',
    cwc.getString('Data Acquisition Time')
  );

  var candidates = s2.filterDate(
    cwcDate.advance(-2, 'day'),
    cwcDate.advance(2, 'day')
  );

  var candidateCount = candidates.size();

  var closest = candidates
    .map(function(image) {

      var diff = image.date()
        .difference(cwcDate, 'day')
        .abs();

      return image.set(
        'days_difference',
        diff
      );

    })
    .sort('days_difference')
    .first();


  return cwc.set({

    candidate_count: candidateCount,

    sentinel_id: ee.Algorithms.If(
      candidateCount.gt(0),
      closest.get('system:index'),
      null
    ),

    satellite_date: ee.Algorithms.If(
      candidateCount.gt(0),
      ee.Date(
        closest.get('system:time_start')
      ).format('YYYY-MM-dd'),
      null
    ),

    days_difference: ee.Algorithms.If(
      candidateCount.gt(0),
      closest.get('days_difference'),
      null
    ),

    cloud_percentage: ee.Algorithms.If(
      candidateCount.gt(0),
      closest.get('CLOUDY_PIXEL_PERCENTAGE'),
      null
    )

  });

});


// ------------------------------------------------------------
// Keep successful matches
// ------------------------------------------------------------

var matched = matches.filter(
  ee.Filter.notNull([
    'sentinel_id'
  ])
);


print('==============================================');
print('MATCHING RESULTS');
print('==============================================');

print(
  'CWC observations:',
  station.size()
);

print(
  'Matched CWC observations:',
  matched.size()
);


// ------------------------------------------------------------
// Matching thresholds
// ------------------------------------------------------------

var match15 = matched.filter(
  ee.Filter.lte(
    'days_difference',
    1.5
  )
);

var match10 = matched.filter(
  ee.Filter.lte(
    'days_difference',
    1.0
  )
);

var cloud20 = matched.filter(
  ee.Filter.lte(
    'cloud_percentage',
    20
  )
);

print(
  'Matches <= 1.5 days:',
  match15.size()
);

print(
  'Matches <= 1.0 day:',
  match10.size()
);

print(
  'Matches with cloud <=20%:',
  cloud20.size()
);


// ------------------------------------------------------------
// Unique Sentinel acquisitions
// ------------------------------------------------------------

var uniqueIDs = matched
  .aggregate_array('sentinel_id')
  .distinct();

print(
  'Unique Sentinel-2 acquisitions:',
  uniqueIDs.size()
);


// ------------------------------------------------------------
// Matching statistics
// ------------------------------------------------------------

print(
  'Mean date difference:',
  matched.aggregate_mean(
    'days_difference'
  )
);

print(
  'Mean cloud percentage:',
  matched.aggregate_mean(
    'cloud_percentage'
  )
);


// ------------------------------------------------------------
// Show matched records
// ------------------------------------------------------------

print(
  'Matched CWC / Sentinel records:',
  matched.select([
    'Station',
    'Data Acquisition Time',
    'Latitude',
    'Longitude',
    'Turbidity (NTU)',
    'sentinel_id',
    'satellite_date',
    'days_difference',
    'cloud_percentage'
  ])
);


// ------------------------------------------------------------
// Map
// ------------------------------------------------------------

Map.centerObject(point, 11);

Map.addLayer(
  point,
  {
    color: 'red'
  },
  STATION_NAME
);


// ------------------------------------------------------------
// Export
// ------------------------------------------------------------

Export.table.toDrive({

  collection: matched,

  description:
    STATION_NAME.replace(/[^A-Za-z0-9]/g, '_') +
    '_CWC_Sentinel2_Matches',

  folder:
    'Etawah_Water_Quality',

  fileNamePrefix:
    STATION_NAME.replace(/[^A-Za-z0-9]/g, '_') +
    '_CWC_Sentinel2_Matches',

  fileFormat:
    'CSV'

});

print('==============================================');
print('STEP 2B COMPLETE');
print('==============================================');