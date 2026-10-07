// ============================================================
// STEP 16 — EXPANDED INDEPENDENT SPATIAL TEST
// CORRECTED MEMORY-SAFE VERSION
// ============================================================
//
// PURPOSE:
// Match CWC Yamuna observations from 8 additional stations
// to the nearest Sentinel-2 image.
//
// Stations:
//   Baghpat
//   Mawi
//   Kalanaur
//   Mathura (Gokul Barrage)
//   Vrindavan-Yamuna ExpresswayLink Road Bridge U/S of Mathura
//   Yamuna Highway Road Bridge
//   AGRA(POIYAGHAT)
//   Kailas Mandir, Agra
//
// IMPORTANT:
// - CWC geometry is reconstructed from Latitude/Longitude.
// - CWC datetime is explicitly parsed.
// - Matching is performed station-by-station.
// - Sentinel-2 is spatially filtered to each station.
// - Nearest image is selected using actual acquisition time.
// - No model is trained or modified.
// - Station summary is NOT calculated in GEE because it can cause
//   memory errors. Calculate it later from the exported CSV.
//
// ============================================================


// ------------------------------------------------------------
// 1. SETTINGS
// ------------------------------------------------------------

var CWC_ASSET =
    'projects/sentinel2-research-508107/assets/swq_physical_parameter_manual_cwc_up_2021_2025';

var S2_COLLECTION =
    'COPERNICUS/S2_SR_HARMONIZED';

var START_DATE = '2021-01-01';
var END_DATE   = '2025-01-01';

var MAX_DAYS = 2;


// ------------------------------------------------------------
// 2. STATIONS
// ------------------------------------------------------------

var TEST_STATIONS = [
  'Baghpat',
  'Mawi',
  'Kalanaur',
  'Mathura (Gokul Barrage)',
  'Vrindavan-Yamuna ExpresswayLink Road Bridge U/S of Mathura',
  'Yamuna Highway Road Bridge',
  'AGRA(POIYAGHAT)',
  'Kailas Mandir, Agra'
];

print(
  'Expanded independent spatial test stations:',
  TEST_STATIONS
);


// ------------------------------------------------------------
// 3. LOAD CWC
// ------------------------------------------------------------

var cwc = ee.FeatureCollection(
  CWC_ASSET
);


// ------------------------------------------------------------
// 4. FILTER YAMUNA + SELECT STATIONS
// ------------------------------------------------------------

var cwcYamuna = cwc
  .filter(
    ee.Filter.eq(
      'Local River',
      'Yamuna'
    )
  )
  .filter(
    ee.Filter.inList(
      'Station',
      TEST_STATIONS
    )
  );

print(
  'Expanded Yamuna CWC observations:',
  cwcYamuna.size()
);


// ------------------------------------------------------------
// 5. RECONSTRUCT GEOMETRY + DATETIME
// ------------------------------------------------------------

var cwcPrepared = cwcYamuna.map(
  function(feature) {

    // --------------------------------------------------------
    // Coordinates
    // --------------------------------------------------------

    var lat = ee.Number.parse(
      ee.String(
        feature.get('Latitude')
      )
    );

    var lon = ee.Number.parse(
      ee.String(
        feature.get('Longitude')
      )
    );

    var point = ee.Geometry.Point([
      lon,
      lat
    ]);


    // --------------------------------------------------------
    // Parse CWC datetime
    // Format:
    // DD-MM-YYYY HH:MM
    // --------------------------------------------------------

    var textDate = ee.String(
      feature.get(
        'Data Acquisition Time'
      )
    );

    var dateParts =
      textDate.split(' ');

    var datePart =
      ee.String(
        dateParts.get(0)
      );

    var timePart =
      ee.String(
        dateParts.get(1)
      );


    // DD-MM-YYYY

    var dmy =
      datePart.split('-');

    var day =
      ee.Number.parse(
        ee.String(
          dmy.get(0)
        )
      );

    var month =
      ee.Number.parse(
        ee.String(
          dmy.get(1)
        )
      );

    var year =
      ee.Number.parse(
        ee.String(
          dmy.get(2)
        )
      );


    // HH:MM

    var hm =
      timePart.split(':');

    var hour =
      ee.Number.parse(
        ee.String(
          hm.get(0)
        )
      );

    var minute =
      ee.Number.parse(
        ee.String(
          hm.get(1)
        )
      );


    // Construct exact datetime

    var datetime =
      ee.Date.fromYMD(
        year,
        month,
        day
      )
      .advance(
        hour,
        'hour'
      )
      .advance(
        minute,
        'minute'
      );


    return feature
      .setGeometry(point)
      .set({

        cwc_datetime_iso:
          datetime.format(
            "YYYY-MM-dd'T'HH:mm:ss"
          ),

        cwc_time_ms:
          datetime.millis(),

        cwc_date:
          datetime.format(
            'YYYY-MM-dd'
          )

      });

  }
);


// ------------------------------------------------------------
// 6. SENTINEL-2 BASE COLLECTION
// ------------------------------------------------------------
//
// Only temporal filtering is applied here.
//
// Spatial filtering is done separately for each station.
// This avoids building one huge multi-station geometry.
//
// ------------------------------------------------------------

var s2Base =
  ee.ImageCollection(
    S2_COLLECTION
  )
  .filterDate(
    START_DATE,
    END_DATE
  );

print(
  'Base Sentinel-2 collection loaded.'
);


// ------------------------------------------------------------
// 7. FUNCTION: MATCH ONE STATION
// ------------------------------------------------------------

function matchStation(stationName) {

  print(
    'Preparing station:',
    stationName
  );


  // ----------------------------------------------------------
  // CWC observations for this station
  // ----------------------------------------------------------

  var stationCWC =
    cwcPrepared.filter(
      ee.Filter.eq(
        'Station',
        stationName
      )
    );


  // ----------------------------------------------------------
  // Get station point
  // ----------------------------------------------------------

  var firstStation =
    ee.Feature(
      stationCWC.first()
    );

  var stationPoint =
    firstStation.geometry();


  // ----------------------------------------------------------
  // Sentinel-2 only around station
  // ----------------------------------------------------------

  var stationS2 =
    s2Base.filterBounds(
      stationPoint.buffer(1000)
    );


  // ----------------------------------------------------------
  // Match every CWC observation
  // ----------------------------------------------------------

  var matched =
    stationCWC.map(
      function(cwcFeature) {

        var cwcTime =
          ee.Date(
            cwcFeature.get(
              'cwc_time_ms'
            )
          );


        // ----------------------------------------------------
        // Sentinel candidates within ±2 days
        // ----------------------------------------------------

        var candidates =
          stationS2
          .filter(
            ee.Filter.maxDifference({

              difference:
                MAX_DAYS *
                24 *
                60 *
                60 *
                1000,

              leftField:
                'system:time_start',

              rightValue:
                cwcTime.millis()

            })
          );


        // ----------------------------------------------------
        // Calculate absolute temporal difference
        // ----------------------------------------------------

        candidates =
          candidates.map(
            function(image) {

              var imageTime =
                ee.Date(
                  image.get(
                    'system:time_start'
                  )
                );

              var diffHours =
                imageTime
                  .difference(
                    cwcTime,
                    'hour'
                  )
                  .abs();

              return image.set(
                'match_difference_hours',
                diffHours
              );

            }
          );


        // ----------------------------------------------------
        // Sort nearest first
        // ----------------------------------------------------

        candidates =
          candidates.sort(
            'match_difference_hours'
          );


        // ----------------------------------------------------
        // Check whether candidate exists
        // ----------------------------------------------------

        var count =
          candidates.size();

        var hasMatch =
          count.gt(0);


        // ----------------------------------------------------
        // Nearest image
        // ----------------------------------------------------

        var nearest =
          ee.Image(
            candidates.first()
          );


        // ----------------------------------------------------
        // Sentinel datetime
        // ----------------------------------------------------

        var sentinelTime =
          ee.Algorithms.If(
            hasMatch,

            nearest.get(
              'system:time_start'
            ),

            null
          );


        // ----------------------------------------------------
        // Sentinel ID
        // ----------------------------------------------------

        var sentinelId =
          ee.Algorithms.If(
            hasMatch,

            nearest.get(
              'system:index'
            ),

            null
          );


        // ----------------------------------------------------
        // Temporal difference
        // ----------------------------------------------------

        var daysDifference =
          ee.Algorithms.If(

            hasMatch,

            ee.Number(
              ee.Date(
                sentinelTime
              )
              .difference(
                cwcTime,
                'day'
              )
            ).abs(),

            null
          );


        // ----------------------------------------------------
        // Cloud percentage
        // ----------------------------------------------------

        var cloudPercentage =
          ee.Algorithms.If(

            hasMatch,

            nearest.get(
              'CLOUDY_PIXEL_PERCENTAGE'
            ),

            null
          );


        // ----------------------------------------------------
        // Return CWC feature + match metadata
        // ----------------------------------------------------

        return cwcFeature.set({

          match_status:
            ee.Algorithms.If(
              hasMatch,
              'MATCHED',
              'NO_MATCH'
            ),

          sentinel2_id:
            sentinelId,

          sentinel2_datetime:
            ee.Algorithms.If(

              hasMatch,

              ee.Date(
                sentinelTime
              ).format(
                "YYYY-MM-dd'T'HH:mm:ss"
              ),

              null
            ),

          sentinel2_date:
            ee.Algorithms.If(

              hasMatch,

              ee.Date(
                sentinelTime
              ).format(
                'YYYY-MM-dd'
              ),

              null
            ),

          days_difference:
            daysDifference,

          hours_difference:
            ee.Algorithms.If(

              hasMatch,

              ee.Number(
                daysDifference
              ).multiply(24),

              null
            ),

          cloud_percentage:
            cloudPercentage

        });

      }
    );


  return matched;

}


// ------------------------------------------------------------
// 8. RUN STATION-BY-STATION
// ------------------------------------------------------------

var stationResults = [];


// ------------------------------------------------------------
// Baghpat
// ------------------------------------------------------------

stationResults.push(
  matchStation(
    'Baghpat'
  )
);


// ------------------------------------------------------------
// Mawi
// ------------------------------------------------------------

stationResults.push(
  matchStation(
    'Mawi'
  )
);


// ------------------------------------------------------------
// Kalanaur
// ------------------------------------------------------------

stationResults.push(
  matchStation(
    'Kalanaur'
  )
);


// ------------------------------------------------------------
// Mathura
// ------------------------------------------------------------

stationResults.push(
  matchStation(
    'Mathura (Gokul Barrage)'
  )
);


// ------------------------------------------------------------
// Vrindavan
// ------------------------------------------------------------

stationResults.push(
  matchStation(
    'Vrindavan-Yamuna ExpresswayLink Road Bridge U/S of Mathura'
  )
);


// ------------------------------------------------------------
// Yamuna Highway
// ------------------------------------------------------------

stationResults.push(
  matchStation(
    'Yamuna Highway Road Bridge'
  )
);


// ------------------------------------------------------------
// Agra Poiyaghat
// ------------------------------------------------------------

stationResults.push(
  matchStation(
    'AGRA(POIYAGHAT)'
  )
);


// ------------------------------------------------------------
// Kailas Mandir
// ------------------------------------------------------------

stationResults.push(
  matchStation(
    'Kailas Mandir, Agra'
  )
);


// ------------------------------------------------------------
// 9. MERGE RESULTS
// ------------------------------------------------------------

var matchedAll =
  ee.FeatureCollection(
    stationResults
  ).flatten();


// ------------------------------------------------------------
// 10. KEEP ONLY MATCHED OBSERVATIONS
// ------------------------------------------------------------

var matchedOnly =
  matchedAll.filter(
    ee.Filter.eq(
      'match_status',
      'MATCHED'
    )
  );


print(
  'Expanded matched observations:',
  matchedOnly.size()
);


// ------------------------------------------------------------
// 11. BASIC QUALITY COUNTS
// ------------------------------------------------------------
//
// These are simple filters and are safe enough to retain.
//
// ------------------------------------------------------------

print(
  'Matches <= 1 day:',
  matchedOnly
    .filter(
      ee.Filter.lte(
        'days_difference',
        1
      )
    )
    .size()
);


print(
  'Matches <= 1.5 days:',
  matchedOnly
    .filter(
      ee.Filter.lte(
        'days_difference',
        1.5
      )
    )
    .size()
);


print(
  'Matches <= 2 days:',
  matchedOnly
    .filter(
      ee.Filter.lte(
        'days_difference',
        2
      )
    )
    .size()
);


print(
  'Cloud <= 20%:',
  matchedOnly
    .filter(
      ee.Filter.lte(
        'cloud_percentage',
        20
      )
    )
    .size()
);


print(
  'Cloud <= 40%:',
  matchedOnly
    .filter(
      ee.Filter.lte(
        'cloud_percentage',
        40
      )
    )
    .size()
);


// ------------------------------------------------------------
// 12. SAMPLE MATCHED OBSERVATIONS
// ------------------------------------------------------------
//
// Do NOT call aggregate operations over the full collection.
//
// ------------------------------------------------------------

print(
  'Sample matched observations:',
  matchedOnly.limit(10)
);


// ------------------------------------------------------------
// 13. MAP
// ------------------------------------------------------------

Map.centerObject(
  cwcPrepared,
  7
);


Map.addLayer(
  cwcPrepared,
  {
    color: 'red'
  },
  'Expanded Yamuna CWC stations'
);


// ------------------------------------------------------------
// 14. EXPORT MATCHED DATA
// ------------------------------------------------------------
//
// THIS IS THE IMPORTANT EXPORT.
// Use this CSV for all subsequent Python verification.
//
// ------------------------------------------------------------

Export.table.toDrive({

  collection:
    matchedOnly,

  description:
    'YAMUNA_EXPANDED_INDEPENDENT_SPATIAL_MATCH_2021_2024',

  folder:
    'Yamuna_Independent_Test',

  fileNamePrefix:
    'Yamuna_Expanded_Independent_Spatial_Match_2021_2024',

  fileFormat:
    'CSV'

});


// ------------------------------------------------------------
// 15. DO NOT EXPORT STATION SUMMARY FROM GEE
// ------------------------------------------------------------
//
// Station-level statistics will be calculated locally from
// the downloaded verified CSV.
//
// This avoids the previous:
//
//   User memory limit exceeded
//
// error.
//
// ------------------------------------------------------------


print(
  '================================================'
);

print(
  'STEP 16 CORRECTED MEMORY-SAFE VERSION COMPLETE'
);

print(
  'Start the Drive export:'
);

print(
  'YAMUNA_EXPANDED_INDEPENDENT_SPATIAL_MATCH_2021_2024'
);

print(
  'After downloading the CSV, calculate the station'
);

print(
  'summary and verification in Python.'
);

print(
  '================================================'
);