// ============================================================
// STEP 3A — YAMUNA INDEPENDENT SPATIAL TEST DATASET
// OPTIMIZED VERSION
// ============================================================


// ------------------------------------------------------------
// SETTINGS
// ------------------------------------------------------------

var START_DATE = '2021-01-01';
var END_DATE   = '2026-01-01';
var MATCH_DAYS = 2;


// ------------------------------------------------------------
// CWC
// ------------------------------------------------------------

var CWC = ee.FeatureCollection(
  'projects/sentinel2-research-508107/assets/swq_physical_parameter_manual_cwc_up_2021_2025'
);


// ------------------------------------------------------------
// Sentinel-2
// ------------------------------------------------------------

var S2 = ee.ImageCollection(
  'COPERNICUS/S2_SR_HARMONIZED'
);


// ------------------------------------------------------------
// Six independent test stations
// ------------------------------------------------------------

var stations = [
  'AURAIYA',
  'KALPI',
  'HAMIRPUR',
  'RAJAPUR',
  'Mathura (Gokul Barrage)',
  'AGRA (J.B.)'
];


// ============================================================
// PROCESS ONE STATION
// ============================================================

function processStation(stationName) {

  var station = CWC
    .filter(
      ee.Filter.eq(
        'Local River',
        'Yamuna'
      )
    )
    .filter(
      ee.Filter.eq(
        'Station',
        stationName
      )
    );


  // ----------------------------------------------------------
  // Representative station coordinates
  // ----------------------------------------------------------

  var lat = ee.Number(
    station.aggregate_mean(
      'Latitude'
    )
  );

  var lon = ee.Number(
    station.aggregate_mean(
      'Longitude'
    )
  );

  var point = ee.Geometry.Point([
    lon,
    lat
  ]);


  // ----------------------------------------------------------
  // Sentinel-2 collection
  // ----------------------------------------------------------

  var s2 = S2
    .filterBounds(point)
    .filterDate(
      START_DATE,
      END_DATE
    );


  // ----------------------------------------------------------
  // CWC → Sentinel-2 matching
  // ----------------------------------------------------------

  var matches = station.map(
    function(cwc) {

      var cwcDate = ee.Date.parse(
        'dd-MM-yyyy HH:mm',
        cwc.getString(
          'Data Acquisition Time'
        )
      );


      var candidates = s2.filterDate(
        cwcDate.advance(
          -MATCH_DAYS,
          'day'
        ),
        cwcDate.advance(
          MATCH_DAYS,
          'day'
        )
      );


      var candidateCount =
        candidates.size();


      var closest = candidates
        .map(
          function(image) {

            var diff =
              image.date()
              .difference(
                cwcDate,
                'day'
              )
              .abs();


            return image.set(
              'days_difference',
              diff
            );

          }
        )
        .sort(
          'days_difference'
        )
        .first();


      return cwc.set({

        sentinel_id:
          ee.Algorithms.If(
            candidateCount.gt(0),
            closest.get(
              'system:index'
            ),
            null
          ),

        satellite_date:
          ee.Algorithms.If(
            candidateCount.gt(0),
            ee.Date(
              closest.get(
                'system:time_start'
              )
            ).format(
              'YYYY-MM-dd'
            ),
            null
          ),

        days_difference:
          ee.Algorithms.If(
            candidateCount.gt(0),
            closest.get(
              'days_difference'
            ),
            null
          ),

        cloud_percentage:
          ee.Algorithms.If(
            candidateCount.gt(0),
            closest.get(
              'CLOUDY_PIXEL_PERCENTAGE'
            ),
            null
          )

      });

    }
  );


  var matched = matches.filter(
    ee.Filter.notNull([
      'sentinel_id'
    ])
  );


  // ----------------------------------------------------------
  // Extract Compact-8 features
  // ----------------------------------------------------------

  var features = matched.map(
    function(record) {


      var satelliteID =
        ee.String(
          record.get(
            'sentinel_id'
          )
        );


      var image = ee.Image(
        S2
          .filter(
            ee.Filter.eq(
              'system:index',
              satelliteID
            )
          )
          .first()
      );


      // ------------------------------------------------------
      // Bands
      // ------------------------------------------------------

      var B2 =
        image.select('B2');

      var B3 =
        image.select('B3');

      var B4 =
        image.select('B4');

      var B8 =
        image.select('B8');

      var B11 =
        image.select('B11');

      var B12 =
        image.select('B12');


      // ------------------------------------------------------
      // Indices
      // ------------------------------------------------------

      var NDWI =
        B3.subtract(B8)
          .divide(
            B3.add(B8)
          )
          .rename('NDWI');


      var MNDWI =
        B3.subtract(B11)
          .divide(
            B3.add(B11)
          )
          .rename('MNDWI');


      var waterMask =
        MNDWI.gt(0)
          .rename(
            'MNDWI_WATER'
          );


      // ------------------------------------------------------
      // Stack
      // ------------------------------------------------------

      var stack =
        ee.Image.cat([
          B2,
          B3,
          B4,
          B8,
          B11,
          B12,
          NDWI,
          MNDWI,
          waterMask
        ]);


      // ------------------------------------------------------
      // CWC location
      // ------------------------------------------------------

      var lat =
        ee.Number(
          record.get(
            'Latitude'
          )
        );

      var lon =
        ee.Number(
          record.get(
            'Longitude'
          )
        );


      var samplePoint =
        ee.Geometry.Point([
          lon,
          lat
        ]);


      // ------------------------------------------------------
      // 256 m patch
      // ------------------------------------------------------

      var patch =
        samplePoint.buffer(128);


      // ------------------------------------------------------
      // Mean + median
      // ------------------------------------------------------

      var stats =
        stack.reduceRegion({

          reducer:
            ee.Reducer.mean()
              .combine({
                reducer2:
                  ee.Reducer.median(),
                sharedInputs:
                  true
              }),

          geometry:
            patch,

          scale:
            20,

          maxPixels:
            50000,

          bestEffort:
            true

        });


      // ------------------------------------------------------
      // Water fraction
      // ------------------------------------------------------

      var waterFraction =
        waterMask.reduceRegion({

          reducer:
            ee.Reducer.mean(),

          geometry:
            patch,

          scale:
            20,

          maxPixels:
            50000,

          bestEffort:
            true

        }).get(
          'MNDWI_WATER'
        );


      // ------------------------------------------------------
      // Final Compact-8 record
      // ------------------------------------------------------

      return ee.Feature(
        samplePoint,
        {

          station:
            stationName,

          latitude:
            lat,

          longitude:
            lon,

          cwc_datetime:
            record.get(
              'Data Acquisition Time'
            ),

          turbidity_NTU:
            record.get(
              'Turbidity (NTU)'
            ),

          satellite_id:
            satelliteID,

          satellite_date:
            record.get(
              'satellite_date'
            ),

          days_difference:
            record.get(
              'days_difference'
            ),

          cloud_percentage:
            record.get(
              'cloud_percentage'
            ),

          // Compact-8
          MNDWI_median:
            stats.get(
              'MNDWI_median'
            ),

          NDWI_median:
            stats.get(
              'NDWI_median'
            ),

          MNDWI_mean:
            stats.get(
              'MNDWI_mean'
            ),

          NDWI_mean:
            stats.get(
              'NDWI_mean'
            ),

          water_fraction:
            waterFraction,

          B11_mean:
            stats.get(
              'B11_mean'
            ),

          B12_mean:
            stats.get(
              'B12_mean'
            ),

          B8_mean:
            stats.get(
              'B8_mean'
            )

        }
      );

    }
  );


  return features;
}


// ============================================================
// PROCESS SIX STATIONS
// ============================================================

var allTestFeatures =
  ee.FeatureCollection([]);


stations.forEach(
  function(stationName) {

    print(
      'Processing station:',
      stationName
    );

    var result =
      processStation(
        stationName
      );

    allTestFeatures =
      allTestFeatures.merge(
        result
      );

  }
);


// ============================================================
// EXPECTED RECORD COUNT
// ============================================================

print(
  'Expected total records:',
  666
);


// ============================================================
// DATASET
// ============================================================

print(
  'Independent test FeatureCollection:',
  allTestFeatures
);


// ============================================================
// EXPORT
// ============================================================

Export.table.toDrive({

  collection:
    allTestFeatures,

  description:
    'Yamuna_Independent_Spatial_Test_Features',

  folder:
    'Etawah_Water_Quality',

  fileNamePrefix:
    'Yamuna_Independent_Spatial_Test_Features',

  fileFormat:
    'CSV'

});


print(
  '=============================================='
);

print(
  'STEP 3A EXPORT READY'
);

print(
  'Expected records: 666'
);

print(
  '=============================================='
);