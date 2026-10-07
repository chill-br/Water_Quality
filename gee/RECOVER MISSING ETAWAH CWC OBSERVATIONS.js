// ============================================================
// RECOVER MISSING ETAWAH CWC OBSERVATIONS
// ============================================================

var CWC_ASSET =
  'projects/sentinel2-research-508107/assets/swq_physical_parameter_manual_cwc_up_2021_2025';

var cwc = ee.FeatureCollection(CWC_ASSET);

// Missing Sentinel acquisition dates that need investigation
var targetDates = [
  '2021-07-25',
  '2021-08-14',
  '2022-07-05',
  '2023-08-19',
  '2024-03-11',
  '2024-06-24'
];

// ------------------------------------------------------------
// Convert CWC date string: DD-MM-YYYY HH:MM -> YYYY-MM-DD
// ------------------------------------------------------------
function parseCWCDate(dateString) {
  dateString = ee.String(dateString);

  var datePart = dateString.slice(0, 10);

  var day = datePart.slice(0, 2);
  var month = datePart.slice(3, 5);
  var year = datePart.slice(6, 10);

  return ee.String(year)
    .cat('-')
    .cat(month)
    .cat('-')
    .cat(day);
}

// ------------------------------------------------------------
// Keep only ETAWAH observations
// ------------------------------------------------------------
var etawah = cwc.filter(
  ee.Filter.eq('station', 'ETAWAH')
);

// ------------------------------------------------------------
// Add a clean YYYY-MM-DD date property
// ------------------------------------------------------------
var etawahWithDate = etawah.map(function(f) {

  var rawDate = f.get('cwc_datetime');

  return f.set(
    'cwc_date_clean',
    parseCWCDate(rawDate)
  );
});

// ------------------------------------------------------------
// Filter to the six missing dates
// ------------------------------------------------------------
var missing = etawahWithDate.filter(
  ee.Filter.inList('cwc_date_clean', targetDates)
);

// ------------------------------------------------------------
// Print compact information
// ------------------------------------------------------------
print('Missing-date CWC observations:', missing.size());

print(
  'Recovered CWC records:',
  missing.select([
    'station',
    'cwc_datetime',
    'latitude',
    'longitude',
    'turbidity_NTU',
    'cwc_date_clean'
  ])
);

// ------------------------------------------------------------
// Print each date separately
// ------------------------------------------------------------

targetDates.forEach(function(date) {

  var records = missing.filter(
    ee.Filter.eq('cwc_date_clean', date)
  );

  print(
    'CWC records for ' + date,
    records.select([
      'station',
      'cwc_datetime',
      'latitude',
      'longitude',
      'turbidity_NTU',
      'cwc_date_clean'
    ])
  );
});