# Data

## `usgs_01646500_annual_peaks.csv`

Annual peak streamflow for USGS gauge **01646500, Potomac River near
Washington DC (Little Falls Pump Station)** — 95 water years, 1931 to 2025.

| Column | Meaning |
| --- | --- |
| `water_year` | Water year of the peak |
| `peak_date` | Date of the annual maximum |
| `peak_discharge_cfs` | Peak discharge as published, cubic feet per second |
| `peak_discharge_m3s` | The same, converted at 0.028316846592 |
| `gage_height_ft` | Stage at the peak where recorded, feet (blank in early years) |

Retrieved from the National Water Information System peak-streamflow service on
2026-08-19:

```
https://nwis.waterdata.usgs.gov/nwis/peak?site_no=01646500&agency_cd=USGS&format=rdb
```

Refresh it, or fetch a different gauge, with:

```bash
python scripts/fetch_usgs_peaks.py --site 01646500
```

Years without a published discharge are dropped rather than treated as zero —
a blank is an unmeasured year, not a year without a flood.

### Licence

USGS data are works of the United States government and are in the public
domain. No attribution is required, though the agency asks to be credited.

### Used by

[`notebooks/05_flood_risk_real_data.ipynb`](../notebooks/05_flood_risk_real_data.ipynb),
which fits a flood-frequency distribution to the record and estimates the
probability of a levee being overtopped.
