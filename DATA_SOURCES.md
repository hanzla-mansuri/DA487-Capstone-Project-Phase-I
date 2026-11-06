# Data Sources

Data files are **not tracked in git** (see `.gitignore`). Neither dataset ships a README or licence file; items marked *unverified* must be checked on the source page.

## 1. PRIMARY: HKUST rooftop PV dataset (Dryad)
- **Location:** `data/hkust/` (~1 GB)
- **Source:** https://doi.org/10.5061/dryad.m37pvmd99
- **Paper / citation:** Lin et al., *Scientific Data* 2025, doi:[10.1038/s41597-025-04397-y](https://doi.org/10.1038/s41597-025-04397-y)
- **Licence:** not included in the files; check the Dryad page (*unverified*).
- **Contents (verified locally by sampling):**
  - 60 site-level PV stations: 37 with panel-level optimizers (15-min) and 23 without (hourly).
  - 44 inverter-level files (5-min) for the optimizer stations.
  - One campus-wide meteorological series at 1-min: irradiance, temperature, relative humidity, pressure, visibility, wind speed/direction (2021-2023), rainfall (2022-2023 only). There is no per-station weather.
  - Metadata: `Metadata/PV generation system metadata.ttl` (Brick schema).
- **Period:** 2021-01-01 to 2023-12-31 (stations start between 2021-03 and 2023-08).
- **Known quirks to handle later:** at least one non-optimizer station has implausible power values (max ~31 MW); irradiance reaches 3608 W/m2; a few negative power values in non-optimizer files.

## 2. SECONDARY: HK-Rooftop-Microclimate-PV (Kaggle)
- **Location:** `data/raw/`
- **Source:** https://www.kaggle.com/datasets/ipcheng1845620444/hk-rooftop-microclimate-pv
- **Origin:** HKUST dataset, per the project owner (not stated in the files).
- **Citation:** *unverified*; use the Kaggle page. Authors are not stated in the files.
- **Licence:** **TO BE FILLED BY USER**
- **Contents (verified locally):** 31 panel-optimizer power columns at 5-min (2024-02-01 to 2025-10-13), hourly weather from one station, ~5-min microclimate boards. Role: optional secondary check only.

## Relationship between the two
- No temporal overlap: Dryad ends 2023-12-31, Kaggle starts 2024-01.
- Same hardware family and possibly overlapping campus sites is plausible but **unverified**: there is no station mapping between them.
- Units differ (Kaggle: per-optimizer W; Dryad: site/inverter totals), so Kaggle suits only a rough out-of-time sanity check, not a like-for-like test.
