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

- **Duplicate timestamps (checked in Stage 2):** 4 site files repeat timestamps, and the copies usually disagree (disagreeing timestamps / duplicated timestamps: `SQ Apartment37-48 Flexible PV` 432/432, `SQ14` 383/586, `Zone L1 (Station 1)` 5937/10087, `Zone L1(Station 2)` 1548/1566). Up to 4 copies per timestamp. The cause is not stated in the files (*unverified*); in `Zone L1 (Station 1)` the copies are close (correlation 0.98), in the others they are not. Rule in `src/preprocess.py`: identical copies are merged, disagreeing copies are set to missing.
  - In `SQ Apartment37-48 Flexible PV` every timestamp from 2023-11-20 07:00 to 2023-12-31 17:00 is duplicated (two different rows per hour, e.g. 780 W and 0 W). All 116 values above 1.2 x rated found in Stage 1 are on these timestamps, so they are a duplicate problem, not a capacity change. The duplicate rule sets them to missing; the station loses its daytime data for this period (test year).

- **Irradiance sensor scale (found in Stage 4, 2026-11-16):** the campus irradiance reading changes scale on about 2023-02-15 (clear-day peaks go from about 0.85 x to about 1.38 x clear-sky irradiance, factor about 1.6, while PV output does not change), and it reads about 0 from late March to mid May 2021 (April 2021 maximum 17 W/m2). The cause is not stated in the files (*unverified*: sensor replacement or recalibration is a guess). Handling: `src/preprocess.py::irradiance_level` and `calibrate_irradiance` (trailing 30-day clear-day level, earlier days only; missing while the sensor is dead). Raw values stay in the `irradiance_raw` column.

### Units and timezone (verified in `notebooks/01_eda.ipynb`, section 3)
- **Power:** `power(W)`; `generation(kWh)` is the energy of the step. Hourly stations: kWh x 1000 = W exactly (where not missing). 15-min stations: summed kWh is 2.8-3.3 % above W x 0.25 h / 1000, so `power(W)` is only approximately the step mean.
- **Irradiance:** W/m2, never negative; 739 of 1,567,792 values exceed 1500 W/m2 (max 3608.6); small night offset (median ~5 W/m2 between 00:00 and 04:59).
- **Timezone:** timestamps have no timezone; the irradiance peak matches local Hong Kong time (UTC+8, no DST). - **Interval label (inferred from correlation with 1-min irradiance, `notebooks/01_eda.ipynb` section 4; not stated in the dataset files):** 15-min optimizer stations are labelled at the interval **start** (`t` = `[t, t+15 min)`, best offset ~15 min, stable across 2021-2023); hourly stations are labelled at the interval **end** (`t` = `[t-60 min, t)`). Accuracy of the offset is about +-3 min. A 15-min row labelled `t` is therefore only known at `t + 15 min`.

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
