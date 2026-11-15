# Work Log

| Date | Hours | Work done |
|---|---|---|

## 2026-11-06 — hours: _TBD (fill in)_
- Set up repo scaffold, Python venv (pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm, shap, jupyter), `.gitignore`, git init.
- Built `notebooks/01_eda.ipynb` (Kaggle India dataset audit); tested on synthetic data only, never run on real data.
- Scope change: project moved to multi-horizon forecasting on the HKUST rooftop PV dataset (Lin et al., Sci. Data 2025).
- Inspected `data/raw/` (structure, timestamps, units, missing data) and reviewed git state. No models trained.
- Updated `.gitignore`, `README.md` (new title/scope/citation) and created this log.

## 2026-11-06 (later): Stage 0 git setup, hours: _TBD (fill in)_
- Inspected GitHub remote: 1 old commit (Kaggle-India EDA, unrelated history); no secrets or large files.
- Backups: tag `backup-before-capstone-cleanup`, branches `archive/old-scope`, `archive/github-old-main`.
- Merged remote `main` with `--allow-unrelated-histories` (no force push); archived old notebook/figures under `archive/`.
- Hardened `.gitignore`, added `.gitkeep`, `CLAUDE.md`, `DATA_SOURCES.md`, `PLAN.md`; updated README.
- Documented both datasets (primary HKUST Dryad in `data/hkust/`, secondary Kaggle in `data/raw/`).
- Pushed backup tag, both archive branches and `main` (fast-forward, no force). Stage 0 DONE.

## 2026-11-06 (planning), hours: _TBD (fill in)_
- Wrote the full `PLAN.md`: objectives and success criteria, Stages 0-11 (tasks, outputs, acceptance criteria, viva questions, hours, weeks), 12-week schedule with buffer, design decisions and risks, experiment protocol, Resume Protocol.
- Rewrote `CLAUDE.md` (project summary and session-start rule); updated `README.md` (objective, scope, reproduce steps).
- Checked the `.ttl` metadata field names (ratedPowerOutput, tiltAngle, azimuth, connectionDate) to plan capacity normalisation; values not yet parsed.
- Waiting for approval of `PLAN.md` before Stage 1 (EDA).

## 2026-11-06 (plan review), hours: _TBD (fill in)_
- User revised `PLAN.md` (140 h incl. 10 h buffer; §3 design decisions; leakage, fault-detection and generalisation fixes).
- Fixed two consistency issues (Kaggle cut-list order; pvlib/pyarrow install task) and committed the plan.
- Pushed `main` to GitHub (fast-forward to `cb3bd74`). Next: Stage 1 (EDA), awaiting go-ahead.

## 2026-11-07 (Stage 1: data loaders), hours: _TBD (fill in)_
- Wrote `src/data_io.py`: chunked loaders for site-level power, inverter and 1-min weather files, with explicit timestamp formats (Zone.Identifier files ignored).
- Smoke-tested on all files: 60 site files (37 optimizer / 23 non-optimizer), 44 inverter files (37 stations), 7 weather variables load cleanly. Rainfall covers 2022-2023 only (`Rainfall_2021.xlsx` not loaded).
- Finding: two timestamp styles exist (ISO `2023-07-01 00:15:00` in 84 files, slash `2021/7/1 0:00` in 20 files, including 10 site files and 13 inverter files). Each file is uniform; the loader picks one of the two whitelisted formats per file and still parses strictly.
- Finding (for the audit, loaders do not alter data): 4 site files have duplicate timestamps (`SQ Apartment37-48 Flexible PV` 432, `SQ14` 586 and not sorted, `Zone L1 (Station 1)` 10099, `Zone L1(Station 2)` 2074). To be handled in Stage 1/2.
- Finding: weather series have fewer rows than the 1,576,800 minutes in 3 years (e.g. irradiance 1,567,792), so there are gaps to profile.
- Next Stage 1 task: build the station inventory from `.ttl` and the CSVs.

## 2026-11-07 (Stage 1: station inventory), hours: _TBD (fill in)_
- Wrote `src/inventory.py` (regex parse of the `.ttl`, file facts from each CSV, join by normalised name) and started `notebooks/01_eda.ipynb` (short basic cells).
- Saved `results/tables/01_station_inventory.csv`: 60 rows (type, resolution, start/end, rows, duplicates, inverter files, lat/lon, altitude, rated kW, tilt, azimuth, contractor). All 60 files matched a `.ttl` system (one manual name fix: Indoor Sports Centre); all rated powers are in kW.
- Counts agree with `DATA_SOURCES.md`: 37 optimizer stations at 15 min, 23 non-optimizer at 60 min, 44 inverter files. (Station-set gate not closed yet; that is a separate task.)
- Notes: 4 optimizer stations start in 2023 (SQ Block P, Shaw Auditorium, UG Hall8, UG Hall9); `Zone L1 (Station 1)` runs to 2024-01-01 03:00 (outside the stated period); tilt/azimuth are kept as raw text (several are `Mixed` or two values like `0/15deg`) and need a documented parsing rule in Stage 2.
- Next Stage 1 task: confirm the timezone and units (W vs kWh; irradiance W/m2).

## 2026-11-07 (Stage 1: timezone and units), hours: _TBD (fill in)_
- Added section 3 to `notebooks/01_eda.ipynb` (basic cells) and `results/figures/01_irradiance_by_hour.png`; documented the results in `DATA_SOURCES.md`.
- Power/energy: hourly stations have kWh x 1000 = W exactly; 15-min stations have summed kWh 2.8-3.3 % above W x 0.25 h / 1000, so `power(W)` is only approximately the step mean (target choice to be decided in Stage 2; W is planned).
- Irradiance is in W/m2, never negative, 739 of 1,567,792 values above 1500 (max 3608.6), night offset median ~5 W/m2.
- Timezone: irradiance centre of mass is 12.0-12.6 h in every month vs expected solar noon ~12.1-12.6 h, so timestamps are local Hong Kong time (UTC+8). Interval start/end convention still open.
- Next Stage 1 task: determine the timestamp convention (interval start vs end label) for power and weather, and check weather/power alignment.

## 2026-11-07 (Stage 1: timestamp convention), hours: _TBD (fill in)_
- Added section 4 to `notebooks/01_eda.ipynb`; tables `results/tables/01_timestamp_alignment_15min.csv` and `_hourly.csv`; figure `results/figures/01_timestamp_alignment.png`.
- Method: correlation between station power and 1-min irradiance averaged over a 15-/60-min window shifted by k minutes (daytime only).
- Result: 15-min optimizer stations peak at k ~ 15 (label = interval START, same in 2021/2022/2023, same for power and generation); hourly stations peak at k between -10 and 0 (label = interval END). The two station types use different conventions. Offset accuracy about +-3 min; relative to the weather timestamps; not stated in the dataset documentation (inferred).
- Weather and power sit on the same minute grid (weather on whole minutes, power on :00/:15/:30/:45 and :00).
- Consequence for Stage 2: a 15-min row labelled t is known only at t+15 min; features and the forecast issue time must respect that.
- Next Stage 1 task: profile missing data and gaps per station and per weather variable.

## 2026-11-07 (Stage 1: missing data and gaps), hours: _TBD (fill in)_
- Added `src/quality.py` (basic gap helpers) and section 5 of `notebooks/01_eda.ipynb`; saved `results/tables/01_missing_summary.csv` (60 stations + 8 weather variables) and figures `01_missing_by_station.png`, `01_missing_by_hour.png`.
- Stations: 15-min optimizer median 0.19 % missing (max 4.2 %); hourly median 1.9 % (max 10.6 %, UG Hall4 Flexible PV). Missing values only occur 07:00-17:59 (night zeros are recorded), so inside that window the worst station misses 23 %.
- Typical gap = one lost daytime of one day (10 steps hourly, 44 steps = 11 h at 15 min). Mostly station-specific; system-wide days: 2021-04-28 (all 11 active), 2021-05-12 (all 12), 2023-09-08 (42 of 60).
- Weather: 0.57-0.86 % of minutes missing; shared gap of 98.7 h from 2023-09-10 06:58 (inside the 2023 test year; exclude those targets in Stage 2, no filling), then 44.7 h (2022-12-02) and 6.6 h (2022-01-19).
- Stage 1 tasks done so far: loaders, station inventory, timezone/units, timestamp convention, missing-data profile. Next: find outliers (negative power, power far above rated, irradiance above 1500 W/m2). User said WORKLOG is not submitted, so entries are optional; this one was requested at the end of the session.
- Nothing pushed; local `main` is ahead of `origin/main`.

## 2026-11-15 (Stage 1 finished, Stage 2 started), hours: _TBD (fill in)_
- Stage 1 tasks done and approved: outliers (section 6; table `01_outlier_summary.csv`), day/night and seasonal profiles (7), power vs irradiance (8; table `01_power_vs_irradiance.csv`), station-set gate (9; table `01_station_sets.csv`: 31 of 37 optimizer and 51 of 60 stations seen in train, validation and test), findings summary and preprocessing to-do list (10). PLAN changes: irradiance limit 1500 -> 2000 W/m2; Stage 2 station sets updated (Change log 2026-11-15).
- Stage 2 started: installed `pvlib` and `pyarrow` (pinned in `requirements.txt`). Added `src/preprocess.py` with `resolve_duplicates` (duplicate copies usually disagree: identical copies merged, disagreeing copies set to missing; details in `DATA_SOURCES.md`) and `to_interval_start` (hourly stations shifted by -1 h).
- Open: cleaning task not finished (next: out-of-range power/irradiance rules, grid, gap fill with `is_filled`, night = 0). `SQ Apartment37-48 Flexible PV` over-rated values in 2023-11/12 coincide with its duplicate timestamps (cause unverified). Licence check and submission/viva dates still open. Per-station % missing and outlier flags are in separate tables, not in the inventory.
- Nothing pushed; local `main` is ahead of `origin/main`.
