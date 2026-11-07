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
