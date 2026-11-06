# PLAN: Explainable and Efficient Machine Learning for Multi-Horizon Solar Power Forecasting

DA487 capstone, 12 credits, standalone. Budget: ~10-12 h/week × 12 weeks ≈ **130 h**.
Week 1 = week starting **2026-11-09** (adjust the dates if needed; the week numbers stay fixed).
Status values: `NOT STARTED` / `IN PROGRESS` / `DONE` / `BLOCKED`.

---

## 1. Project summary and objectives

Forecast rooftop PV power for 60 HKUST stations (Dryad dataset, 2021-2023) at three horizons (15 min, 1 h, day-ahead) with models that are **accurate, cheap to run and explainable**, and use the forecast residuals to flag underperforming stations.

**Primary objective:** build and compare a ladder of models (persistence → linear → XGBoost/LightGBM → LSTM) for multi-horizon PV forecasting, and show which model gives the best accuracy-efficiency-explainability trade-off.

**Measurable success criteria.** These are the targets that define success. They are not results.
1. **Accuracy:** the best model has a positive skill score vs persistence on the 2023 test year at every horizon it is evaluated on, reported for all hours and daytime only.
2. **Efficiency:** train time, inference latency (ms/sample, CPU) and model size are reported for every model. At least one model is quantized or exported (ONNX/int8), with its accuracy change measured.
3. **Explainability:** SHAP global importance for the best GBM, plus a stability score (rank correlation of importances) across stations and seasons.
4. **Fault flagging:** residual-based flagging is evaluated on synthetic injected faults, with precision/recall reported, because the dataset has no fault labels.
5. **Generalisation:** the accuracy drop on held-out (unseen) stations is quantified, and a Kaggle out-of-time sanity check is reported.
6. **Deliverables:** a dashboard that runs locally, a report, a video, and a clean repo in which every stage is backed by commits, `WORKLOG.md` entries and figures/tables.

---

## 2-3. Stages

Every stage writes figures to `results/figures/<stage>_*.png` and tables to `results/tables/<stage>_*.csv`, so the report can pick them up directly.

### Stage 0: Setup, git and data documentation
- **Status:** DONE · **Hours:** 6 · **Week:** 1
- **Goal:** a safe, clean repo and documented data sources.
- **Tasks**
  - [x] Repo structure, venv, `requirements.txt`, `.gitignore` (data/ and models/ excluded).
  - [x] GitHub remote merged without force; old code archived; backup tag and branches pushed.
  - [x] `DATA_SOURCES.md` (primary and secondary datasets).
  - [x] `PLAN.md`, `CLAUDE.md`, `README.md`, `WORKLOG.md`.
  - [ ] *(user)* Fill in the Kaggle licence and confirm the Dryad licence/authors in `DATA_SOURCES.md`.
- **Inputs:** none. **Outputs:** the root docs and `.gitignore`.
- **Done when:** the repo is clean and pushed, ignore rules are verified, and the backup tag and branches exist.
- **Viva question:** "How do you guarantee reproducibility, and that no data or secrets are in the repo?"

### Stage 1: EDA and data audit (primary data)
- **Status:** NOT STARTED · **Hours:** 12 · **Weeks:** 1-2
- **Goal:** understand the stations, weather and data quality before any modelling.
- **Tasks**
  - [ ] Write `src/data_io.py` loaders that read in chunks and use explicit timestamp formats.
  - [ ] Build a station inventory from `.ttl` and the CSVs: type (optimizer or not), resolution, start/end dates, rated power, tilt, azimuth.
  - [ ] Confirm the timezone and units (W vs kWh; irradiance W/m²).
  - [ ] Profile missing data and gaps per station and per weather variable.
  - [ ] Find outliers: negative power, implausible maximum vs rated power (one station shows ~31 MW), irradiance above 1500 W/m².
  - [ ] Check day/night patterns and seasonal profiles.
  - [ ] Plot power vs irradiance per station type.
  - [ ] Check whether the 1-min weather timestamps align with the power timestamps.
  - [ ] Write a findings summary and a preprocessing to-do list.
- **Inputs:** `data/hkust/`.
- **Outputs:**
  - `notebooks/01_eda.ipynb`, `src/data_io.py`
  - `results/tables/01_station_inventory.csv`, `results/tables/01_missing_summary.csv`
  - `results/figures/01_*.png`
- **Done when:** every station has a row in the inventory (resolution, dates, capacity, % missing, outlier flags), and the preprocessing to-do list is written.
- **Viva question:** "What is wrong with the raw data, and how do you know?"

### Stage 2: Preprocessing, features and splits
- **Status:** NOT STARTED · **Hours:** 16 · **Weeks:** 2-3
- **Goal:** model-ready datasets per horizon with no leakage.
- **Tasks**
  - [ ] Clean the data:
    - Remove or clip outliers.
    - Reindex to a regular grid.
    - Interpolate gaps up to a documented maximum; drop longer gaps.
    - Set night to 0 using sun elevation.
  - [ ] Resample:
    - Weather from 1-min to 15-min and hourly means.
    - Inverter data from 5-min to 15-min where needed.
  - [ ] Normalise power by rated capacity, so stations are comparable.
  - [ ] Build features:
    - Lags and rolling statistics of power and weather.
    - Calendar features.
    - Solar geometry (elevation/azimuth) and clear-sky index.
    - Station metadata (tilt, azimuth, type).
    - **All features use only information available at forecast time.**
  - [ ] Build targets for each horizon (t+15 min, t+1 h, day-ahead = next day's 24 hourly values, issued at a fixed time).
  - [ ] Choose the station sets:
    - 15-min horizon: the 37 optimizer stations (plus inverter data).
    - 1-h and day-ahead: all 60 stations.
  - [ ] Chronological split: train = start-2022-09-30, validation = 2022-10-01-2022-12-31, test = 2023 (full year). Confirm in EDA that each season is in train.
  - [ ] Unseen-station split: hold out ~6 stations stratified by type and capacity, plus the stations that start in 2023.
  - [ ] Unit tests for the leakage checks (no feature timestamp ≥ target timestamp).
- **Inputs:** `data/hkust/`, Stage 1 to-do list.
- **Outputs:**
  - `src/preprocess.py`, `src/features.py`, `src/splits.py`, `tests/test_leakage.py`
  - `data/processed/*.parquet` (git-ignored)
  - `results/tables/02_split_summary.csv`
  - `notebooks/02_preprocessing_checks.ipynb`
- **Done when:** the datasets can be regenerated with one command, the leakage tests pass, and the split table shows rows/stations/date ranges per split.
- **Viva question:** "How do you prevent data leakage, and why this split?"

### Stage 3: Baselines
- **Status:** NOT STARTED · **Hours:** 8 · **Week:** 4
- **Goal:** reference scores that every model must beat.
- **Tasks**
  - [ ] Persistence models:
    - Last value (15 min, 1 h).
    - Same time yesterday (day-ahead).
    - Smart persistence (clear-sky-index persistence).
  - [ ] Linear/ridge regression on the Stage 2 features.
  - [ ] Write `src/metrics.py` (MAE, RMSE, nMAE, skill score; all-hours and daytime).
  - [ ] Evaluate every baseline on validation and test.
- **Inputs:** `data/processed/`.
- **Outputs:**
  - `src/models/baselines.py`, `src/metrics.py`
  - `results/tables/03_baselines.csv`, `results/figures/03_*.png`
- **Done when:** the baseline table covers all horizons × both metric variants, and the evaluation script is reproducible.
- **Viva question:** "Why is persistence the right benchmark, and what does skill score mean?"

### Stage 4: Gradient-boosting models and tuning
- **Status:** NOT STARTED · **Hours:** 14 · **Weeks:** 4-5
- **Goal:** strong, fast tabular models.
- **Tasks**
  - [ ] Train XGBoost and LightGBM for each horizon (one global model across stations).
  - [ ] Tune with a small time-aware search on validation only, with early stopping.
  - [ ] Evaluate on test: per station, per season, daytime-only.
  - [ ] Save the models and their configurations.
- **Inputs:** `data/processed/`.
- **Outputs:**
  - `src/models/gbm.py`, `configs/gbm_*.yaml`, `models/gbm_*` (git-ignored)
  - `results/tables/04_gbm_results.csv`, `results/figures/04_*.png`
- **Done when:** test metrics beat or are compared with the baselines for all horizons, and the tuning log is saved.
- **Viva question:** "Why do tree ensembles work well here, and how did you tune them without touching the test set?"

### Stage 5: LSTM
- **Status:** NOT STARTED · **Hours:** 14 · **Week:** 6
- **Goal:** a sequence model for comparison.
- **Tasks**
  - [ ] Install `torch` (CPU) and add it to the requirements.
  - [ ] Build windowed sequence datasets (same features, same splits).
  - [ ] Train a small LSTM per horizon with early stopping and a fixed seed.
  - [ ] Evaluate it with the same metrics, and compare it with the GBMs.
- **Inputs:** `data/processed/`.
- **Outputs:**
  - `src/models/lstm.py`, `models/lstm_*` (git-ignored)
  - `results/tables/05_lstm_results.csv`, `results/figures/05_*.png`
- **Done when:** the LSTM results sit in the same table format as Stages 3-4, and the training curves are saved.
- **Viva question:** "Is the deep model worth its cost compared with the GBMs?"

### Stage 6: Efficiency benchmark
- **Status:** NOT STARTED · **Hours:** 8 · **Week:** 7
- **Goal:** a fair cost comparison.
- **Tasks**
  - [ ] Write `src/benchmark.py`:
    - Train time.
    - Inference latency (single sample and batch, median over N repeats, CPU, after warm-up).
    - Model size on disk.
    - Peak memory, if it is easy to measure.
  - [ ] Export to ONNX (LightGBM/XGBoost; LSTM), and apply int8 dynamic quantization to the LSTM.
  - [ ] Measure the accuracy change after quantization or export.
  - [ ] Plot the accuracy-vs-latency trade-off.
- **Inputs:** the Stage 3-5 models.
- **Outputs:**
  - `src/benchmark.py`
  - `results/tables/06_efficiency.csv` (including hardware specs)
  - `results/figures/06_accuracy_vs_latency.png`
- **Done when:** every model has all the cost metrics measured on the same hardware, recorded together with that hardware's specs.
- **Viva question:** "Which model would you deploy, and why?"

### Stage 7: SHAP explainability
- **Status:** NOT STARTED · **Hours:** 10 · **Weeks:** 7-8
- **Goal:** explanations, and evidence that they are stable.
- **Tasks**
  - [ ] TreeSHAP for the best GBM: global importance, dependence plots, a few local examples.
  - [ ] Stability: rank correlation of mean |SHAP| across stations and across seasons.
  - [ ] Correlated features (irradiance vs lags): discuss their shared credit, and optionally group the features.
  - [ ] Check that the explanations agree with the physics (irradiance and clear-sky dominate; temperature has a negative effect).
- **Inputs:** the best GBM, a test sample.
- **Outputs:**
  - `src/explain.py`, `notebooks/07_shap.ipynb`
  - `results/tables/07_shap_stability.csv`, `results/figures/07_*.png`
- **Done when:** the global and local plots are saved and the stability scores are reported per station and per season.
- **Viva question:** "Can we trust these explanations? Are they stable?"

### Stage 8: Residual-based fault / underperformance flagging
- **Status:** NOT STARTED · **Hours:** 10 · **Weeks:** 8-9
- **Goal:** flag stations or inverters producing less than expected.
- **Tasks**
  - [ ] Expected power = model forecast (or a nowcast); residual = actual − expected, normalised by capacity.
  - [ ] Flagging rule: rolling daytime residual below a threshold set from validation quantiles.
  - [ ] Inject synthetic faults into test data (e.g. −20 % output for N days, zero output, gradual degradation), then measure precision, recall and detection delay.
  - [ ] Inverter-level comparison within the same station (5-min data).
  - [ ] List the real periods that get flagged, for discussion. These are not ground truth.
- **Inputs:** Stage 4 models, test data.
- **Outputs:**
  - `src/faults.py`
  - `results/tables/08_fault_detection.csv`, `results/figures/08_*.png`
- **Done when:** precision/recall on the injected faults are reported, and the flagging threshold and its rationale are documented.
- **Viva question:** "Without fault labels, how do you know your flags are meaningful?"

### Stage 9: Generalisation (unseen stations and Kaggle out-of-time check)
- **Status:** NOT STARTED · **Hours:** 8 · **Week:** 9
- **Goal:** test beyond the training stations and period.
- **Tasks**
  - [ ] Evaluate the best models on the held-out stations; compare their nMAE with that of the seen stations.
  - [ ] Kaggle out-of-time sanity check (2024-25):
    - Map to comparable units (capacity-normalised).
    - Hourly horizon only.
    - Report it as a sanity check, not a benchmark.
  - [ ] Discuss the limits: one weather source, different hardware/units.
- **Inputs:** the Stage 4-5 models, `data/raw/`.
- **Outputs:**
  - `results/tables/09_generalisation.csv`, `results/figures/09_*.png`
  - `notebooks/09_generalisation.ipynb`
- **Done when:** the unseen-station degradation is quantified and the Kaggle check is reported with its caveats.
- **Viva question:** "Would your model work on a new rooftop or in a new year?"

### Stage 10: Streamlit dashboard
- **Status:** NOT STARTED · **Hours:** 10 · **Week:** 10
- **Goal:** a demo of forecasts, explanations and flags.
- **Tasks**
  - [ ] Install `streamlit`.
  - [ ] Build the pages:
    - Station selector.
    - Forecast vs actual per horizon.
    - SHAP explanation for the selected time.
    - Residual and fault flags.
    - Efficiency table.
  - [ ] Make the dashboard read only saved models and processed samples (small demo data, git-ignored or regenerated).
- **Inputs:** saved models, results.
- **Outputs:** `app/streamlit_app.py`, `results/figures/10_dashboard_*.png` (screenshots).
- **Done when:** `streamlit run app/streamlit_app.py` works from a clean setup, following the README.
- **Viva question:** "How would an operator use this?"

### Stage 11: Report, video and repo cleanup (+ buffer)
- **Status:** NOT STARTED · **Hours:** 20 (≈12 for the work + ≈8 of buffer) · **Weeks:** 11-12
- **Goal:** final deliverables.
- **Tasks**
  - [ ] Fill in the report template (shared later) from `results/`.
  - [ ] Record the presentation video.
  - [ ] Tidy the repo:
    - Bring the README reproduce steps up to date.
    - Pin `requirements.txt`.
    - Strip the notebook outputs.
    - Run the code end-to-end from a clean clone.
  - [ ] Prepare viva answers from each stage's "Viva question".
- **Inputs:** everything.
- **Outputs:** `report/`, a video link in the README, a final tag `v1.0`.
- **Done when:** a clean clone reproduces the key tables, and the report and video are submitted.
- **Viva question:** "Walk us through your contribution and its limitations."

**Total:** 6 + 12 + 16 + 8 + 14 + 14 + 8 + 10 + 10 + 8 + 10 + 20 = **136 h**

---

## 4. Week-by-week schedule

| Week | Focus | Stages | Hours |
|---|---|---|---|
| 1 | Setup done; start EDA | 0, 1 | ~12 |
| 2 | Finish EDA; start preprocessing | 1, 2 | ~11 |
| 3 | Preprocessing, features, splits | 2 | ~11 |
| 4 | Baselines; start GBM | 3, 4 | ~12 |
| 5 | GBM tuning and evaluation | 4 | ~10 |
| 6 | LSTM | 5 | ~12 |
| 7 | Efficiency benchmark; start SHAP | 6, 7 | ~12 |
| 8 | SHAP; start fault flagging | 7, 8 | ~11 |
| 9 | Fault flagging; generalisation | 8, 9 | ~11 |
| 10 | Dashboard | 10 | ~10 |
| 11 | Report and video (**buffer**) | 11 | ~10 |
| 12 | Final polish, submission (**buffer**) | 11 | ~10 |

If a stage slips, use the buffer in weeks 11-12 first. If that runs out, cut in this order: Kaggle check → inverter-level fault analysis → LSTM hyper-tuning.

---

## 5. Key design decisions and risks

| Decision / risk | Mitigation |
|---|---|
| **Time-based split with every season in train** | Train covers up to 2022-09-30 (2021 has data from ~mid-year, and 2022 is a full year), validation is Q4 2022, and test is all of 2023. Coverage is checked per season in Stage 2. |
| **Leakage** (future weather, centred rolling windows, scaling fitted on all data) | Use only lagged/trailing features. Fit scalers on train only. Run a leakage unit test. Day-ahead features use only data up to the issue time. |
| **Mixed station resolutions** (5-min, 15-min, hourly) | Each horizon uses only stations whose native resolution supports it. Nothing is upsampled: hourly stations are never used for the 15-min horizon. |
| **Single campus weather source** (no per-station weather, no NWP) | Use station metadata (tilt, azimuth, capacity) plus the station's own lags. State the limitation clearly. For day-ahead, use clear-sky and lagged features only, and report it as the weakest horizon. |
| **Night-time zeros** inflate all-hours scores | Report metrics for all hours and for daytime only (sun elevation > 0). Set night predictions to 0. |
| **Capacity differences** between stations | Normalise targets by rated power from the `.ttl`. If the rated power is missing or implausible, use the 99th percentile of power and document that. Report nMAE. |
| **Outliers / unit errors** (e.g. a ~31 MW max, irradiance up to 3608 W/m²) | Use physical bounds: power ≤ 1.2 × rated, irradiance ≤ 1500 W/m². Flag the values, then clip or drop them as documented in Stage 1. |
| **Stations starting in 2023** | Exclude them from training. Use them in the unseen-station test. |
| **Unseen-station split** | Hold out ~6 stations stratified by type and capacity, fixed by a seed and listed in `results/tables/02_split_summary.csv`. |
| **Compute limits** (laptop CPU, WSL) | Keep processed data as parquet. Read in chunks. Use a small LSTM. Subsample stations for tuning. Fix the time budget for each search. |
| **No fault labels** | Evaluate on synthetic injected faults. Present real flags only as candidates. |
| **Time overrun** | Weekly hour tracking in `WORKLOG.md`, the buffer in weeks 11-12, and the cut list in §4. |

---

## 6. Experiment protocol

- **Metrics:**
  - MAE and RMSE (kW).
  - nMAE = MAE / rated capacity (%).
  - Skill score = 1 − RMSE_model / RMSE_persistence.
  - Each is reported for **all hours** and **daytime only**.
- **Grouping:** per horizon, overall, per station type and per season.
- **Seeds:** global seed `42` for numpy, model libraries and torch, set in `src/utils.py`. The unseen-station selection uses the same seed.
- **Tuning:** validation set only. The test set is touched once per final model.
- **Results:**
  - Tables go in `results/tables/<stage>_<name>.csv` and figures in `results/figures/<stage>_<name>.png`.
  - Model configs go in `configs/`; models go in `models/` (git-ignored).
  - Each results table records the git commit hash and the date.
- **Hardware for timing:** CPU model, cores, RAM and OS are recorded in `results/tables/06_efficiency.csv`.

---

## 7. Repo structure

```
data/            hkust/ (primary), raw/ (Kaggle secondary), processed/: all git-ignored
notebooks/       01_eda.ipynb, 02_..., exploration and figures only
src/             data_io, preprocess, features, splits, metrics, benchmark, explain, faults, utils, models/
tests/           leakage and metric tests
configs/         model/experiment configs
models/          saved models (git-ignored)
results/         figures/, tables/ (inputs for the report)
app/             Streamlit dashboard
report/          report drafts and final
archive/         old Kaggle-India scope (reference only)
PLAN.md  WORKLOG.md  DATA_SOURCES.md  README.md  CLAUDE.md  requirements.txt
```

---

## 8. Resume Protocol (every session)

1. Read `PLAN.md`, `WORKLOG.md`, `DATA_SOURCES.md` and `git log --oneline -15`.
2. Report in 5 lines: current stage, what is done, the next unchecked task, blockers. Then **wait for the go-ahead**.
3. Work on one stage at a time. Do not jump ahead.
4. After each task:
   - Tick the box here and update the stage Status.
   - Add a `WORKLOG.md` entry (date, hours, what was done).
   - Make a small commit (`feat:` / `docs:` / `chore:` / `fix:`).
5. If the plan must change, edit `PLAN.md` first, add a Change log entry below, and tell the user.
6. When a stage is done, give a 10-line summary with key numbers and **ask for approval** before the next stage.
7. Never push without approval. Never force push, rebase, `reset --hard` or amend pushed commits. Never commit `data/` or `models/`. Never fabricate numbers, citations or licences; mark anything unknown as *unverified*.

---

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-11-06 | Replaced the Stage 0-only plan with the full 12-stage plan. | The user requested a master plan. |
