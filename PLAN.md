# PLAN: Explainable and Efficient Machine Learning for Multi-Horizon Solar Power Forecasting

DA487 capstone, 12 credits, standalone. Budget: ~10-12 h/week × 12 weeks ≈ 120-144 h. **Planned: 140 h** (130 h of work + 10 h buffer), i.e. ~11.7 h/week.
Week 1 = week starting **2026-11-09**; week 12 ends **2027-01-31** (adjust the dates if needed; the week numbers stay fixed).
Hard dates: submission = *TBD (user to fill)*, viva = *TBD (user to fill)*.
Status values: `NOT STARTED` / `IN PROGRESS` / `DONE` / `BLOCKED`.

---

## 1. Project summary and objectives

Forecast rooftop PV power for 60 HKUST stations (Dryad dataset, 2021-2023) at three horizons (15 min, 1 h, day-ahead) with models that are **accurate, cheap to run and explainable**, and use the forecast residuals to flag underperforming stations.

**Primary objective:** build and compare a ladder of models (persistence → linear → XGBoost/LightGBM → LSTM) for multi-horizon PV forecasting, and show which model gives the best accuracy-efficiency-explainability trade-off.

**Measurable success criteria.** These are the targets that define success. They are not results.
1. **Accuracy:** on the 2023 test year, the best model has a positive skill score vs the benchmark persistence (defined in §3) at the 15-min and 1-h horizons, reported for all hours and daytime only. For day-ahead (no NWP input, expected to be the weakest horizon), skill is reported with a confidence interval; if it is not positive, that is reported as a finding with an explanation, not hidden.
2. **Efficiency:** train time, end-to-end inference latency (ms/sample, CPU, including feature construction) and model size are reported for every model. At least one model is quantized or exported (ONNX/int8), with its accuracy change measured.
3. **Explainability:** SHAP global importance for the best GBM, plus a stability score (rank correlation of importances) across stations, seasons and retrain seeds.
4. **Fault flagging:** residual-based flagging (using a dedicated expected-power model, see Stage 8) is evaluated on synthetic injected faults, with precision/recall/detection delay and the false-alarm rate on fault-free data, against a simple baseline. The dataset has no fault labels.
5. **Generalisation:** the accuracy drop on held-out (unseen) stations is quantified (station-grouped CV plus a fixed hold-out set). A Kaggle transfer sanity check is reported if it is not cut (see §4).
6. **Uncertainty:** headline skill/nMAE differences between models come with block-bootstrap (by day) confidence intervals.
7. **Deliverables:** a dashboard that runs locally from a clean setup, a report, a video, and a clean repo in which every stage is backed by commits, `WORKLOG.md` entries and figures/tables.

---

## 2. Stages

Every stage writes figures to `results/figures/<stage>_*.png` and tables to `results/tables/<stage>_*.csv`, so the report can pick them up directly.

### Stage 0: Setup, git and data documentation
- **Status:** DONE · **Hours:** 6 · **Week:** 1
- **Goal:** a safe, clean repo and documented data sources.
- **Tasks**
  - [x] Repo structure, venv, `requirements.txt`, `.gitignore` (data/ and models/ excluded).
  - [x] GitHub remote merged without force; old code archived; backup tag and branches pushed.
  - [x] `DATA_SOURCES.md` (primary and secondary datasets).
  - [x] `PLAN.md`, `CLAUDE.md`, `README.md`, `WORKLOG.md`.
- **Carry-over (not part of the DONE status):**
  - [ ] *(user, do in week 1)* Fill in the Kaggle licence and confirm the Dryad licence/authors in `DATA_SOURCES.md`. **This gates committing any demo data in Stage 10.**
- **Inputs:** none. **Outputs:** the root docs and `.gitignore`.
- **Done when:** the repo is clean and pushed, ignore rules are verified, and the backup tag and branches exist.
- **Viva question:** "How do you guarantee reproducibility, and that no data or secrets are in the repo?"

### Stage 1: EDA and data audit (primary data)
- **Status:** NOT STARTED · **Hours:** 12 · **Weeks:** 1-2
- **Goal:** understand the stations, weather and data quality before any modelling, and **confirm or revise the assumptions this plan makes** (station sets, resolutions, location/timezone).
- **Tasks**
  - [ ] Write `src/data_io.py` loaders that read in chunks and use explicit timestamp formats.
  - [ ] Build a station inventory from `.ttl` and the CSVs: type (optimizer or not), resolution, start/end dates, rated power, tilt, azimuth, latitude/longitude.
  - [ ] Confirm the timezone and units (W vs kWh; irradiance W/m²).
  - [ ] **Determine the timestamp convention** (interval start vs end label) for power and for weather, and check the 1-min weather timestamps align with the power timestamps.
  - [ ] Profile missing data and gaps per station and per weather variable.
  - [ ] Find outliers: negative power, implausible maximum vs rated power (one station shows ~31 MW), irradiance above 1500 W/m².
  - [ ] Check day/night patterns and seasonal profiles.
  - [ ] Plot power vs irradiance per station type.
  - [ ] **Station-set gate:** confirm which stations really support each horizon (the plan assumes 37 optimizer stations for 15-min and all 60 for 1-h/day-ahead). If the counts differ, update §2 Stage 2 and add a Change log entry.
  - [ ] Write a findings summary and a preprocessing to-do list.
- **Inputs:** `data/hkust/`.
- **Outputs:**
  - `notebooks/01_eda.ipynb`, `src/data_io.py`
  - `results/tables/01_station_inventory.csv`, `results/tables/01_missing_summary.csv`
  - `results/figures/01_*.png`
- **Done when:** every station has a row in the inventory (resolution, dates, capacity, % missing, outlier flags), the timestamp convention is documented, the station-set gate is closed, and the preprocessing to-do list is written.
- **Viva question:** "What is wrong with the raw data, and how do you know?"

### Stage 2: Preprocessing, features and splits
- **Status:** NOT STARTED · **Hours:** 20 · **Weeks:** 2-4
- **Goal:** model-ready datasets per horizon with no leakage.
- **Tasks**
  - [ ] Install `pvlib` and `pyarrow` (parquet) and add them to `requirements.txt`.
  - [ ] Clean the data:
    - Remove or clip outliers.
    - Reindex to a regular grid.
    - Fill gaps up to a documented maximum; drop longer gaps.
    - **Causal fill only for features** (forward-fill; no interpolation that uses future values).
    - **Keep an `is_filled` mask per row. Targets that were filled are excluded from all evaluation metrics.**
    - Set night to 0 using sun elevation.
  - [ ] Resample (respecting the timestamp convention from Stage 1):
    - Weather from 1-min to 15-min and hourly means.
    - Inverter data from 5-min to 15-min where needed.
  - [ ] Normalise power by rated capacity, so stations are comparable. Any fallback capacity (99th percentile) and any scaler is **computed on train only**.
  - [ ] Build features:
    - Lags and rolling statistics of power and weather (trailing windows only).
    - Calendar features.
    - Solar geometry (elevation/azimuth) and clear-sky index (pvlib; clip the index at low sun elevation).
    - Station metadata (tilt, azimuth, type).
    - **All features use only information available at forecast time.**
  - [ ] Build targets for each horizon (t+15 min, t+1 h, day-ahead per §3).
  - [ ] Choose the station sets (as confirmed by the Stage 1 gate):
    - 15-min horizon: the optimizer stations (plus inverter data).
    - 1-h and day-ahead: all stations with the needed resolution.
  - [ ] Chronological split: train = start-2022-09-30, validation = 2022-10-01-2022-12-31, test = 2023 (full year). Confirm in EDA that each season is in train.
  - [ ] Unseen-station split: hold out ~6 stations (fixed seed; listed in the split table) plus the stations that start in 2023. Also define **station-grouped K-fold** folds for the GBM generalisation estimate (Stage 9). Sets are defined per horizon set.
  - [ ] Unit tests for the leakage checks (no feature timestamp ≥ the forecast issue time; no filled target in evaluation; scalers fitted on train only).
  - [ ] **Vertical slice check:** one horizon (1 h) runs end-to-end through `preprocess → features → splits` with one command before the other horizons are built.
- **Inputs:** `data/hkust/`, Stage 1 to-do list.
- **Outputs:**
  - `src/preprocess.py`, `src/features.py`, `src/splits.py`, `tests/test_leakage.py`
  - `data/processed/*.parquet` (git-ignored)
  - `results/tables/02_split_summary.csv`
  - `notebooks/02_preprocessing_checks.ipynb`
- **Done when:** the datasets can be regenerated with one command, the leakage tests pass, and the split table shows rows/stations/date ranges per split and per horizon.
- **Viva question:** "How do you prevent data leakage, and why this split?"

### Stage 3: Baselines
- **Status:** NOT STARTED · **Hours:** 8 · **Week:** 4
- **Goal:** reference scores that every model must beat.
- **Tasks**
  - [ ] Persistence models (see §3 for the benchmark used in the skill score):
    - Last value (15 min, 1 h).
    - Same hour on the previous day, and the mean of the same hour over the last 7 days (day-ahead).
    - Smart persistence (clear-sky-index persistence).
  - [ ] Linear/ridge regression on the Stage 2 features.
  - [ ] Write `src/metrics.py` (MAE, RMSE, nMAE, skill score; all-hours and daytime; pooled and per-station; block-bootstrap CIs by day).
  - [ ] Evaluate every baseline on validation and test (observed targets only).
- **Inputs:** `data/processed/`.
- **Outputs:**
  - `src/models/baselines.py`, `src/metrics.py`, `tests/test_metrics.py`
  - `results/tables/03_baselines.csv`, `results/figures/03_*.png`
- **Done when:** the baseline table covers all horizons × both metric variants (with CIs), and the evaluation script is reproducible.
- **Viva question:** "Why is persistence the right benchmark, and what does skill score mean?"

### Stage 4: Gradient-boosting models and tuning
- **Status:** NOT STARTED · **Hours:** 14 · **Weeks:** 5-6
- **Goal:** strong, fast tabular models.
- **Tasks**
  - [ ] **First (vertical slice):** train a default LightGBM on the 1-h horizon, evaluate it, and run a quick SHAP summary, to surface pipeline problems early.
  - [ ] Train XGBoost and LightGBM for each horizon (one global model across stations).
  - [ ] Tune with a small time-aware search on validation only (optionally an expanding-window CV inside train), with early stopping.
  - [ ] **Final models: refit on train + validation with the tuned settings and a fixed number of iterations**, then evaluate on test once.
  - [ ] Evaluate on test: per station, per season, daytime-only.
  - [ ] Save the models and their configurations.
- **Inputs:** `data/processed/`.
- **Outputs:**
  - `src/models/gbm.py`, `configs/gbm_*.yaml`, `models/gbm_*` (git-ignored)
  - `results/tables/04_gbm_results.csv`, `results/figures/04_*.png`
- **Done when:** test metrics are reported next to the baselines for all horizons (with CIs), and the tuning log is saved.
- **Viva question:** "Why do tree ensembles work well here, and how did you tune them without touching the test set?"

### Stage 5: LSTM
- **Status:** NOT STARTED · **Hours:** 12 · **Weeks:** 6-7
- **Goal:** a sequence model for comparison.
- **Tasks**
  - [ ] Install `torch` (CPU) and add it to the requirements.
  - [ ] Build windowed sequence datasets lazily (no full in-memory window tensor), with the same features and splits.
  - [ ] Train a small LSTM for the **15-min and 1-h horizons** with early stopping and a fixed seed. Day-ahead LSTM is a stretch goal (first item on the cut list).
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
    - **End-to-end** inference latency (feature construction + prediction; single sample and batch, median over N repeats, CPU, after warm-up), plus model-only latency.
    - Model size on disk.
    - Peak memory, if it is easy to measure.
  - [ ] Export the GBMs to ONNX (time-box this, converters are version-sensitive) and the LSTM to ONNX (fp32).
  - [ ] Apply torch int8 dynamic quantization to the LSTM. This is a separate path from ONNX; do not plan on an int8 ONNX LSTM.
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
  - [ ] TreeSHAP for the best GBM on a test subsample: global importance, dependence plots, a few local examples.
  - [ ] Stability: rank correlation of mean |SHAP| across stations, across seasons, and across retrain seeds / bootstrap resamples.
  - [ ] Correlated features (irradiance vs lags): discuss their shared credit, and group the features (e.g. power lags, weather, solar geometry, calendar, metadata). Expect lag-1 power to dominate the 15-min model; report grouped importance too.
  - [ ] Compare the explanations with physical expectations (irradiance and clear-sky dominate; temperature has a negative effect). These are hypotheses to test, not conclusions to confirm.
- **Inputs:** the best GBM, a test sample.
- **Outputs:**
  - `src/explain.py`, `notebooks/07_shap.ipynb`
  - `results/tables/07_shap_stability.csv`, `results/figures/07_*.png`
- **Done when:** the global and local plots are saved and the stability scores are reported per station, per season and per seed.
- **Viva question:** "Can we trust these explanations? Are they stable?"

### Stage 8: Residual-based fault / underperformance flagging
- **Status:** NOT STARTED · **Hours:** 12 · **Weeks:** 8-9
- **Goal:** flag stations or inverters producing less than expected.
- **Key design point:** forecasts that use the station's own lagged power adapt to a persistent fault (the lags drop too), which shrinks the residual and hides the fault. Therefore flagging uses a **dedicated expected-power model that has no own-power lags** (irradiance, temperature, solar geometry, station metadata only; a nowcast). The Stage 4 forecasts are used only as a comparison.
- **Tasks**
  - [ ] Train the expected-power GBM (no own-power lags) on train data; residual = actual − expected, normalised by capacity.
  - [ ] Flagging rule: rolling daytime residual below a threshold set from validation quantiles.
  - [ ] Baseline detector: performance ratio (power / irradiance-based expectation) or peer-station comparison with a simple threshold.
  - [ ] Inject synthetic faults into test data (e.g. −20 % output for N days, zero output, gradual degradation) at several severities and durations, then measure precision, recall and detection delay for both detectors.
  - [ ] Measure the false-alarm rate (flags per station-month) on fault-free data.
  - [ ] Show, on one example, that a lag-based forecast residual misses the same fault (justifies the design).
  - [ ] Inverter-level comparison within the same station (5-min data). *Cut-list item.*
  - [ ] List the real periods that get flagged, for discussion. These are not ground truth.
- **Inputs:** Stage 2 data, the expected-power model, test data.
- **Outputs:**
  - `src/faults.py`
  - `results/tables/08_fault_detection.csv`, `results/figures/08_*.png`
- **Done when:** precision/recall/delay and false-alarm rate are reported across fault severities, the flagging threshold and its rationale are documented, and the comparison with the baseline detector is shown.
- **Viva question:** "Without fault labels, how do you know your flags are meaningful?"

### Stage 9: Generalisation (unseen stations and Kaggle transfer check)
- **Status:** NOT STARTED · **Hours:** 8 · **Weeks:** 9-10
- **Goal:** test beyond the training stations and, if time allows, beyond the dataset.
- **Tasks**
  - [ ] Evaluate the best models on the held-out stations; compare their nMAE with that of seen stations **over the same 2023 period**.
  - [ ] Station-grouped K-fold for the best GBM (cheap); report the spread, not just one split. LSTM is evaluated on the fixed hold-out only.
  - [ ] **Kaggle transfer sanity check** *(second item on the cut list, see §4)*:
    - Check `DATA_SOURCES.md` for where and when the data comes from. If it is a different location or hardware, call it a transfer check, not an "out-of-time" check, because time, climate and hardware change together.
    - Retrain a reduced common-feature model (irradiance, temperature, solar geometry, lags) since the HKUST-specific features do not exist there.
    - Capacity-normalised, hourly horizon only; report as a sanity check, not a benchmark.
  - [ ] Discuss the limits: one weather source, different hardware/units.
- **Inputs:** the Stage 4-5 models, `data/raw/`.
- **Outputs:**
  - `results/tables/09_generalisation.csv`, `results/figures/09_*.png`
  - `notebooks/09_generalisation.ipynb`
- **Done when:** the unseen-station degradation is quantified (hold-out and grouped CV) and the Kaggle check is reported with its caveats, or explicitly marked cut in the Change log and criterion 5.
- **Viva question:** "Would your model work on a new rooftop or in a new year?"

### Stage 10: Streamlit dashboard
- **Status:** NOT STARTED · **Hours:** 10 · **Weeks:** 10-11
- **Goal:** a demo of forecasts, explanations and flags.
- **Tasks**
  - [ ] Install `streamlit`.
  - [ ] Build the pages:
    - Station selector.
    - Forecast vs actual per horizon.
    - SHAP explanation for the selected time.
    - Residual and fault flags.
    - Efficiency table.
  - [ ] **Demo bundle:** a small sample of processed data plus one small model per horizon, either committed (only if the Stage 0 licence check allows redistribution) or rebuilt by one command (`make demo` / script) from the raw data. The dashboard reads only this bundle.
  - [ ] Test the README steps from a clean clone.
- **Inputs:** saved models, results.
- **Outputs:** `app/streamlit_app.py`, demo-bundle script, `results/figures/10_dashboard_*.png` (screenshots).
- **Done when:** `streamlit run app/streamlit_app.py` works from a clean setup, following the README.
- **Viva question:** "How would an operator use this?"

### Stage 11: Report, video and repo cleanup (+ buffer)
- **Status:** NOT STARTED · **Hours:** 20 (10 for the work + 10 of buffer) · **Weeks:** 11-12
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

**Total:** 6 + 12 + 20 + 8 + 14 + 12 + 8 + 10 + 12 + 8 + 10 + 20 = **140 h** (130 h work + 10 h buffer).

---

## 3. Design decisions (fixed here, revisit only via the Change log)

| Item | Decision |
|---|---|
| **Day-ahead formulation** | *Provisional, confirm in Stage 2.* Issued at 23:00 local time on day D-1; targets are the 24 hourly values of day D. One global model with target-hour as a feature (plus same-hour lags from D-1, D-2, D-7). Every input timestamp ≤ issue time. No NWP input, so this is the weakest horizon by design. |
| **Benchmark for skill score** | 15-min and 1-h: last-value persistence. Day-ahead: same hour on the previous day. Smart (clear-sky-index) persistence is reported as an additional reference at every horizon. |
| **Skill score** | 1 − RMSE_model / RMSE_benchmark. Reported pooled over all stations and as the mean of per-station skill. |
| **Daytime definition** | Sun elevation > 5° (the clear-sky index is unstable below that). Night predictions are set to 0. |
| **Evaluation targets** | Only observed (not filled) target values count. |
| **Final training** | Tune on train → validation; refit on train + validation with fixed settings; one test evaluation per final model. |
| **Uncertainty** | Block bootstrap by day for all headline metrics. |
| **Fault detection** | Dedicated expected-power model without own-power lags; compared with a performance-ratio baseline. |

---

## 4. Week-by-week schedule

| Week | Focus | Stages | Hours |
|---|---|---|---|
| 1 | Setup done; licence check; start EDA | 0, 1 | 12 |
| 2 | Finish EDA (station-set gate); start preprocessing | 1, 2 | 11 |
| 3 | Preprocessing, features, splits | 2 | 11 |
| 4 | Finish splits/tests; baselines | 2, 3 | 12 |
| 5 | GBM vertical slice, training and tuning | 4 | 12 |
| 6 | Finish GBM; start LSTM | 4, 5 | 12 |
| 7 | Finish LSTM; efficiency benchmark; start SHAP | 5, 6, 7 | 12 |
| 8 | SHAP; start fault flagging | 7, 8 | 11 |
| 9 | Fault flagging; start generalisation | 8, 9 | 11 |
| 10 | Generalisation; start dashboard | 9, 10 | 11 |
| 11 | Dashboard; report; **buffer (2 h)** | 10, 11 | 12 |
| 12 | Report, video, polish, submission; **buffer (8 h)** | 11 | 13 |

Total: 140 h. Overlap between adjacent stages within a week is allowed; the order of stages is not changed.

If a stage slips, use the buffer in weeks 11-12 first. If that runs out, cut in this order (approximate hours saved):
1. LSTM day-ahead (already a stretch goal) and LSTM hyper-tuning, ~3 h.
2. Kaggle transfer check, ~4 h. *Update success criterion 5 and Stage 9 "Done when" in the Change log if cut.*
3. Inverter-level fault analysis, ~2 h.

---

## 5. Key design decisions and risks

| Decision / risk | Mitigation |
|---|---|
| **Time-based split with every season in train** | Train covers up to 2022-09-30 (2021 has data from ~mid-year, and 2022 is a full year), validation is Q4 2022, and test is all of 2023. Coverage is checked per season in Stage 2. Validation is a single (dry) season, so tune conservatively and optionally use expanding-window CV inside train. |
| **Final models never see Q4 2022** | Refit on train + validation with fixed settings before the one test run (Stage 4). |
| **Leakage** (future weather, centred rolling windows, scaling fitted on all data, interpolation that uses future values, wrong timestamp convention) | Trailing features only. Causal fill for features. Scalers and fallback capacities fitted on train only. Timestamp convention documented in Stage 1. Leakage unit tests. Day-ahead features use only data up to the issue time. |
| **Filled values inflate scores** | `is_filled` mask; filled targets excluded from evaluation. |
| **Mixed station resolutions** (5-min, 15-min, hourly) | Each horizon uses only stations whose native resolution supports it. Nothing is upsampled: hourly stations are never used for the 15-min horizon. The station sets are confirmed at the Stage 1 gate. |
| **Single campus weather source** (no per-station weather, no NWP) | Use station metadata (tilt, azimuth, capacity) plus the station's own lags. State the limitation clearly. Day-ahead uses clear-sky and lagged features only; report it as the weakest horizon and do not require positive skill for it. |
| **Night-time zeros** inflate all-hours scores | Report metrics for all hours and for daytime only (sun elevation > 5°). Set night predictions to 0. |
| **Capacity differences** between stations | Normalise targets by rated power from the `.ttl`. If the rated power is missing or implausible, use the 99th percentile of train power and document that. Report nMAE. |
| **Outliers / unit errors** (e.g. a ~31 MW max, irradiance up to 3608 W/m²) | Use physical bounds: power ≤ 1.2 × rated, irradiance ≤ 1500 W/m². Flag the values, then clip or drop them as documented in Stage 1. |
| **Stations starting in 2023** | Exclude them from training. Use them in the unseen-station test. |
| **Unseen-station split (6 stations is one draw)** | Fixed seed, listed in `results/tables/02_split_summary.csv`, plus station-grouped K-fold for the GBM. Seen vs unseen compared over the same period. |
| **One test year, differences may be noise** | Block-bootstrap CIs on skill and nMAE. |
| **Faults hidden by lag features** | Dedicated expected-power model without own-power lags (Stage 8). |
| **No fault labels** | Evaluate on synthetic injected faults across severities, with a baseline detector and false-alarm rate. Present real flags only as candidates. |
| **Kaggle data is a different setting** | Treat as a transfer sanity check with a reduced feature set; second item on the cut list (§4). |
| **Stage 2 is the largest and riskiest stage** | 20 h allocated, vertical-slice check at the end of week 3-4, buffer reserved. |
| **Dashboard cannot run from a clean clone (data and models are git-ignored)** | Demo bundle (committed if the licence allows, otherwise rebuilt by one command). |
| **Licence unknown** | Stage 0 carry-over task, done in week 1. Mark unknowns as *unverified*. |
| **Compute limits** (laptop CPU, WSL) | Keep processed data as parquet. Read in chunks. Lazy LSTM windows. Small LSTM. Subsample stations for tuning. Fix the time budget for each search. |
| **Non-determinism** (multi-threaded GBM) | Seed 42 everywhere; use deterministic settings / fixed `n_jobs` for final runs; state residual non-determinism in the report. |
| **Time overrun** | Weekly hour tracking in `WORKLOG.md`, the buffer in weeks 11-12, and the cut list in §4. |

---

## 6. Experiment protocol

- **Metrics:**
  - MAE and RMSE (kW).
  - nMAE = MAE / rated capacity (%).
  - Skill score = 1 − RMSE_model / RMSE_benchmark (benchmark per horizon, see §3).
  - Each is reported for **all hours** and **daytime only**, on observed targets only.
  - Block-bootstrap (by day) 95 % CIs for headline numbers.
- **Grouping:** per horizon, overall, per station type and per season.
- **Seeds:** global seed `42` for numpy, model libraries and torch, set in `src/utils.py`. The unseen-station selection and K-fold assignment use the same seed.
- **Tuning:** validation only. Final models are refit on train + validation; the test set is touched once per final model.
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
app/             Streamlit dashboard (+ demo-bundle script)
report/          report drafts and final
archive/         old Kaggle-India scope (reference only)
PLAN.md  WORKLOG.md  DATA_SOURCES.md  README.md  CLAUDE.md  requirements.txt
```

---

## 8. Resume Protocol (every session)

1. Read `PLAN.md`, `WORKLOG.md`, `DATA_SOURCES.md` and `git log --oneline -15`.
2. Report in 5 lines: current stage, what is done, the next unchecked task, blockers. Then **wait for the go-ahead**.
3. Work on one stage at a time. Do not jump ahead. (Finishing the tail of one stage and starting the next within the same week, as in §4, is allowed.)
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
| 2026-11-06 | Plan review applied. Added §3 (design decisions). Stage 8 now uses an expected-power model without own-power lags plus a baseline detector and false-alarm rate. Day-ahead formulation, issue time and benchmark persistence defined; criterion 1 reworded for day-ahead. Final models refit on train + validation. Causal fill, `is_filled` mask, timestamp convention and train-only capacity fallback added. Station-set gate added to Stage 1. Kaggle check reframed as a transfer check and its cut-list link to criterion 5 made explicit. Station-grouped K-fold and block-bootstrap CIs added. End-to-end latency, ONNX/int8 paths separated. SHAP stability extended to retrain seeds. Dashboard demo bundle added. Hours rebalanced: Stage 2 16→20, Stage 5 14→12 (LSTM for 15-min and 1-h only), Stage 8 10→12; total 140 h with a 10 h buffer; weekly table now sums to the stage total. Stage 0 licence task moved to a carry-over line. Section 2-3 heading renumbered. | Review findings: leakage and fault-masking risks, schedule not adding up (136 h stages vs 132 h weekly vs 130 h budget), inconsistent Kaggle cut/criterion. |
| 2026-11-06 | Kaggle check labelled the second cut-list item in Stage 9 and §5 (matching §4). Stage 2 task added to install `pvlib` and `pyarrow`. | Consistency fixes after reviewing the user's plan edit. |
