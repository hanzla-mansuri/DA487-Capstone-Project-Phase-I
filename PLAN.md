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
- **Status:** DONE · **Hours:** 12 · **Weeks:** 1-2
- **Goal:** understand the stations, weather and data quality before any modelling, and **confirm or revise the assumptions this plan makes** (station sets, resolutions, location/timezone).
- **Tasks**
  - [x] Write `src/data_io.py` loaders that read in chunks and use explicit timestamp formats.
  - [x] Build a station inventory from `.ttl` and the CSVs: type (optimizer or not), resolution, start/end dates, rated power, tilt, azimuth, latitude/longitude.
  - [x] Confirm the timezone and units (W vs kWh; irradiance W/m²).
  - [x] **Determine the timestamp convention** (interval start vs end label) for power and for weather, and check the 1-min weather timestamps align with the power timestamps.
  - [x] Profile missing data and gaps per station and per weather variable.
  - [x] Find outliers: negative power, implausible maximum vs rated power (one station shows ~31 MW), irradiance above 1500 W/m².
  - [x] Check day/night patterns and seasonal profiles.
  - [x] Plot power vs irradiance per station type.
  - [x] **Station-set gate:** confirm which stations really support each horizon (the plan assumes 37 optimizer stations for 15-min and all 60 for 1-h/day-ahead). If the counts differ, update §2 Stage 2 and add a Change log entry.
  - [x] Write a findings summary and a preprocessing to-do list.
- **Inputs:** `data/hkust/`.
- **Outputs:**
  - `notebooks/01_eda.ipynb`, `src/data_io.py`
  - `results/tables/01_station_inventory.csv`, `results/tables/01_missing_summary.csv`
  - `results/figures/01_*.png`
- **Done when:** every station has a row in the inventory (resolution, dates, capacity, % missing, outlier flags), the timestamp convention is documented, the station-set gate is closed, and the preprocessing to-do list is written.
- **Viva question:** "What is wrong with the raw data, and how do you know?"

### Stage 2: Preprocessing, features and splits
- **Status:** DONE · **Hours:** 20 · **Weeks:** 2-4
- **Goal:** model-ready datasets per horizon with no leakage.
- **Tasks**
  - [x] Install `pvlib` and `pyarrow` (parquet) and add them to `requirements.txt`.
  - [x] Clean the data:
    - Remove or clip outliers.
    - Reindex to a regular grid.
    - Fill gaps up to a documented maximum; drop longer gaps.
    - **Causal fill only for features** (forward-fill; no interpolation that uses future values).
    - **Keep an `is_filled` mask per row. Targets that were filled are excluded from all evaluation metrics.**
    - Set night to 0 using sun elevation.
  - [x] Resample (respecting the timestamp convention from Stage 1):
    - Weather from 1-min to 15-min and hourly means.
    - Inverter data from 5-min to 15-min where needed. *Deferred to Stage 8 (inverter-level comparison, a cut-list item): the site-level files already give 15-min power for all optimizer stations.*
  - [x] Normalise power by rated capacity, so stations are comparable. Any fallback capacity (99th percentile) and any scaler is **computed on train only**.
  - [x] Build features:
    - Lags and rolling statistics of power and weather (trailing windows only).
    - Calendar features.
    - Solar geometry (elevation/azimuth) and clear-sky index (pvlib; clip the index at low sun elevation).
    - Station metadata (tilt, azimuth, type).
    - **All features use only information available at forecast time.**
  - [x] Build targets for each horizon (t+15 min, t+1 h, day-ahead per §3).
  - [x] Choose the station sets (as confirmed by the Stage 1 gate):
    - 15-min horizon: the optimizer stations (plus inverter data): 37 have the resolution, 31 are seen in train, validation and test (Stage 1 gate, `results/tables/01_station_sets.csv`). **`Zone A2` is used from 2023-05-31 only** (capacity change), so it becomes test-only: 30 seen.
    - 1-h and day-ahead: all stations with the needed resolution: 60, of which 51 are seen in train, validation and test. **`UG Hall4 Flexible PV` is dropped** (zero output in 64 % of high-irradiance hours), leaving 59, of which 50 are seen; with `Zone A2` test-only, 49 are seen.
    - Stations with little or no training data are the first candidates for the unseen-station set.
  - [x] Chronological split: train = start-2022-09-30, validation = 2022-10-01-2022-12-31, test = 2023 (full year). Confirm in EDA that each season is in train.
  - [x] Unseen-station split: hold out ~6 stations (fixed seed; listed in the split table) plus the stations that start in 2023. Also define **station-grouped K-fold** folds for the GBM generalisation estimate (Stage 9). Sets are defined per horizon set.
  - [x] Unit tests for the leakage checks (no feature timestamp ≥ the forecast issue time; no filled target in evaluation; scalers fitted on train only).
  - [x] **Vertical slice check:** one horizon (1 h) runs end-to-end through `preprocess → features → splits` with one command before the other horizons are built.
- **Inputs:** `data/hkust/`, Stage 1 to-do list.
- **Outputs:**
  - `src/preprocess.py`, `src/features.py`, `src/splits.py`, `tests/test_leakage.py`
  - `data/processed/*.parquet` (git-ignored)
  - `results/tables/02_split_summary.csv`
  - `notebooks/02_preprocessing_checks.ipynb`
- **Done when:** the datasets can be regenerated with one command, the leakage tests pass, and the split table shows rows/stations/date ranges per split and per horizon.
- **Viva question:** "How do you prevent data leakage, and why this split?"

### Stage 3: Baselines
- **Status:** DONE · **Hours:** 8 · **Week:** 4
- **Goal:** reference scores that every model must beat.
- **Tasks**
  - [x] Persistence models (see §3 for the benchmark used in the skill score):
    - Last value (15 min, 1 h).
    - Same hour on the previous day, and the mean of the same hour over the last 7 days (day-ahead).
    - Smart persistence (clear-sky-index persistence).
  - [x] Linear/ridge regression on the Stage 2 features.
  - [x] Write `src/metrics.py` (MAE, RMSE, nMAE, skill score; all-hours and daytime; pooled and per-station; block-bootstrap CIs by day).
  - [x] Evaluate every baseline on validation and test (observed targets only).
- **Inputs:** `data/processed/`.
- **Outputs:**
  - `src/models/baselines.py`, `src/metrics.py`, `tests/test_metrics.py`
  - `results/tables/03_baselines.csv`, `results/figures/03_*.png`
- **Done when:** the baseline table covers all horizons × both metric variants (with CIs), and the evaluation script is reproducible.
- **Viva question:** "Why is persistence the right benchmark, and what does skill score mean?"

### Stage 4: Gradient-boosting models and tuning
- **Status:** DONE · **Hours:** 14 · **Weeks:** 5-6
- **Goal:** strong, fast tabular models.
- **Tasks**
  - [x] **First (vertical slice):** train a default LightGBM on the 1-h horizon, evaluate it, and run a quick SHAP summary, to surface pipeline problems early.
  - [x] Train XGBoost and LightGBM for each horizon (one global model across stations).
  - [x] Tune with a small time-aware search on validation only (optionally an expanding-window CV inside train), with early stopping.
  - [x] **Final models: refit on train + validation with the tuned settings and a fixed number of iterations**, then evaluate on test once.
  - [x] Evaluate on test: per station, per season, daytime-only.
  - [x] Save the models and their configurations.
- **Inputs:** `data/processed/`.
- **Outputs:**
  - `src/models/gbm.py`, `configs/gbm_*.yaml`, `models/gbm_*` (git-ignored)
  - `results/tables/04_gbm_results.csv`, `results/figures/04_*.png`
- **Done when:** test metrics are reported next to the baselines for all horizons (with CIs), and the tuning log is saved.
- **Viva question:** "Why do tree ensembles work well here, and how did you tune them without touching the test set?"

### Stage 5: LSTM
- **Status:** DONE · **Hours:** 12 · **Weeks:** 6-7
- **Goal:** a sequence model for comparison.
- **Tasks**
  - [x] Install `torch` (CPU) and add it to the requirements.
  - [x] Build windowed sequence datasets lazily (no full in-memory window tensor), with the same features and splits.
  - [x] Train a small LSTM for the **15-min and 1-h horizons** with early stopping and a fixed seed. Day-ahead LSTM is a stretch goal (first item on the cut list).
  - [x] Evaluate it with the same metrics, and compare it with the GBMs.
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
- **Status:** DONE · **Hours:** 10 · **Weeks:** 7-8
- **Goal:** explanations, and evidence that they are stable.
- **Tasks**
  - [x] TreeSHAP for the best GBM on a test subsample: global importance, dependence plots, a few local examples.
  - [x] Stability: rank correlation of mean |SHAP| across stations, across seasons, and across retrain seeds / bootstrap resamples.
  - [x] Correlated features (irradiance vs lags): discuss their shared credit, and group the features (e.g. power lags, weather, solar geometry, calendar, metadata). Expect lag-1 power to dominate the 15-min model; report grouped importance too.
  - [x] Compare the explanations with physical expectations (irradiance and clear-sky dominate; temperature has a negative effect). These are hypotheses to test, not conclusions to confirm.
- **Inputs:** the best GBM, a test sample.
- **Outputs:**
  - `src/explain.py`, `notebooks/07_shap.ipynb`
  - `results/tables/07_shap_stability.csv`, `results/figures/07_*.png`
- **Done when:** the global and local plots are saved and the stability scores are reported per station, per season and per seed.
- **Viva question:** "Can we trust these explanations? Are they stable?"

### Stage 8: Residual-based fault / underperformance flagging
- **Status:** DONE · **Hours:** 12 · **Weeks:** 8-9
- **Goal:** flag stations or inverters producing less than expected.
- **Key design point:** forecasts that use the station's own lagged power adapt to a persistent fault (the lags drop too), which shrinks the residual and hides the fault. Therefore flagging uses a **dedicated expected-power model that has no own-power lags** (irradiance, temperature, solar geometry, station metadata only; a nowcast). The Stage 4 forecasts are used only as a comparison.
- **Tasks**
  - [x] Train the expected-power GBM (no own-power lags) on train data; residual = actual − expected, normalised by capacity.
  - [x] Flagging rule: rolling daytime residual below a threshold set on the calibration period (changed from validation quantiles to a robust rule, see Change log 2026-11-16).
  - [x] Baseline detector: performance ratio (power / irradiance-based expectation) or peer-station comparison with a simple threshold.
  - [x] Inject synthetic faults into test data (e.g. −20 % output for N days, zero output, gradual degradation) at several severities and durations, then measure precision, recall and detection delay for both detectors.
  - [x] Measure the false-alarm rate (flags per station-month) on fault-free data.
  - [x] Show, on one example, that a lag-based forecast residual misses the same fault (justifies the design).
  - [ ] Inverter-level comparison within the same station (5-min data). *Cut-list item: **cut**, see Change log 2026-11-16.*
  - [x] List the real periods that get flagged, for discussion. These are not ground truth.
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
| **Outliers / unit errors** (e.g. a ~31 MW max, irradiance up to 3608 W/m²) | Use physical bounds: power ≤ 1.2 × rated, irradiance ≤ 2000 W/m² (1500-2000 flagged and kept). Flag the values, then clip or drop them as documented in Stage 1. |
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
7. **Code style:** all code in every notebook is very basic. Lengthy code is split into separate blocks (cells). No complex code.
8. Never push without approval. Never force push, rebase, `reset --hard` or amend pushed commits. Never commit `data/` or `models/`. Never fabricate numbers, citations or licences; mark anything unknown as *unverified*.

---

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-11-06 | Replaced the Stage 0-only plan with the full 12-stage plan. | The user requested a master plan. |
| 2026-11-06 | Plan review applied. Added §3 (design decisions). Stage 8 now uses an expected-power model without own-power lags plus a baseline detector and false-alarm rate. Day-ahead formulation, issue time and benchmark persistence defined; criterion 1 reworded for day-ahead. Final models refit on train + validation. Causal fill, `is_filled` mask, timestamp convention and train-only capacity fallback added. Station-set gate added to Stage 1. Kaggle check reframed as a transfer check and its cut-list link to criterion 5 made explicit. Station-grouped K-fold and block-bootstrap CIs added. End-to-end latency, ONNX/int8 paths separated. SHAP stability extended to retrain seeds. Dashboard demo bundle added. Hours rebalanced: Stage 2 16→20, Stage 5 14→12 (LSTM for 15-min and 1-h only), Stage 8 10→12; total 140 h with a 10 h buffer; weekly table now sums to the stage total. Stage 0 licence task moved to a carry-over line. Section 2-3 heading renumbered. | Review findings: leakage and fault-masking risks, schedule not adding up (136 h stages vs 132 h weekly vs 130 h budget), inconsistent Kaggle cut/criterion. |
| 2026-11-06 | Kaggle check labelled the second cut-list item in Stage 9 and §5 (matching §4). Stage 2 task added to install `pvlib` and `pyarrow`. | Consistency fixes after reviewing the user's plan edit. |
| 2026-11-07 | Added a code-style rule (Resume Protocol §8, item 7; also in `CLAUDE.md`): all notebook code is very basic, lengthy code is split into separate blocks, no complex code. | User request. |
| 2026-11-15 | Irradiance outlier limit raised from 1500 to 2000 W/m² in the §5 risk table (Stage 1 outlier check). | 739 values exceed 1500 but 557 are below 1600 (plausible cloud-enhancement peaks around noon); only 5 exceed 2000. |
| 2026-11-15 | Station-set gate closed (Stage 1). Resolution counts as planned (37 for 15 min, 60 for 1 h/day-ahead), but only 31 and 51 stations have enough data in train, validation and test (thresholds 180 / 45 / 180 days). Stage 2 station-set task updated with these counts. | Stage 1 gate result: 9 stations start in 2023 or have too little training data. |
| 2026-11-16 | `UG Hall4 Flexible PV` dropped from all station sets (1-h/day-ahead: 60 → 59 stations, 51 → 50 seen). Other zero-output hours (power = 0 W at irradiance ≥ 400 W/m², `src/preprocess.py::flag_zero_fault`) are flagged, left out of training and evaluation targets, and kept for Stage 8; this includes `SQ567 Flexible PV` (9 %). Table: `results/tables/02_zero_output.csv`. | Zero-output count: `UG Hall4 Flexible PV` is zero in 1,452 of 2,268 high-irradiance hours over 310 days; all other stations ≤ 9 %, optimizer stations 0.03 % in total. |
| 2026-11-16 | Maximum gap filled (Stage 2 cleaning): 4 steps (1 h) for 15-min stations, 1 step for hourly stations; forward fill only, longer gaps stay missing, `is_filled` mask kept (`src/preprocess.py::reindex_and_fill`). On all stations: 2,554 values filled, 35,889 stay missing. | Stage 1: most gaps are whole lost daytimes, which should not be filled; a 1-h limit only bridges short dropouts. |
| 2026-11-16 | Night-to-zero rule (Stage 2 cleaning): power and energy set to 0 when sun elevation at the interval middle is below 5° (pvlib, station lat/lon, UTC+8; `src/preprocess.py::set_night_to_zero`). Removes 0.11 % (optimizer) and 0.27 % (hourly) of energy per station at the median, max 0.53 % (`SQ567 Flexible PV`); 81,775 non-zero rows set to 0, 12,827 of them with the sun below the horizon; 1,185 missing night rows become a known 0. | Stage 1 to-do item 16; the removed dawn/dusk values are small (≤ 5.4 % of rated between 0° and 5°). |
| 2026-11-16 | Weather resampling (Stage 2): 1-min weather averaged to 15-min and hourly steps, labelled at the interval start, kept only with ≥ 80 % of minutes (`src/preprocess.py::build_weather`). Rainfall is summed; wind direction uses a sine/cosine mean; irradiance outliers are masked first and the 1500-2000 W/m² flag becomes `irr_high_share`. Missing: 0.57-0.62 % per variable; rainfall 33.9 % (no 2021 data) and 0.86 % in 2022-2023. | Stage 1 to-do item 10. |
| 2026-11-16 | Weather cleaning: wind speed above 20 m/s set to missing (141 isolated 1-min spikes up to 91.9 m/s, almost all 2023-01..06; 99.99th percentile 18.9 m/s). Rainfall dropped as a feature (no 2021 data). Inverter 5-min → 15-min resampling deferred to Stage 8. | Weather checks after resampling; Stage 1 to-do item 13. |
| 2026-11-16 | Humidity dropped as a feature. | Faulty sensor periods: 2021-01..03 (stuck at about 10-15 %, 0 % around noon) and 2023-05-17..24 in the test year (jumps between about 0 and 100 %); a threshold rule cannot remove the stuck 100 % values, and masking by hand-picked dates is hard to defend. |
| 2026-11-16 | Power normalised by the `.ttl` rated power (all 60 stations have one in kW, so no fallback is needed; `src/preprocess.py::normalise_power`). `Zone A2` is used from its `.ttl` connection date 2023-05-31 only (`cut_before_capacity_change`); it moves to the unseen-station set as test-only (seen stations: 15 min 31 → 30, 1 h/day-ahead 50 → 49). | `Zone A2` output is about 0.25 x rated until 2023-05 and about 0.75 x rated afterwards, so the 249.5 kW rating fits only the expanded system; the earlier capacity is unknown. |
| 2026-11-16 | Calendar and solar-geometry features (`src/features.py`): hour, day of year, month and their sine/cosine pairs; sun elevation/azimuth at the interval middle; clear-sky GHI from the pvlib Haurwitz model (needs only the sun position). Clear-sky index = irradiance / clear-sky GHI, set to 0 below 5° sun elevation and capped at 1.5 (0.28 % of daytime hours above 1.5; median 0.51). The index uses measured irradiance, so it is a lagged feature only. | Stage 2 features; Haurwitz chosen because it needs no turbidity data. The irradiance sensor is assumed horizontal (GHI); this is *unverified*. |
| 2026-11-16 | Station metadata features (`src/features.py::station_features`): `is_optimizer`, `tilt_deg` (pair → mean, `Mixed` → NaN, filled later from train stations for linear/LSTM), `multi_orientation` flag (40 of 60 stations). No azimuth feature. | All 30 single-orientation stations are flat (tilt 0°, azimuth meaningless) and every tilted station has a pair or `Mixed` azimuth; the .ttl azimuth reference (north/south) is not stated. |
| 2026-11-16 | Lag and rolling features (`src/features.py`). Output rows are labelled with the target time T; lag 1 = row T − step (newest known row); rolling windows end at lag 1. 15 min: power lags 1, 2, 3, 4, 8, 96 + 1-h mean/std; irradiance lags 1, 2, 4 + 1-h mean; clear-sky index lags 1, 2 + 1-h mean; temperature, wind speed, pressure, visibility lag 1 (29 features + 3 station). 1 h: power lags 1, 2, 3, 24 + 3-h mean/std; irradiance lags 1, 2 + 3-h mean; clear-sky index lags 1, 2 + 3-h mean; same weather lag 1 (26 + 3). Day-ahead (issued 23:00 D−1): same-hour power on D−1 (missing for hour 23), D−2, D−7; D−1 daytime mean power and clear-sky index; D−1 mean temperature (00:00-22:00 rows); temperature, pressure, visibility at 22:00 D−1; 24-h pressure change (20 + 3). Wind direction not used; 1-h features use hourly data for all stations (`preprocess.to_hourly`). Leakage check on `SQ1` and `SQ10`: junking all measured rows from the issue time on changes no feature. | Lags and windows approved by the user. |
| 2026-11-16 | Targets (`src/features.py::add_target`): `target` = `power_norm` at T (day-ahead: the 24 hourly values of day D), with `target_is_filled` and `target_is_fault` (zero output at irradiance ≥ 400 W/m²). `usable_target` = observed, not filled, not a fault; only these rows are used for training and evaluation. Check: fault counts match `results/tables/02_zero_output.csv` (`SQ1` 27 h, `SQ567 Flexible PV` 286 h); usable share 99.8-99.9 % (`SQ1`) and 97.3 % (`SQ567 Flexible PV`). | Stage 2 targets task. |
| 2026-11-16 | Station sets final (`src/splits.py::station_sets`, `results/tables/02_station_sets.csv`): roles seen / unseen (test only, never trained on) / excluded. 15 min: 30 seen, 5 unseen, 2 excluded. 1 h and day-ahead: 49 seen, 8 unseen, 3 excluded. `SQ Block P` and `Shaw Auditorium` stay excluded (Stage 1 gate: no train data, under 180 test days). | Stage 1 gate plus the `UG Hall4 Flexible PV` and `Zone A2` decisions. |
| 2026-11-16 | Chronological split (`src/splits.py::time_split`, by target time): train ≤ 2022-09-30, validation 2022-10-01..2022-12-31, test 2023. Season check on the 49 seen 1-h stations (usable daytime hours 07-16): all four seasons in train (autumn 46,237, spring 50,454, summer 69,425, winter 40,154 rows; winter covers 47 of 49 stations). Validation has autumn and winter only, so tuning never sees spring/summer conditions (limitation for the report). | Stage 2 split task; Stage 1 to-do item 19. |
| 2026-11-16 | Unseen-station split (`src/splits.py`): 6 random hold-outs drawn once (seed 42) from the 30 seen optimizer stations, so they are unseen at every horizon: `LSK South`, `SQ3`, `UG Hall3`, `UG Hall7`, `Zone A6`, `Zone J1`. Final sets: 15 min 24 seen / 11 unseen / 2 excluded; 1 h and day-ahead 43 seen / 14 unseen / 3 excluded. Station-grouped 5-fold over the seen stations per horizon set (seed 42): 15 min 5/5/5/5/4 stations, 1 h 9/9/9/8/8. Tables: `results/tables/02_station_sets.csv`, `02_folds_15min.csv`, `02_folds_1h_dayahead.csv`. | User chose one draw for all horizons. Random hold-outs have training data, so Stage 9 can compare seen and unseen stations fairly; hourly stations are unseen only through the Stage 1 test-only group. |
| 2026-11-16 | Leakage tests (`tests/test_leakage.py`, 13 tests, run with `python -m pytest`; `pytest` added to `requirements.txt`, `pytest.ini` added). Made-up data, so no `data/` needed: junking every measured value from the issue time on changes no feature (15 min, 1 h, all 24 day-ahead hours); two control tests show a cut one step early is caught; filled, faulty and missing targets are excluded; gap fill is causal; normalisation of train rows ignores test rows; split boundaries; seen/unseen/excluded sets do not overlap; folds hold only seen stations. No train-fitted scaler exists yet; when one is added (Stage 3 linear model, LSTM), a test for it goes here. | Stage 2 leakage-test task. |
| 2026-11-16 | Vertical slice done: `python -m src.pipeline --horizon 1h` (`src/pipeline.py`) builds `data/processed/features_1h.parquet` in about 75 s: 57 stations (43 seen, 14 unseen), 1,133,152 rows, 36 columns, 98.0 % usable targets. `results/tables/02_split_summary.csv` written (rows, usable rows, stations, date range per role and split). Unseen stations keep their train/val rows in the table but carry `role = unseen`; models must train on `role == seen` only. Daytime usable seen rows: 26.1 % have a missing feature, almost all `tilt_deg` (`Mixed` stations, by design); other features ≤ 1.4 % missing. | Stage 2 vertical-slice task. |
| 2026-11-16 | Pipeline extended to all horizons: `python -m src.pipeline --horizon 15min|1h|dayahead`. 15 min: 35 stations (24 seen, 11 unseen), 2,679,552 rows, 99.4 % usable, about 110 s. Day-ahead: 57 stations (43 seen, 14 unseen), 1,133,152 rows, 98.0 % usable, about 75 s. `results/tables/02_split_summary.csv` now holds all three horizons. Daytime usable seen rows with a missing feature (without `tilt_deg`): 15 min 1.1 %, day-ahead 5.5 % (mostly `power_d7`). | Remaining Stage 2 output after the vertical slice. |
| 2026-11-16 | Stage 2 done. `notebooks/02_preprocessing_checks.ipynb` added (checks only: cleaning order, split summary, station sets, zero-output faults, clean vs gap-filled day `results/figures/02_clean_vs_filled_day.png`, missing features, seasons in training, leakage tests). `src/splits.py` and `src/pipeline.py` now use paths from the repo root, so they also run from `notebooks/`. | Stage 2 "Done when" met: one command per horizon, leakage tests pass, split table per horizon. |
| 2026-11-16 | Persistence baselines (`src/models/baselines.py`): last value (15 min, 1 h); same hour D−1 and mean of the same hour D−1..D−7 (day-ahead; D−1 hour 23 left out); smart persistence = last value × clear-sky(T) / clear-sky(T − step), falling back to the last value when clear-sky(T − step) < 50 W/m²; day-ahead smart persistence = (D−1 daytime mean power / D−1 daytime mean clear-sky) × clear-sky(T). All predictions capped to [0, 1.2] and set to 0 below 5° sun elevation. Quick check (validation, seen, daytime usable, power/rated): 1 h MAE last value 0.089 vs smart 0.050; 15 min 0.042 vs 0.033; day-ahead D−1 0.080, 7-day 0.098, smart 0.089. 3 baseline leakage tests added (16 tests in total). | Stage 3 persistence task; formal metrics follow in `src/metrics.py`. |
| 2026-11-16 | Metrics (`src/metrics.py`, 6 tests in `tests/test_metrics.py`; 22 tests in total). Errors computed on power/rated; MAE/RMSE in kW use each station's rated power; nMAE = MAE / rated capacity (%); skill = 1 − RMSE_model / RMSE_benchmark on power/rated (every station counts equally), pooled and as the mean of per-station skill; NaN if the benchmark error is 0. Model and benchmark are scored on the same rows. 95 % CIs: block bootstrap over days (1,000 resamples, seed 42) using per-day error sums. Example, 1 h validation daytime, smart persistence vs last value: nMAE 4.99 % [4.57, 5.46], skill 0.364 [0.329, 0.401], mean station skill 0.349. | Done before ridge (user agreed) so every baseline is scored with the same code. Skill on power/rated chosen so large stations do not dominate. |
| 2026-11-16 | Ridge baseline (`src/models/ridge.py`): one global model per horizon; scikit-learn pipeline = median fill → standard scaler → ridge, all fitted on train rows only (leakage test added; 23 tests in total). Fitted on seen stations, usable targets, **daytime rows only** (night target is 0 and the prediction is forced to 0). Alpha chosen on validation RMSE from 0.1 … 100,000: 15 min 0.1, 1 h 10 (both flat: RMSE changes < 0.0001), day-ahead 10,000. Validation daytime RMSE (power/rated): 15 min 0.060 (last value 0.066), 1 h 0.067 (0.116), day-ahead 0.114 (same hour D−1 0.131; MAE is worse, 0.084 vs 0.080). | Stage 3 ridge task; daytime-only fitting is a design choice. |
| 2026-11-16 | Stage 3 done. `python -m src.evaluate_baselines` (about 3 min) writes `results/tables/03_baselines.csv` (40 rows: 3 horizons × val/test × all hours/daytime × models, 95 % CIs, mean station skill, commit `e8b8666`) and `results/figures/03_baselines_nmae.png`. Seen stations, usable targets; ridge refit on train + val for test. Test 2023, daytime, nMAE % / skill: 15 min last value 5.32 / 0, smart persistence 4.69 / 0.031, ridge 5.00 / 0.080; 1 h last value 9.01 / 0, smart 5.67 / 0.277, ridge 5.80 / 0.298; day-ahead same hour D−1 10.10 / 0, 7-day mean 10.77 / 0.008, smart 9.98 / 0.093, ridge 9.80 / 0.142. Smart persistence has the lowest nMAE at 15 min and 1 h, ridge the best skill (RMSE) at every horizon. Skill is the same for all hours and daytime, because night errors are 0 for every model. | Stage 3 "Done when" met. |
| 2026-11-16 | Stage 4 vertical slice: default LightGBM (100 trees, seed 42; `src/models/gbm.py`) on the 1-h table, same fit rows as ridge, fitted in 4.6 s. Evaluated on **validation only** (test stays untouched for final models): daytime nMAE 4.07 % [3.71, 4.43], skill 0.468 [0.426, 0.510], mean station skill 0.437 (ridge 4.72 % / 0.422, smart persistence 4.99 % / 0.364). Train RMSE 0.0620 vs validation 0.0618: no sign of overfitting or leakage. Quick SHAP (3,000 validation rows, `results/figures/04_slice_shap_1h.png`): `power_lag1` dominates (0.139), then hour, `power_lag24`, sun azimuth, `csi_lag1`. `pressure_lag1` is in the top 10; it is campus-wide and may act as a weather-regime or season signal (to check in Stage 7). | Stage 4 first task; no pipeline problems found. |
| 2026-11-16 | **Irradiance sensor step found and corrected** (`src/preprocess.py::irradiance_level`, `calibrate_irradiance`). The campus irradiance sensor changes scale on about 2023-02-15: clear-day readings jump from about 0.85 x clear-sky to about 1.38 x clear-sky (factor about 1.6; hourly 99th percentile about 780 -> 1,300 W/m2) while PV output per station does not change. It also reads about 0 from late March to mid May 2021 (April 2021 maximum 17 W/m2). Fix, applied to all years: irradiance is divided by the sensor's own clear-day level = 90th percentile over the previous 30 days of the daily peak of (hourly irradiance / clear-sky) at sun elevation >= 30 deg, and multiplied by the training-period median level; it is set to missing while the level is below 0.5 (dead sensor). The rule is causal (earlier days only) and uses no power data; the level catches up about 5 days after the jump and that period stays in the evaluation. Raw values are kept as `irradiance_raw`. Feature tables rebuilt, Stage 3 re-evaluated, Stage 4 tuned again. 3 new tests (35 tests in total with the Stage 5-8 tests). | **How it was found:** the first final GBMs had 15-min test skill 0.014 (LightGBM) and -0.019 (XGBoost) against 0.15 on validation; monthly skill fell from 2023-02 on and an ablation without irradiance features did not show the fall. So the trigger was test-set evidence; this is unavoidable because the jump exists only in the test year. The first results are kept as an ablation in `results/tables/04_gbm_results_uncalibrated.csv` (test daytime skill, LightGBM / XGBoost: 15 min 0.014 / -0.019, 1 h 0.341 / 0.316, day-ahead 0.136 / 0.147). Stage 2 row counts change slightly because the zero-output rule (irradiance >= 400 W/m2) now sees the corrected scale: usable seen test rows +70 (15 min), +82 (1 h, day-ahead). Limitation for the report: the correction assumes the clear-day level of the sensor is stable over 30 days. |
| 2026-11-16 | Stage 4 done. Tuning (`src/tune_gbm.py`): 20 random settings per model and horizon, each fitted on train with early stopping on validation; lowest validation RMSE wins (`configs/gbm_*.yaml`, log in `results/tables/04_tuning_log.csv`). Final models (`src/train_gbm.py`): refit on train + validation with the tuned setting and its fixed number of trees, test scored once; models in `models/`, test predictions in `data/processed/pred_gbm_<h>.parquet`. Test 2023, daytime, seen stations, nMAE % / skill [95 % CI]: 15 min XGBoost 4.61 / 0.112 [0.099, 0.124], LightGBM 4.65 / 0.109 [0.097, 0.120] (ridge 4.98 / 0.090, smart persistence 4.69 / 0.031); 1 h LightGBM 4.95 / 0.348 [0.324, 0.370], XGBoost 4.95 / 0.346 [0.321, 0.369] (ridge 5.71 / 0.306, smart 5.67 / 0.277); day-ahead XGBoost 9.75 / 0.140 [0.114, 0.162], LightGBM 9.78 / 0.131 [0.106, 0.156] (ridge 9.72 / 0.146, smart 9.98 / 0.093). Per season and per station: `04_gbm_per_season.csv`, `04_gbm_per_station.csv`; skill is positive in every season at every horizon and for every station at 15 min and 1 h. `notebooks/04_gbm.ipynb` shows the tables. | Stage 4 "Done when" met. Findings for the report: (1) the GBMs clearly beat ridge at 15 min and 1 h; (2) day-ahead, without weather forecasts, the GBMs are **not** better than ridge (CIs overlap) and skill is small but positive, as expected for the weakest horizon; (3) two flexible-PV stations have negative day-ahead skill (`SQ567 Flexible PV` about -0.5, `UG Hall7 Flexible PV` about -0.2); (4) validation skill is higher than test skill at 15 min and 1 h because validation is one dry season. |
| 2026-11-16 | Stage 5 done (`src/models/lstm.py`, `src/train_lstm.py`, torch 2.15.0 CPU). One LSTM layer + linear head; input = the last 16 rows (15 min, 4 h) or 24 rows (1 h, 1 day) of the same feature table as the GBMs, median-filled and standardised on train rows only; windows are built per batch (lazy). Trained on seen stations, usable daytime targets, Adam 1e-3, batch 512, seed 42, early stopping on validation (patience 4). Hidden size 32 vs 64 chosen on validation: 64 at both horizons, best epoch 3 at both. Final model refit on train + validation for 3 epochs, test scored once. Test 2023, daytime, nMAE % / skill [95 % CI]: 15 min 4.86 / 0.108 [0.096, 0.119]; 1 h 5.24 / 0.332 [0.309, 0.354]. Final fit time: 421 s (15 min), 190 s (1 h), against 21-61 s for the GBMs. **Day-ahead LSTM cut** (first item on the cut list, PLAN §4). Tables `05_lstm_results.csv`, `05_lstm_training.csv`, `05_lstm_per_station.csv`; figures `05_lstm_curves.png`, `05_models_*.png`; `notebooks/05_lstm.ipynb`. | Stage 5 "Done when" met. Finding: the LSTM matches the GBMs on skill at 15 min (0.108 vs 0.109-0.112, CIs overlap) and is slightly behind at 1 h (0.332 vs 0.346-0.348, CIs overlap), with higher nMAE at both horizons (4.86 vs 4.61-4.65; 5.24 vs 4.95) and 5-10 x the training time. Validation RMSE stops improving after 2-3 epochs while train RMSE keeps falling (overfitting to the training seasons). The search-run seconds in `05_lstm_training.csv` for 15 min, hidden 64 are inflated: another job ran at the same time; the final-fit times are clean. |
| 2026-11-16 | Stage 8 done (`src/faults.py`, `notebooks/08_faults.ipynb`). **Expected-power model:** LightGBM on an hourly nowcast table (`data/processed/features_nowcast.parquet`): calendar, sun position, clear-sky irradiance, and irradiance / clear-sky index / temperature / wind of the SAME hour, station metadata and station identity (a category); no own-power input (unit test). 516 trees (early stopping on validation). Daytime RMSE (power/rated): 0.055 train and 0.048 validation (both out-of-fold), 0.077 test. **Score:** daily energy ratio actual / expected over hours with irradiance >= 200 W/m2 (at least 3 such hours), divided by the station's normal ratio, minus 1, 3-day rolling mean, minus the all-station median of the same day. **Baseline detector:** the same with the performance ratio (energy / irradiance). **Threshold:** -3 robust standard deviations (from the median absolute deviation) of the calibration scores: -0.122 (model), -0.160 (baseline). **Unmodified 2023 test data:** 10.7 % of station-days flagged by the model detector (10.4 % baseline), 207 episodes = 0.53 per station-month; but the median station is flagged on 0.35 % of days (baseline 1.10 %) and 13 of 43 stations have more than 10 % of days flagged (`SQ567 Flexible PV` 228 days, `Zone J2` 188, `UG Hall2 RF` 148). **Synthetic faults** (215 per setting: 43 stations x 5 random starts), recall model / baseline: -10 % for 7 days 0.40 / 0.32; -20 % for 3, 7, 14 days 0.76 / 0.72, 0.83 / 0.76, 0.87 / 0.85; -30 % for 7 days 0.96 / 0.97; -50 % and zero output 0.98-1.00 for both; gradual loss to -30 % over 14 days 0.90 / 0.93. Median delay 1-2 days (constant loss), 6-8 days (gradual over 14 days). Event-level precision 0.05-0.17 for both (one fault per station-year assumed; every flagged episode on unmodified data counted as false). **Lag example** (`LSK North`, -30 % for 14 days): mean score -0.315 (model), -0.317 (baseline), -0.008 for the residual of the 1-h forecast with own-power lags. Tables `08_*.csv` (detection, false alarms, threshold sensitivity, flagged periods, station normal, fleet median, lag example, model quality); figures `08_*.png`. | Stage 8 "Done when" met. **Changes from the plan, with reasons:** (1) threshold from validation quantiles replaced: validation (Oct-Dec 2022) contains real outages, so its pooled 1 % quantile was -0.69 and only total failures were caught; a robust rule on train + validation (all seasons, out-of-fold expected power) is not moved by those outages. (2) All-station daily median subtracted: all stations share one weather station, and its errors (for example April-May 2023, median score -0.17 / -0.10 before the correction) flagged every station at once. (3) Station identity added to the expected-power model: it removes fixed per-station differences; it is a label, not a measurement, so a fault cannot change it. (4) **Inverter-level comparison cut** (third item on the cut list, PLAN §4). **Findings for the report:** the model detector is better than the baseline for mild faults and has a tighter threshold, but after the all-station correction the simple baseline is close; the high flag rate on unmodified data comes from 13 stations that run below their 2021-2022 level in 2023, which may be real degradation or soiling but cannot be confirmed without labels, so the precision figures are a lower limit. |
| 2026-11-16 | Stage 7 done (`src/explain.py`, `notebooks/07_shap.ipynb`). TreeSHAP values from the model libraries (`pred_contrib`; unit test: base value + SHAP values = prediction) for the GBM with the lower validation RMSE per horizon (XGBoost at all three), on 20,000 random daytime test rows of seen stations. **Global importance (share of total mean \|SHAP\|):** 15 min `power_lag1` 54 %, `power_mean4` 11 %, `hour` 6 %, `irr_lag1` 5 %; 1 h `power_lag1` 48 %, `hour` 15 %, `power_lag24` 8 %, sun azimuth and elevation 5 % each; day-ahead `power_d1` 28 %, sun elevation 17 %, `power_d2` 14 %, `power_d1_daymean` 10 %. **Grouped:** own-power lags 75 % (15 min), 61 % (1 h), 53 % (day-ahead); measured irradiance / clear-sky index 10 %, 7 %, 1 %; calendar + solar geometry 12 %, 26 %, 33 %; other weather 3 %, 5 %, 12 %; station metadata under 1.5 %. **Stability (mean / minimum Spearman rank correlation of the importance vectors):** across stations 0.993 / 0.964 (15 min), 0.988 / 0.953 (1 h), 0.981 / 0.918 (day-ahead); across seasons 0.974 / 0.962, 0.981 / 0.968, 0.953 / 0.912; across 5 models retrained on bootstrap resamples with different seeds 0.983 / 0.969, 0.990 / 0.983, 0.966 / 0.942; top-5 overlap 0.92-1.00. **Physical checks** (`07_physical_checks.csv`): lagged power, irradiance, clear-sky index, clear-sky irradiance and sun elevation push the forecast up as expected; temperature does **not** show the expected negative effect at 15 min and 1 h (correlation between `temperature_lag1` and its SHAP value +0.70 and +0.23) and is about neutral day-ahead (-0.04); its importance is small (about 0.0015). `clearsky_ghi` is not used by the 1-h model (sun elevation and azimuth carry the same information). | Stage 7 "Done when" met. Findings for the report: (1) the explanations are stable, so the ranking can be trusted as a description of the model; (2) as the plan expected, lag-1 power dominates the short horizons, so grouped importance is reported too; (3) the hypothesis "temperature has a negative effect" is rejected for these models: warm hours are sunny hours, and with a single campus weather station the model cannot separate panel heating from sunshine; (4) the importance of calendar and solar-geometry features grows with the horizon as recent measurements lose value. |
