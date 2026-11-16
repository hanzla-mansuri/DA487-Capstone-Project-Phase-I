# Explainable and Efficient Machine Learning for Multi-Horizon Solar Power Forecasting

B.Sc. Data Science capstone (DA487, 12 credits).

**Objective:** compare persistence/linear, gradient-boosting and LSTM models for multi-horizon rooftop PV forecasting on the best accuracy-efficiency-explainability trade-off. The full stage plan and status are in [`PLAN.md`](PLAN.md); milestones of the autonomous run are in [`PROGRESS.md`](PROGRESS.md) and its technical choices in [`DECISIONS.md`](DECISIONS.md).

## Scope
- **Horizons:** 15 min (stations with 5/15-min data), 1 hour and day-ahead (all stations).
- **Models:** persistence / linear → XGBoost / LightGBM → LSTM.
- **Efficiency benchmark:** training time, inference latency, model size, quantization/ONNX.
- **Explainability:** SHAP, with stability across stations and seasons.
- **Fault flagging:** residual-based underperformance detection.
- **Generalisation:** unseen-station test; Kaggle out-of-time sanity check.
- **Deliverable:** Streamlit dashboard (in `app/`).

## Dataset
Primary: HKUST rooftop PV dataset (Dryad, doi:10.5061/dryad.m37pvmd99): 60 rooftop PV stations in Hong Kong, 2021-2023, plus campus weather.

> Lin et al., *Scientific Data* 2025, doi:[10.1038/s41597-025-04397-y](https://doi.org/10.1038/s41597-025-04397-y)

Secondary (optional sanity check only): Kaggle "HK-Rooftop-Microclimate-PV" (HKUST, 2024-2025). The earlier Kaggle India dataset was dropped; its notebook is in `archive/`.

Full source links, citations and licence status: see [`DATA_SOURCES.md`](DATA_SOURCES.md).
Place the primary data in `data/hkust/` and the secondary data in `data/raw/`. Neither is tracked in git (~6.5 GB).

## Setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0   # CPU build of PyTorch
python -m pytest                                                             # leakage, metric and model tests
```
Developed with Python 3.14 on Linux (WSL2), CPU only.

## Reproduce
Obtain both datasets as described in `DATA_SOURCES.md` and place them in `data/hkust/` and `data/raw/`.
Then run the stages in order from the repo root. Run them one at a time: each uses all CPU cores.

| Stage | Command | Writes |
|---|---|---|
| 1 EDA | `notebooks/01_eda.ipynb` | `results/tables/01_*`, `results/figures/01_*` |
| 2 Features | `python -m src.pipeline --horizon 15min` (also `1h`, `dayahead`) | `data/processed/features_<h>.parquet` |
| 3 Baselines | `python -m src.evaluate_baselines` | `03_baselines.csv` |
| 4 GBM tuning | `python -m src.tune_gbm --horizon 15min --trials 20` (also `1h`, `dayahead`) | `configs/gbm_*.yaml`, `04_tuning_log.csv` |
| 4 Final GBMs | `python -m src.train_gbm` | `models/gbm_*`, `04_gbm_*.csv` |
| 5 LSTM | `python -m src.train_lstm` | `models/lstm_*.pt`, `05_lstm_*.csv` |
| 6 Efficiency | `python -m src.benchmark` | `06_efficiency.csv` |
| 7 SHAP | `python -m src.explain` | `07_*.csv` |
| 8 Faults | `python -m src.faults --build` | `08_*.csv` |
| 9 Generalisation | `python -m src.generalise` | `09_*.csv` |
| 10 Dashboard | `python -m app.build_demo` then `streamlit run app/streamlit_app.py` | `app/demo/` (git-ignored) |

Tables go to `results/tables/`, figures to `results/figures/`. The notebooks `notebooks/04_*.ipynb` to
`09_*.ipynb` only display these results, with the reasoning behind each stage.

On a 2-core laptop CPU the whole chain takes a few hours; tuning (stage 4) is the longest step.

## Dashboard
```bash
python -m app.build_demo            # once, after stages 2-9: builds app/demo/ from your local data
streamlit run app/streamlit_app.py
```
Pages: results overview, forecast vs actual, SHAP explanation of a single forecast, fault flags with a
what-if fault injector, efficiency, and generalisation. No data or model is stored in the repository
(dataset licences are unverified, see `DATA_SOURCES.md`), so the dashboard needs the bundle built locally.

## Layout
| Path | Purpose |
|---|---|
| `data/hkust`, `data/raw`, `data/processed` | primary data / secondary data / our cleaned data (git-ignored) |
| `notebooks/` | analysis notebooks, numbered in run order |
| `src/` | reusable code |
| `models/` | saved models (git-ignored) |
| `results/figures`, `results/tables` | figures and tables for the report |
| `app/` | Streamlit dashboard and its demo-bundle script |
| `report/` | written report |
| `tests/`, `configs/` | leakage, metric and model tests; tuned model configs |
| `PLAN.md` | stage plan, schedule, protocol |
| `archive/` | old Kaggle-India scope (kept for reference) |
| `WORKLOG.md` | dated log of work and hours |
| `DATA_SOURCES.md` | dataset sources, citations, licences |
