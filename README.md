# Explainable and Efficient Machine Learning for Multi-Horizon Solar Power Forecasting

B.Sc. Data Science capstone (DA487, 12 credits).

## Scope
- **Horizons:** 15 min, 1 hour, day-ahead.
- **Models:** persistence / linear → XGBoost / LightGBM → LSTM.
- **Efficiency benchmark:** training time, inference latency, model size, quantization.
- **Explainability:** SHAP.
- **Fault flagging:** residual-based underperformance detection.
- **Generalisation:** testing on unseen stations.
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
jupyter lab
```
To reproduce: obtain the data as described in `DATA_SOURCES.md`, then run the numbered notebooks in `notebooks/` in order (none yet; the stage plan is in `PLAN.md` once approved).

## Layout
| Path | Purpose |
|---|---|
| `data/hkust`, `data/raw`, `data/processed` | primary data / secondary data / our cleaned data (git-ignored) |
| `notebooks/` | analysis notebooks, numbered in run order |
| `src/` | reusable code |
| `models/` | saved models (git-ignored) |
| `results/figures`, `results/tables` | figures and tables for the report |
| `app/` | Streamlit dashboard (later) |
| `report/` | written report |
| `archive/` | old Kaggle-India scope (kept for reference) |
| `WORKLOG.md` | dated log of work and hours |
| `DATA_SOURCES.md` | dataset sources, citations, licences |
