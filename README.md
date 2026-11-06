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
Primary: HKUST rooftop PV dataset (rooftop PV stations in Hong Kong, PV power plus on-site weather/microclimate).

> Lin et al., *Scientific Data* 2025, doi:[10.1038/s41597-025-04397-y](https://doi.org/10.1038/s41597-025-04397-y)

Secondary (optional sanity check only): Kaggle "Solar Power Generation Data" (2 plants in India, ~34 days, 15-min).
The earlier EDA notebook (`notebooks/01_eda.ipynb`) targets this dataset.

The data is **not tracked in git** (~5.8 GB). Put it under `data/raw/`:
`raw_data/` is the source of truth; `processed_data/` is the dataset authors' reference version.
Details of what the local files actually contain are tracked in `WORKLOG.md` and will be formalised in the data-understanding step.

## Setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Layout
| Path | Purpose |
|---|---|
| `data/raw`, `data/processed` | source data / our cleaned data (git-ignored) |
| `notebooks/` | analysis notebooks, numbered in run order |
| `src/` | reusable code |
| `models/` | saved models (git-ignored) |
| `results/figures`, `results/tables` | figures and tables for the report |
| `app/` | Streamlit dashboard (later) |
| `report/` | written report |
| `WORKLOG.md` | dated log of work and hours |
