# Explainable and Efficient Machine Learning for Short-Term Solar Power Forecasting

B.Sc. Data Science capstone (DA487), Phase I.

**Plan:** baselines (persistence, linear) → XGBoost / LightGBM → LSTM → efficiency benchmark
(train time, inference latency, model size) → SHAP explainability → residual-based underperformance flagging.

**Current status:** repo scaffold + exploratory data analysis (`notebooks/01_eda.ipynb`). No models yet.

## Data
Kaggle ["Solar Power Generation Data"](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data):
two plants in India, ~34 days, 15-minute intervals. Place the four CSVs in `data/raw/`
(file names must contain `Plant_1` / `Plant_2` and `Generation` / `Weather`):

```
data/raw/Plant_1_Generation_Data.csv      data/raw/Plant_1_Weather_Sensor_Data.csv
data/raw/Plant_2_Generation_Data.csv      data/raw/Plant_2_Weather_Sensor_Data.csv
```
`data/` and `models/` are git-ignored (only `.gitkeep` placeholders are tracked).

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab            # or: jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```
Run notebooks from inside `notebooks/` (paths are relative to it).

## Layout
| Path | Purpose |
|---|---|
| `data/raw`, `data/processed` | original CSVs / cleaned data (git-ignored) |
| `notebooks/` | analysis notebooks, numbered in run order |
| `src/` | reusable code (preprocessing, features, models) |
| `models/` | saved models (git-ignored) |
| `results/figures`, `results/tables` | all figures and tables used in the report |
| `app/` | demo app (later) |
| `report/` | written report |

## Notebooks
1. `01_eda.ipynb` — data-quality audit: shapes/dtypes/missing/duplicates, datetime format verification,
   timestamp gaps and generation↔weather alignment, DC/AC scale check, physical sanity checks,
   exploratory plots, per-inverter comparison, and a preprocessing to-do list.
