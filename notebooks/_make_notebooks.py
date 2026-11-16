"""Write the Stage 4-9 notebooks (basic cells only). Run from the repo root."""
import sys
import nbformat as nbf

SETUP = """import pandas as pd
from IPython.display import Image

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
TABLES = "../results/tables/"
FIGURES = "../results/figures/"
"""


def notebook(name, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    for kind, text in cells:
        text = text.strip("\n")
        nb.cells.append(nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text))
    nbf.write(nb, f"notebooks/{name}.ipynb")
    print("wrote", name)


which = sys.argv[1:] or ["04", "05", "06", "07", "08", "09"]

if "04" in which:
    notebook("04_gbm", [
        ("md", """# Stage 4: gradient-boosting models (LightGBM and XGBoost)

This notebook only **shows** results. The work is done by two scripts:

- `python -m src.tune_gbm --horizon <h> --trials 20` : random search, validation only;
- `python -m src.train_gbm` : final refit on train + validation, one test run, saved models.

**Why tree ensembles?** The features are a table (lags, weather, sun position, calendar). Trees handle
missing values, need no scaling, and learn interactions (for example: irradiance matters differently
at noon and at sunset) without us writing them by hand.

**How was the test set kept clean?** Every setting is fitted on *train* and scored on *validation*
(early stopping also on validation). The best setting and its number of trees are fixed. Only then is
the model refit on train + validation and the 2023 test year scored **once**."""),
        ("code", SETUP),
        ("md", "## 1. Tuning log\nEvery setting that was tried. The lowest validation RMSE per model and horizon wins."),
        ("code", """log = pd.read_csv(TABLES + "04_tuning_log.csv")
log.groupby(["horizon", "model"])["val_rmse"].agg(["count", "min", "median", "max"]).round(5)"""),
        ("code", """best = log.loc[log.groupby(["horizon", "model"])["val_rmse"].idxmin()]
best[["horizon", "model", "trial", "val_rmse", "n_trees", "params"]]"""),
        ("md", "## 2. Test results next to the baselines\nDaytime, seen stations, 2023. `skill` = 1 - RMSE / RMSE of the benchmark persistence."),
        ("code", """baselines = pd.read_csv(TABLES + "03_baselines.csv")
gbm = pd.read_csv(TABLES + "04_gbm_results.csv")
both = pd.concat([baselines, gbm])"""),
        ("code", """test = both[(both["split"] == "test") & (both["hours"] == "daytime")]
columns = ["horizon", "model", "nmae_pct", "nmae_pct_low", "nmae_pct_high", "skill", "skill_low", "skill_high", "mean_station_skill"]
test[columns].sort_values(["horizon", "nmae_pct"]).round(3)"""),
        ("code", 'Image(FIGURES + "04_gbm_skill.png")'),
        ("code", 'Image(FIGURES + "04_gbm_nmae.png")'),
        ("md", "## 3. All hours\nNight targets are 0 and night forecasts are 0 by rule, so all-hours errors are smaller. Skill is almost the same."),
        ("code", """all_hours = both[(both["split"] == "test") & (both["hours"] == "all")]
all_hours[["horizon", "model", "nmae_pct", "skill"]].sort_values(["horizon", "nmae_pct"]).round(3)"""),
        ("md", "## 4. Validation vs test\nValidation is one dry season (Oct-Dec 2022); the test year has all seasons, so the errors differ."),
        ("code", """day = gbm[gbm["hours"] == "daytime"]
day.pivot(index=["horizon", "model"], columns="split", values=["nmae_pct", "skill"]).round(3)"""),
        ("md", "## 5. Per season (test, daytime)"),
        ("code", """season = pd.read_csv(TABLES + "04_gbm_per_season.csv")
season.pivot(index=["horizon", "model"], columns="season", values="nmae_pct").round(2)"""),
        ("code", 'season.pivot(index=["horizon", "model"], columns="season", values="skill").round(3)'),
        ("md", "## 6. Per station (test, daytime)\nSpread of the skill over the stations: is the model better than persistence everywhere?"),
        ("code", """station = pd.read_csv(TABLES + "04_gbm_per_station.csv")
station.groupby(["horizon", "model"])["skill"].describe().round(3)"""),
        ("code", """worst = station.sort_values("skill").groupby(["horizon", "model"]).head(3)
worst[["horizon", "model", "station", "nmae_pct", "skill"]].round(3)"""),
    ])

if "05" in which:
    notebook("05_lstm", [
        ("md", """# Stage 5: LSTM

Results of `python -m src.train_lstm` (15-min and 1-h horizons; the day-ahead LSTM is a stretch goal
and was cut, see the `PLAN.md` Change log).

**Input:** the last rows of the same feature table the GBMs use (16 rows = 4 h for 15 min, 24 rows = 1 day
for 1 h), as a sequence. Every value in the window is known when the forecast is issued.

**Protocol:** the same as for the GBMs. Two hidden sizes are tried on train with early stopping on
validation; the better one is refit on train + validation for the same number of epochs; the test year
is scored once."""),
        ("code", SETUP),
        ("md", "## 1. Training curves"),
        ("code", 'Image(FIGURES + "05_lstm_curves.png")'),
        ("code", """training = pd.read_csv(TABLES + "05_lstm_training.csv")
search = training[training["run"].str.startswith("train")]
search.groupby(["horizon", "hidden"])["val_rmse"].agg(["min", "count"]).round(5)"""),
        ("md", "## 2. LSTM next to every other model (test, daytime)"),
        ("code", """names = ["03_baselines.csv", "04_gbm_results.csv", "05_lstm_results.csv"]
results = pd.concat([pd.read_csv(TABLES + n) for n in names])
test = results[(results["split"] == "test") & (results["hours"] == "daytime")]
test = test[test["horizon"] != "dayahead"]"""),
        ("code", """columns = ["horizon", "model", "nmae_pct", "nmae_pct_low", "nmae_pct_high", "skill", "skill_low", "skill_high"]
test[columns].sort_values(["horizon", "nmae_pct"]).round(3)"""),
        ("code", 'Image(FIGURES + "05_models_skill.png")'),
        ("md", "## 3. Is the deep model worth its cost?\nTraining time of the final fit (seconds, this laptop CPU). The latency and size comparison is in Stage 6."),
        ("code", """lstm = pd.read_csv(TABLES + "05_lstm_results.csv")
gbm = pd.read_csv(TABLES + "04_gbm_results.csv")
cost = pd.concat([lstm, gbm])
cost = cost[(cost["split"] == "test") & (cost["hours"] == "daytime") & (cost["horizon"] != "dayahead")]
cost[["horizon", "model", "fit_seconds", "nmae_pct", "skill"]].sort_values(["horizon", "nmae_pct"]).round(3)"""),
    ])

if "06" in which:
    notebook("06_efficiency", [
        ("md", """# Stage 6: efficiency benchmark

Results of `python -m src.benchmark`. Everything is measured on the same machine (specs below).

- **train time**: final fit on train + validation;
- **model-only latency**: one sample, and per sample in a batch of 1,000 (median, after warm-up);
- **end-to-end latency**: building the features for one forecast + the model;
- **size**: the model file on disk;
- **exports**: ONNX (float32) for the GBMs and the LSTM, int8 dynamic quantization for the LSTM,
  each scored again on the test set."""),
        ("code", SETUP),
        ("code", """eff = pd.read_csv(TABLES + "06_efficiency.csv")
eff[["cpu", "cores", "ram_gb", "os", "python"]].drop_duplicates()"""),
        ("md", "## 1. Cost and accuracy of every model"),
        ("code", """columns = ["horizon", "model", "train_seconds", "single_ms", "batch_ms_per_sample", "end_to_end_single_ms", "size_kB", "test_nmae_pct"]
eff[columns].round(4)"""),
        ("md", "## 2. Feature construction dominates one forecast\n`feature_ms` is the same for every model of a horizon."),
        ("code", 'eff.groupby("horizon")["feature_ms"].first().round(1)'),
        ("md", "## 3. Exports: does accuracy change?\nA change of 0 means the exported model gives the same error as the original."),
        ("code", """exports = eff[eff["model"].str.contains("onnx|int8")]
exports[["horizon", "model", "size_kB", "single_ms", "test_nmae_pct", "nmae_change_vs_original_pct_points", "note"]].round(4)"""),
        ("md", "## 4. Accuracy vs latency"),
        ("code", 'Image(FIGURES + "06_accuracy_vs_latency.png")'),
    ])

if "07" in which:
    notebook("07_shap", [
        ("md", """# Stage 7: SHAP explainability

Results of `python -m src.explain`, for the best GBM of each horizon.

**What is a SHAP value?** For one forecast, it is the share of the forecast that one feature is
responsible for. Base value + all SHAP values = the forecast. The mean absolute SHAP value of a feature
over many rows is its global importance. Units: power / rated.

**Can we trust the explanations?** That is what the stability section tests: if the importance ranking
changed a lot between stations, seasons or retrained models, it would say little about the problem."""),
        ("code", SETUP),
        ("md", "## 1. Global importance"),
        ("code", 'Image(FIGURES + "07_shap_global.png")'),
        ("code", """imp = pd.read_csv(TABLES + "07_shap_importance.csv")
imp[imp["rank"] <= 8][["horizon", "model", "rank", "feature", "group", "share_pct"]].round(2)"""),
        ("md", """## 2. Grouped importance
Lagged power and lagged irradiance are strongly correlated, so SHAP splits their shared credit between
them. Summing the importance per feature group is more robust than reading single features."""),
        ("code", 'Image(FIGURES + "07_shap_groups.png")'),
        ("code", """groups = pd.read_csv(TABLES + "07_shap_groups.csv")
groups.pivot(index="group", columns="horizon", values="share_pct").round(1)"""),
        ("md", """## 3. Stability
Spearman rank correlation between importance vectors (1 = the same ranking). `top5_overlap` is the mean
share of the top-5 features that two vectors have in common."""),
        ("code", 'pd.read_csv(TABLES + "07_shap_stability.csv").round(3)'),
        ("md", """## 4. Physical expectations (hypotheses, not conclusions)
For each feature: does a higher value push the forecast in the expected direction? The correlation
between the feature value and its SHAP value gives the direction."""),
        ("code", 'pd.read_csv(TABLES + "07_physical_checks.csv").round(3)'),
        ("md", "## 5. Dependence plots (1 h)"),
        ("code", 'Image(FIGURES + "07_shap_dependence_1h.png")'),
        ("md", "## 6. Local explanations (1 h)\nThree single forecasts: a clear hour, a cloudy midday hour and a ramp."),
        ("code", 'Image(FIGURES + "07_shap_local_1h.png")'),
    ])

if "08" in which:
    notebook("08_faults", [
        ("md", """# Stage 8: residual-based fault / underperformance flagging

Results of `python -m src.faults`.

**Design.** A forecast that uses the station's own lagged power adapts to a lasting fault and hides it.
So the detector uses an **expected-power model without own-power inputs** (weather of the same hour, sun
position, calendar, station identity and metadata).

**Score per day and station:**
1. ratio = actual energy / expected energy over the bright hours (irradiance >= 200 W/m2);
2. divide by the station's normal ratio (median in the calibration period) and subtract 1;
3. 3-day rolling mean;
4. subtract the median of all stations on the same day (a weather-sensor error moves every station,
   a fault moves one).

A day is flagged when the score is below **-3 robust standard deviations** of the calibration scores.
Calibration = train + validation (all seasons), with out-of-fold expected power.

**Baseline detector:** the same recipe with a plain performance ratio (energy / irradiance).

**No fault labels exist**, so the detectors are scored on synthetic faults injected into the 2023 test
data, and the flags on unmodified data give an upper limit of the false-alarm rate."""),
        ("code", SETUP),
        ("md", "## 1. The expected-power model\nError of the expected power (no own-power inputs). Train and validation are out-of-fold."),
        ("code", 'pd.read_csv(TABLES + "08_expected_power_model.csv")'),
        ("md", "## 2. Thresholds and flags on unmodified test data\nMost stations are almost never flagged; a few are flagged for a large part of 2023."),
        ("code", 'pd.read_csv(TABLES + "08_false_alarms.csv").round(3).T'),
        ("code", 'Image(FIGURES + "08_score_spread.png")'),
        ("md", "## 3. Synthetic faults: recall, precision and delay\nPrecision assumes one fault per station-year and counts every flagged episode on unmodified data as a false alarm."),
        ("code", """det = pd.read_csv(TABLES + "08_fault_detection.csv")
det.pivot(index=["kind", "severity", "duration_days"], columns="detector", values="recall").round(3)"""),
        ("code", 'det.pivot(index=["kind", "severity", "duration_days"], columns="detector", values="precision").round(3)'),
        ("code", 'det.pivot(index=["kind", "severity", "duration_days"], columns="detector", values="median_delay_days")'),
        ("code", 'Image(FIGURES + "08_recall_by_severity.png")'),
        ("md", "## 4. How strict should the threshold be?\nA stricter threshold (more sigmas) flags fewer normal days but misses more mild faults."),
        ("code", 'pd.read_csv(TABLES + "08_threshold_sensitivity.csv").round(3)'),
        ("md", "## 5. Why not use the forecast residual?\nOne injected fault seen by three residuals. The 1-h forecast uses lagged power, which drops with the fault, so its residual stays near 0."),
        ("code", 'Image(FIGURES + "08_lag_vs_nowcast.png")'),
        ("code", 'pd.read_csv(TABLES + "08_lag_example.csv").round(3).T'),
        ("md", "## 6. Real flagged periods (candidates, not ground truth)\n`has_known_zero_output_hours` = the period overlaps hours flagged in Stage 2 as zero output under strong sun."),
        ("code", """real = pd.read_csv(TABLES + "08_flagged_periods.csv")
len(real), real["has_known_zero_output_hours"].sum()"""),
        ("code", "real.head(15)"),
        ("code", 'real.groupby("station")["days"].agg(["count", "sum"]).sort_values("sum", ascending=False).head(13)'),
    ])

if "09" in which:
    notebook("09_generalisation", [
        ("md", """# Stage 9: generalisation

Results of `python -m src.generalise`.

1. **Unseen stations**: the final models on stations they never trained on (6 random hold-outs with a
   full 2023, and stations that are new in 2023), next to the seen stations.
2. **Station-grouped 5-fold** for the best GBM: the same stations with and without them in training.
3. **Kaggle transfer check** (1 h): a reduced model with common features only, applied unchanged to a
   different rooftop dataset from 2024-2025."""),
        ("code", SETUP),
        ("md", "## 1. Seen vs unseen stations (test 2023, daytime)"),
        ("code", """gen = pd.read_csv(TABLES + "09_generalisation.csv")
columns = ["horizon", "model", "group", "stations", "nmae_pct", "nmae_pct_low", "nmae_pct_high", "skill", "mean_station_skill"]
gen[columns].round(3)"""),
        ("code", 'Image(FIGURES + "09_seen_vs_unseen.png")'),
        ("md", "The stations that are new in 2023 cover only part of the year, so their numbers are not for the same period as the seen stations. The random hold-outs are."),
        ("md", "## 2. Station-grouped 5-fold"),
        ("code", """kfold = pd.read_csv(TABLES + "09_kfold.csv")
kfold.drop(columns="commit").round(3)"""),
        ("code", 'kfold.groupby("horizon")[["nmae_gap_pct_points", "nmae_gap_relative_pct"]].agg(["mean", "std", "min", "max"]).round(3)'),
        ("code", 'Image(FIGURES + "09_kfold.png")'),
        ("md", """## 3. Kaggle transfer sanity check
Limits: the Kaggle data is from 2024-2025 (no overlap), panel-optimizer level, with its own weather
station; its capacity is not in the files, so a proxy is used; its timestamp convention is not stated.
Time, hardware and weather source change together, so this is **not** an out-of-time benchmark."""),
        ("code", """kaggle = pd.read_csv(TABLES + "09_kaggle_transfer.csv")
kaggle[["data", "model", "n_rows", "n_days", "nmae_pct", "skill", "skill_low", "skill_high", "capacity_proxy_kW"]].round(3)"""),
        ("code", 'Image(FIGURES + "09_kaggle_transfer.png")'),
    ])
