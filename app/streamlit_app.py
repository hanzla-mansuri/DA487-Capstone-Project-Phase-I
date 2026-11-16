"""Streamlit dashboard (Stage 10).

    python -m app.build_demo          # once: build the demo bundle
    streamlit run app/streamlit_app.py

Pages: overview of the results, forecast vs actual, SHAP explanation of one
forecast, fault flags (with a what-if fault injector), efficiency, and
generalisation. The app reads only `app/demo/` (the bundle) and the result
tables and figures in `results/`.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import altair as alt
import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from src import faults

DEMO = REPO / "app" / "demo"
TABLES = REPO / "results" / "tables"
FIGURES = REPO / "results" / "figures"

HORIZON_NAMES = {"15min": "15 minutes ahead", "1h": "1 hour ahead", "dayahead": "Day ahead"}
MODEL_NAMES = {"benchmark": "Persistence (benchmark)", "lgbm": "LightGBM", "xgb": "XGBoost",
               "lstm": "LSTM"}
NOT_FEATURES = ["target", "target_is_filled", "target_is_fault", "usable", "station", "role",
                "split", "benchmark", "lgbm", "xgb", "lstm"]
COLORS = {"Actual": "#0b0b0b", "Persistence (benchmark)": "#9a9892", "LightGBM": "#2a78d6",
          "XGBoost": "#d6602a", "LSTM": "#3aa37a"}

st.set_page_config(page_title="Solar forecasting dashboard", page_icon="☀️", layout="wide")


# ------------------------------------------------------------ loading

def require_bundle():
    """Stop with a clear message when the demo bundle has not been built."""
    if not (DEMO / "manifest.json").exists():
        st.error("The demo bundle is missing. Data and models are not stored in the "
                 "repository, so build the bundle once from the project root:")
        st.code("python -m app.build_demo", language="bash")
        st.caption("This needs the outputs of the earlier stages; see the README, section Reproduce.")
        st.stop()


@st.cache_data(show_spinner=False)
def load_manifest():
    return json.loads((DEMO / "manifest.json").read_text())


@st.cache_data(show_spinner="Loading forecasts...")
def load_features(horizon):
    return pd.read_parquet(DEMO / f"features_{horizon}.parquet")


@st.cache_data(show_spinner=False)
def load_stations():
    return pd.read_csv(DEMO / "stations.csv", index_col="station")


@st.cache_data(show_spinner="Loading fault data...")
def load_daily():
    return pd.read_parquet(DEMO / "fault_daily.parquet")


@st.cache_data(show_spinner=False)
def load_table(name):
    """A results table, or None when the stage has not been run yet."""
    path = TABLES / name
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource(show_spinner="Loading model...")
def load_booster(model, horizon):
    if model == "lgbm":
        return lgb.Booster(model_file=str(DEMO / f"gbm_lgbm_{horizon}.txt"))
    booster = xgb.Booster()
    booster.load_model(str(DEMO / f"gbm_xgb_{horizon}.json"))
    return booster


def show_figure(name, caption=None):
    path = FIGURES / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Figure `{name}` has not been produced yet.")


def missing_table(name):
    st.warning(f"`results/tables/{name}` was not found. Run the stage that writes it "
               "(see the README, section Reproduce).")


# ------------------------------------------------------------ pages

def page_overview():
    st.title("Multi-horizon solar power forecasting")
    st.write("Rooftop PV power of HKUST stations, forecast 15 minutes, 1 hour and one day ahead. "
             "All numbers are from the **2023 test year**, daytime hours, stations the models "
             "were trained on. Lower nMAE is better; higher skill is better.")

    parts = [load_table(n) for n in ["03_baselines.csv", "04_gbm_results.csv", "05_lstm_results.csv"]]
    parts = [p for p in parts if p is not None]
    if not parts:
        missing_table("03_baselines.csv")
        return
    results = pd.concat(parts)
    results = results[(results["split"] == "test") & (results["hours"] == "daytime")]

    horizon = st.session_state["horizon"]
    d = results[results["horizon"] == horizon].sort_values("nmae_pct")
    best = d.iloc[0]
    left, middle, right = st.columns(3)
    left.metric("Best model", best["model"])
    middle.metric("nMAE (% of rated power)", f"{best['nmae_pct']:.2f}")
    right.metric("Skill vs persistence", f"{best['skill']:.3f}")

    table = pd.DataFrame({
        "Model": d["model"],
        "nMAE (%)": d["nmae_pct"].round(2),
        "nMAE 95 % CI": [f"{a:.2f} to {b:.2f}" for a, b in zip(d["nmae_pct_low"], d["nmae_pct_high"])],
        "RMSE (kW)": d["rmse_kW"].round(2),
        "Skill": d["skill"].round(3),
        "Skill 95 % CI": [f"{a:.3f} to {b:.3f}" for a, b in zip(d["skill_low"], d["skill_high"])],
    })
    st.subheader(f"All models, {HORIZON_NAMES[horizon].lower()}")
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(f"Skill = 1 - RMSE / RMSE of `{best['benchmark']}`. Intervals: block bootstrap over days.")

    chart = alt.Chart(d).mark_bar(color="#2a78d6").encode(
        x=alt.X("skill:Q", title="Skill vs benchmark persistence"),
        y=alt.Y("model:N", sort="-x", title=None),
        tooltip=["model", alt.Tooltip("skill:Q", format=".3f"), alt.Tooltip("nmae_pct:Q", format=".2f")])
    st.altair_chart(chart, width="stretch")


def window_metrics(data, models, rated_kW):
    """nMAE, RMSE and skill of each model on the usable daytime rows shown."""
    ok = data["usable"] & (data["sun_elevation"] >= 5) & data["benchmark"].notna()
    rows = []
    for model in models:
        use = ok & data[model].notna()
        if use.sum() == 0:
            continue
        err = data.loc[use, model] - data.loc[use, "target"]
        bench_err = data.loc[use, "benchmark"] - data.loc[use, "target"]
        bench_rmse = np.sqrt((bench_err ** 2).mean())
        rmse = np.sqrt((err ** 2).mean())
        rows.append({"Model": MODEL_NAMES[model], "nMAE (%)": 100 * err.abs().mean(),
                     "RMSE (kW)": rmse * rated_kW,
                     "Skill": 1 - rmse / bench_rmse if bench_rmse > 0 else np.nan,
                     "Hours scored": int(use.sum())})
    return pd.DataFrame(rows)


def page_forecast():
    horizon, station = st.session_state["horizon"], st.session_state["station"]
    st.title("Forecast vs actual")
    data = load_features(horizon)
    data = data[data["station"] == station]
    stations = load_stations()
    rated_kW = float(stations.loc[station, "rated_kW"])
    if stations.loc[station, "role"] == "unseen":
        st.info("This station is a random hold-out: the models never trained on it.")

    available = [m for m in ["lgbm", "xgb", "lstm", "benchmark"] if m in data.columns]
    left, right = st.columns([2, 3])
    dates = left.date_input("Dates (2023)", value=(pd.Timestamp("2023-07-01"), pd.Timestamp("2023-07-07")),
                            min_value=pd.Timestamp("2023-01-01"), max_value=pd.Timestamp("2023-12-31"))
    models = right.multiselect("Models", available, default=available[:2],
                               format_func=lambda m: MODEL_NAMES[m])
    in_kW = st.toggle("Show kW (off: share of rated power)", value=True)

    if not isinstance(dates, (tuple, list)) or len(dates) != 2:
        st.info("Pick a start date and an end date.")
        return
    if not models:
        st.info("Pick at least one model.")
        return
    start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[1]) + pd.Timedelta(days=1)
    if (end - start).days > 62:
        st.warning("Showing at most 62 days; the end date was moved.")
        end = start + pd.Timedelta(days=62)
    view = data[(data.index >= start) & (data.index < end)]
    if view.empty or view["target"].notna().sum() == 0:
        st.warning("No measurements for this station in the chosen dates.")
        return

    scale = rated_kW if in_kW else 1.0
    lines = {"Actual": view["target"] * scale}
    for model in models:
        lines[MODEL_NAMES[model]] = view[model] * scale
    long = pd.DataFrame(lines).rename_axis("time").reset_index().melt(
        "time", var_name="Series", value_name="Power")
    names = list(lines)
    chart = alt.Chart(long).mark_line(strokeWidth=1.6).encode(
        x=alt.X("time:T", title=None),
        y=alt.Y("Power:Q", title="Power (kW)" if in_kW else "Power / rated"),
        color=alt.Color("Series:N", scale=alt.Scale(domain=names, range=[COLORS[n] for n in names]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["time:T", "Series:N", alt.Tooltip("Power:Q", format=".2f")]).interactive(bind_y=False)
    st.altair_chart(chart, width="stretch")

    scores = window_metrics(view, models, rated_kW)
    if scores.empty:
        st.info("No daytime measurements to score in this window.")
        return
    st.subheader("Error in the chosen window (daytime, measured values only)")
    st.dataframe(scores.round(3), hide_index=True, width="stretch")
    st.caption(f"{station}: rated power {rated_kW:g} kW. Night forecasts are 0 by rule. "
               "Skill compares with the persistence benchmark on the same hours.")


def shap_for_row(model, horizon, row):
    """SHAP value of every feature for one row, and the base value."""
    names = json.loads((DEMO / f"features_{horizon}.json").read_text())
    X = row[names].astype(float).to_frame().T
    booster = load_booster(model, horizon)
    if model == "lgbm":
        values = booster.predict(X, pred_contrib=True)[0]
    else:
        values = booster.predict(xgb.DMatrix(X), pred_contribs=True)[0]
    return pd.DataFrame({"feature": names, "value": X.iloc[0].to_numpy(), "shap": values[:-1]}), values[-1]


def page_explain():
    horizon, station = st.session_state["horizon"], st.session_state["station"]
    st.title("Why did the model forecast this?")
    st.write("SHAP values split one forecast into the contribution of each input. "
             "Base value + all contributions = the forecast (in share of rated power).")
    data = load_features(horizon)
    data = data[(data["station"] == station) & (data["sun_elevation"] >= 5)]
    if data.empty:
        st.warning("No daytime rows for this station.")
        return

    left, middle, right = st.columns(3)
    model = left.selectbox("Model", ["lgbm", "xgb"], format_func=lambda m: MODEL_NAMES[m])
    days = sorted(set(data.index.date))
    default_day = pd.Timestamp("2023-07-03").date()
    day = middle.date_input("Day", value=default_day if default_day in days else days[0],
                            min_value=days[0], max_value=days[-1])
    times = data.index[data.index.date == day]
    if len(times) == 0:
        st.warning("No daytime data for this station on that day. Pick another day.")
        return
    time = right.select_slider("Time", options=list(times), value=times[len(times) // 2],
                               format_func=lambda t: f"{t:%H:%M}")
    row = data.loc[time]

    try:
        with st.spinner("Computing SHAP values..."):
            shap, base = shap_for_row(model, horizon, row)
    except Exception as error:
        st.error(f"Could not compute the explanation: {error}")
        return

    rated_kW = float(load_stations().loc[station, "rated_kW"])
    forecast = float(np.clip(base + shap["shap"].sum(), 0, 1.2))
    a, b, c = st.columns(3)
    a.metric("Forecast", f"{forecast * rated_kW:.1f} kW", f"{forecast:.2f} of rated", delta_color="off")
    actual = row["target"]
    b.metric("Actual", "missing" if pd.isna(actual) else f"{actual * rated_kW:.1f} kW")
    c.metric("Base value (average forecast)", f"{base:.2f} of rated")

    top = shap.reindex(shap["shap"].abs().sort_values(ascending=False).index).head(10).copy()
    top["label"] = [f"{f} = {v:.3g}" for f, v in zip(top["feature"], top["value"])]
    top["direction"] = np.where(top["shap"] >= 0, "pushes up", "pushes down")
    chart = alt.Chart(top).mark_bar().encode(
        x=alt.X("shap:Q", title="Contribution to the forecast (share of rated power)"),
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["pushes up", "pushes down"],
                                                       range=["#2a78d6", "#d6602a"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["feature", alt.Tooltip("value:Q", format=".4g"), alt.Tooltip("shap:Q", format=".4f")])
    st.altair_chart(chart, width="stretch")
    with st.expander("All features"):
        st.dataframe(shap.sort_values("shap", key=abs, ascending=False).round(4),
                     hide_index=True, width="stretch")
    with st.expander("Global importance (whole test sample)"):
        show_figure("07_shap_global.png")
        show_figure("07_shap_groups.png")


def flagged_runs(score, threshold):
    """Runs of consecutive flagged days: start, end, number of days, lowest score."""
    flag = (score < threshold).fillna(False)
    run_id = (flag != flag.shift()).cumsum()
    rows = []
    for _, run in score[flag].groupby(run_id[flag]):
        rows.append({"Start": run.index[0].date(), "End": run.index[-1].date(),
                     "Days": len(run), "Lowest score": round(float(run.min()), 3)})
    return pd.DataFrame(rows)


def page_faults():
    st.title("Underperformance flags")
    st.write("Each day the station's energy is compared with what an **expected-power model** "
             "(weather and sun position only, no past power) says it should be. The score is the "
             "3-day mean of *actual / expected - 1*, relative to the station's normal level and to "
             "the median of all stations on the same days (so a weather-sensor error does not flag every station). "
             "A day is flagged when the score falls below the threshold.")
    daily = load_daily()
    normal = load_table("08_station_normal.csv")
    alarms = load_table("08_false_alarms.csv")
    if normal is None or alarms is None or load_table("08_fleet_median.csv") is None:
        missing_table("08_false_alarms.csv")
        return
    normal = normal.set_index("station")
    thresholds = alarms.set_index("detector")["threshold"]
    fleet = load_table("08_fleet_median.csv")
    fleet = fleet.assign(day=pd.to_datetime(fleet["day"])).set_index("day")

    stations = sorted(set(daily["station"]) & set(normal.index))
    current = st.session_state["station"]
    station = st.selectbox("Station (all stations with fault data)", stations,
                           index=stations.index(current) if current in stations else 0)
    d = daily[(daily["station"] == station) & (daily["split"] == "test")]
    if d.empty:
        st.warning("No test-year data for this station.")
        return

    factor = np.ones(len(d))
    window = None
    with st.expander("What if? Inject a synthetic fault", expanded=False):
        inject = st.checkbox("Inject a fault into this station's 2023 data")
        a, b, c, e = st.columns(4)
        kind = a.selectbox("Kind", ["drop", "gradual", "zero"],
                           format_func={"drop": "Constant loss", "gradual": "Growing loss",
                                        "zero": "No output"}.get)
        severity = b.slider("Output lost (%)", 5, 80, 30, step=5, disabled=(kind == "zero"))
        duration = c.slider("Duration (days)", 2, 30, 14)
        start = pd.Timestamp(e.date_input("Start", value=pd.Timestamp("2023-07-10"),
                                          min_value=pd.Timestamp("2023-01-01"),
                                          max_value=pd.Timestamp("2023-12-01")))
        if inject:
            share = 1.0 if kind == "zero" else severity / 100
            factor = faults.fault_factor(d["day"], start, duration, kind, share)
            window = (start, start + pd.Timedelta(days=duration))

    scores = pd.DataFrame({
        "Expected-power model": faults.rolling_score(d["day"], faults.raw_ratio(d, "model", factor),
                                                     normal.loc[station, "model"], fleet["model"]),
        "Performance-ratio baseline": faults.rolling_score(d["day"], faults.raw_ratio(d, "pr", factor),
                                                           normal.loc[station, "pr"], fleet["pr"]),
    })
    long = scores.rename_axis("day").reset_index().melt("day", var_name="Detector", value_name="Score")
    names = ["Expected-power model", "Performance-ratio baseline"]
    lines = alt.Chart(long).mark_line(strokeWidth=1.6).encode(
        x=alt.X("day:T", title=None), y=alt.Y("Score:Q", title="Score (0 = normal)"),
        color=alt.Color("Detector:N", scale=alt.Scale(domain=names, range=["#2a78d6", "#9a9892"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["day:T", "Detector:N", alt.Tooltip("Score:Q", format=".3f")])
    rule = alt.Chart(pd.DataFrame({"y": [thresholds["model"]]})).mark_rule(
        color="#d6602a", strokeDash=[5, 4]).encode(y="y:Q")
    layers = [lines, rule]
    if window:
        band = alt.Chart(pd.DataFrame({"start": [window[0]], "end": [window[1]]})).mark_rect(
            color="#d6602a", opacity=0.15).encode(x="start:T", x2="end:T")
        layers.insert(0, band)
    st.altair_chart(alt.layer(*layers), width="stretch")
    st.caption(f"Dashed line: threshold of the expected-power model ({thresholds['model']:.2f}), "
               "3 robust standard deviations below normal, set on the train + validation period. "
               f"Baseline threshold: {thresholds['pr']:.2f}.")

    runs = flagged_runs(scores["Expected-power model"], thresholds["model"])
    if window:
        hit = scores["Expected-power model"][window[0]:window[1] + pd.Timedelta(days=2)]
        hit = hit[hit < thresholds["model"]]
        if len(hit):
            st.success(f"Injected fault detected {(hit.index[0] - window[0]).days} day(s) after it started.")
        else:
            st.error("Injected fault NOT detected: it is too small or too short for the threshold.")
    st.subheader("Flagged periods (expected-power model)")
    if runs.empty:
        st.write("No flagged days in 2023.")
    else:
        st.dataframe(runs, hide_index=True, width="stretch")
        st.caption("Flags are candidates to inspect, not confirmed faults: the data has no fault labels.")

    with st.expander("How good is the detector? (synthetic faults, all stations)"):
        summary = load_table("08_fault_detection.csv")
        if summary is None:
            missing_table("08_fault_detection.csv")
        else:
            show_figure("08_recall_by_severity.png")
            st.dataframe(summary.drop(columns=["commit"], errors="ignore").round(3),
                         hide_index=True, width="stretch")
            st.dataframe(alarms.drop(columns=["commit"], errors="ignore").round(3),
                         hide_index=True, width="stretch")


def page_efficiency():
    st.title("Efficiency: what does each model cost?")
    results = load_table("06_efficiency.csv")
    if results is None:
        missing_table("06_efficiency.csv")
        return
    horizon = st.session_state["horizon"]
    d = results[results["horizon"] == horizon]
    table = pd.DataFrame({
        "Model": d["model"],
        "Train time (s)": d["train_seconds"].round(1),
        "One forecast, model only (ms)": d["single_ms"].round(3),
        "One forecast, with features (ms)": d["end_to_end_single_ms"].round(1),
        "Per sample in a batch of 1,000 (ms)": d["batch_ms_per_sample"].round(4),
        "Size (kB)": d["size_kB"].round(0),
        "Test nMAE (%)": d["test_nmae_pct"].round(2),
        "Note": d["note"].fillna(""),
    })
    st.subheader(HORIZON_NAMES[horizon])
    st.dataframe(table, hide_index=True, width="stretch")
    first = results.iloc[0]
    st.caption(f"Measured on: {first['cpu']}, {first['cores']} logical cores, {first['ram_gb']} GB RAM, "
               f"{first['os']}. Latency is the median over repeated runs after warm-up.")
    show_figure("06_accuracy_vs_latency.png")


def page_generalisation():
    st.title("Does it work on stations it has never seen?")
    results = load_table("09_generalisation.csv")
    if results is None:
        missing_table("09_generalisation.csv")
        return
    horizon = st.session_state["horizon"]
    d = results[results["horizon"] == horizon]
    table = pd.DataFrame({
        "Model": d["model"], "Stations": d["group"], "Count": d["stations"],
        "nMAE (%)": d["nmae_pct"].round(2),
        "95 % CI": [f"{a:.2f} to {b:.2f}" for a, b in zip(d["nmae_pct_low"], d["nmae_pct_high"])],
        "Skill": d["skill"].round(3)})
    st.subheader(f"Seen vs unseen stations, {HORIZON_NAMES[horizon].lower()}")
    st.dataframe(table, hide_index=True, width="stretch")
    show_figure("09_seen_vs_unseen.png")

    st.subheader("Station-grouped 5-fold")
    kfold = load_table("09_kfold.csv")
    if kfold is None:
        missing_table("09_kfold.csv")
    else:
        st.dataframe(kfold[kfold["horizon"] == horizon].drop(columns=["commit"], errors="ignore").round(3),
                     hide_index=True, width="stretch")
        show_figure("09_kfold.png")

    st.subheader("Transfer check on a different rooftop dataset (Kaggle, 2024-2025)")
    kaggle = load_table("09_kaggle_transfer.csv")
    if kaggle is None:
        missing_table("09_kaggle_transfer.csv")
    else:
        st.dataframe(kaggle[["data", "model", "n_days", "nmae_pct", "skill", "skill_low", "skill_high"]].round(3),
                     hide_index=True, width="stretch")
        st.caption("A sanity check only: different years, hardware and weather station, "
                   "and the capacity of the Kaggle site is a proxy.")


PAGES = {
    "Overview": page_overview,
    "Forecast vs actual": page_forecast,
    "Explain a forecast": page_explain,
    "Fault flags": page_faults,
    "Efficiency": page_efficiency,
    "Generalisation": page_generalisation,
}


def main():
    require_bundle()
    manifest = load_manifest()

    st.session_state.setdefault("horizon", "1h")
    st.session_state.setdefault("station", manifest["stations"][0])

    with st.sidebar:
        st.header("☀️ Solar forecasting")
        page = st.radio("Page", list(PAGES), key="page")
        st.selectbox("Forecast horizon", list(HORIZON_NAMES), key="horizon",
                     format_func=lambda h: HORIZON_NAMES[h])
        st.selectbox("Station", manifest["stations"], key="station")
        st.divider()
        st.caption(f"Demo bundle built {manifest['built']} from commit `{manifest['commit']}`. "
                   "Test year 2023. Data: HKUST rooftop PV dataset (Dryad).")

    try:
        PAGES[page]()
    except FileNotFoundError as error:
        st.error(f"A file is missing: `{error.filename}`. Rebuild the bundle with "
                 "`python -m app.build_demo`.")


main()
