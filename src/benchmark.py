"""Efficiency benchmark (Stage 6).

    python -m src.benchmark

For every model and horizon, on this machine (CPU only):
- train time (s): final fit on train + validation;
- model-only latency: one sample, and per sample in a batch of 1,000
  (median over repeats, after warm-up);
- end-to-end latency for one forecast = feature construction for one
  station (from 10 days of cleaned data: weather join, sun position,
  lags) + the model's one-sample latency;
- model size on disk;
- accuracy on the 2023 test set (daytime, seen stations), so the cost can be
  set against the error.

Exports: LightGBM, XGBoost and the LSTM to ONNX (float32), and the LSTM to
int8 with torch dynamic quantization. Each export is scored again on the
test set to measure the accuracy change. An export that fails is recorded
with its error instead of stopping the run (converters are version-sensitive).

Peak memory is not measured (PLAN: "if it is easy to measure").

Outputs:
    results/tables/06_efficiency.csv   (with the hardware specs)
    results/figures/06_accuracy_vs_latency.png
"""
import argparse
import json
import subprocess
import sys
import time
import warnings

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from src import data_io, evaluate, features, metrics, pipeline, preprocess
from src.models import baselines, lstm, ridge
from src.models.baselines import finish
from src.models.ridge import feature_columns
from src.utils import hardware

warnings.filterwarnings("ignore")

# ridge alphas chosen on validation in Stage 3 (PLAN Change log 2026-11-16)
RIDGE_ALPHA = {"15min": 0.1, "1h": 10, "dayahead": 10000}
BATCH = 1000
SINGLE_REPEATS = 200
BATCH_REPEATS = 20
FEATURE_REPEATS = 20
HISTORY_DAYS = 10
EXAMPLE_STATION = "LSK North"


def median_ms(fn, repeats, warmup=5):
    """Median run time of `fn()` in milliseconds, after `warmup` runs."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return 1000 * float(np.median(times))


def size_kb(path):
    return path.stat().st_size / 1024


# ------------------------------------------------------------ feature time

def feature_time_ms(horizon):
    """Time to build the features of ONE forecast for one station, from the
    last 10 days of cleaned power and weather (reading files is not counted)."""
    inventory = pd.read_csv(pipeline.INVENTORY)
    files = data_io.list_station_files().set_index("station")
    row = inventory.set_index("station").loc[EXAMPLE_STATION]
    freq = pipeline.grid_of(horizon)

    df = pipeline.clean_station(EXAMPLE_STATION, files.loc[EXAMPLE_STATION, "path"], True, inventory)
    if freq == "1h":
        df = preprocess.to_hourly(df)
    end = pd.Timestamp("2023-06-30 23:59")
    start = end - pd.Timedelta(days=HISTORY_DAYS)
    power = df[start:end]
    weather = preprocess.build_weather(freq)[start:end]

    def build():
        base = features.build_base(power, weather, freq, row["latitude"], row["longitude"])
        out = pipeline.BUILDERS[horizon](base)
        return out.tail(lstm.WINDOW.get(horizon, 1))

    return median_ms(build, FEATURE_REPEATS, warmup=2)


# ------------------------------------------------------------ model wrappers
# Every wrapper is a function: numpy feature rows -> numpy predictions.

def onnx_session(path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def export_lgbm_onnx(booster, n_features, path):
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    model = convert_lightgbm(booster, initial_types=[("input", FloatTensorType([None, n_features]))])
    path.write_bytes(model.SerializeToString())


def export_xgb_onnx(booster, n_features, path):
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    booster.feature_names = [f"f{i}" for i in range(n_features)]   # the converter needs f0, f1, ...
    model = convert_xgboost(booster, initial_types=[("input", FloatTensorType([None, n_features]))])
    path.write_bytes(model.SerializeToString())


def export_lstm_onnx(net, window, n_features, path):
    example = torch.zeros(1, window, n_features)
    torch.onnx.export(net, (example,), str(path), input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, dynamo=False)


def quantize_lstm(net):
    """int8 dynamic quantization of the LSTM and linear weights.
    The qnnpack engine is used because this CPU has no AVX2."""
    from torch.ao.quantization import quantize_dynamic
    torch.backends.quantized.engine = "qnnpack"
    return quantize_dynamic(net, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8)


def tabular_models(horizon, table, full_table):
    """{name: (predict function, train seconds, size in kB, note)} for the
    models that take one feature row."""
    n = len(feature_columns(table))
    models = {}

    # persistence: no model, the prediction is an input column
    names = feature_columns(table)
    column = names.index("power_lag1" if horizon != "dayahead" else "power_d1")
    models[evaluate.BENCHMARK[horizon]] = (lambda X: X[:, column], 0.0, 0.0, "")

    # ridge: refit on train + validation and time it
    start = time.time()
    model = ridge.fit(full_table, RIDGE_ALPHA[horizon], splits=["train", "val"])
    seconds = time.time() - start
    path = evaluate.MODELS / f"ridge_{horizon}.joblib"
    joblib.dump(model, path)
    models["ridge"] = (model.predict, seconds, size_kb(path), "")

    gbm_results = pd.read_csv(evaluate.TABLES / "04_gbm_results.csv")
    gbm_results = gbm_results[(gbm_results["horizon"] == horizon) & (gbm_results["split"] == "test")]
    fit_seconds = gbm_results.groupby("model")["fit_seconds"].first()

    path = evaluate.MODELS / f"gbm_lgbm_{horizon}.txt"
    booster = lgb.Booster(model_file=str(path))
    models["lgbm"] = (booster.predict, fit_seconds["lgbm"], size_kb(path), "")
    models["lgbm_onnx"] = export_tabular(export_lgbm_onnx, booster, n,
                                         evaluate.MODELS / f"gbm_lgbm_{horizon}.onnx")

    path = evaluate.MODELS / f"gbm_xgb_{horizon}.json"
    xbooster = xgb.Booster()
    xbooster.load_model(str(path))
    models["xgb"] = (xbooster.inplace_predict, fit_seconds["xgb"], size_kb(path), "")
    models["xgb_onnx"] = export_tabular(export_xgb_onnx, xbooster, n,
                                        evaluate.MODELS / f"gbm_xgb_{horizon}.onnx")
    return models


def export_tabular(export, booster, n_features, path):
    """Export a GBM to ONNX; on failure return the error as the note."""
    try:
        export(booster, n_features, path)
        session = onnx_session(path)
        predict = lambda X: session.run(None, {"input": X.astype(np.float32)})[0].ravel()
        return (predict, np.nan, size_kb(path), "ONNX float32")
    except Exception as error:      # converters break between versions: record, do not stop
        return (None, np.nan, np.nan, f"export failed: {type(error).__name__}: {str(error)[:120]}")


def lstm_models(horizon):
    """{name: (net-like function on window tensors, train seconds, size, note)}."""
    path = evaluate.MODELS / f"lstm_{horizon}.pt"
    net, scaler, info = lstm.load(path)
    window, n = info["window"], len(scaler["mean"])
    models = {"lstm": (net, info["fit_seconds"], size_kb(path), f"hidden {info['hidden']}, fp32")}

    # the int8 model is measured in its own process, see `int8_row`
    try:
        opath = evaluate.MODELS / f"lstm_{horizon}.onnx"
        export_lstm_onnx(net, window, n, opath)
        session = onnx_session(opath)
        run = lambda xb: torch.from_numpy(session.run(None, {"input": xb.numpy()})[0])
        models["lstm_onnx"] = (run, np.nan, size_kb(opath), "ONNX float32")
    except Exception as error:
        models["lstm_onnx"] = (None, np.nan, np.nan, f"export failed: {type(error).__name__}: {str(error)[:120]}")
    return models, scaler


# ------------------------------------------------------------ measuring

def accuracy(table, pred, bench, rated):
    """Test-set daytime nMAE (%) and RMSE (power/rated), seen stations."""
    mask = evaluate.eval_mask(table, "test", "daytime")
    result = metrics.score(table["target"][mask], pred[mask], bench[mask], rated[mask], n_boot=0)
    return result["nmae_pct"], result["rmse_norm"], result["skill"]


def bench_horizon(horizon):
    full_table = evaluate.load_table(horizon)
    table = full_table[full_table["split"] == "test"]
    bench = evaluate.benchmark(table, horizon)
    rated = evaluate.rated_kW(table)
    feature_ms = feature_time_ms(horizon)
    print(f"== {horizon}: feature construction {feature_ms:.1f} ms per forecast")

    X = table[feature_columns(table)].to_numpy(dtype=np.float64)
    seen_day = np.flatnonzero(evaluate.eval_mask(table, "test", "daytime").to_numpy())
    one = X[seen_day[:1]]
    many = X[seen_day[:BATCH]]

    rows = []
    for name, (predict, seconds, size, note) in tabular_models(horizon, table, full_table).items():
        row = {"horizon": horizon, "model": name, "train_seconds": seconds, "size_kB": size, "note": note}
        if predict is not None:
            pred = finish(predict(X), table)
            row["test_nmae_pct"], row["test_rmse_norm"], row["test_skill"] = accuracy(table, pred, bench, rated)
            row["single_ms"] = median_ms(lambda: predict(one), SINGLE_REPEATS)
            row["batch_ms_per_sample"] = median_ms(lambda: predict(many), BATCH_REPEATS) / BATCH
        rows.append(row)

    if horizon in lstm.WINDOW:
        models, scaler = lstm_models(horizon)
        window = lstm.WINDOW[horizon]
        scaled = lstm.scale(table, scaler)
        ends = lstm.window_ends(table, window, evaluate.eval_mask(table, "test", "daytime"))
        one_w = lstm.make_batch(scaled, ends[:1], window)
        many_w = lstm.make_batch(scaled, ends[:BATCH], window)
        for name, (net, seconds, size, note) in models.items():
            row = {"horizon": horizon, "model": name, "train_seconds": seconds, "size_kB": size, "note": note}
            if net is not None:
                pred = lstm.predict(net, scaler, table, horizon)
                row["test_nmae_pct"], row["test_rmse_norm"], row["test_skill"] = accuracy(table, pred, bench, rated)
                with torch.no_grad():
                    row["single_ms"] = median_ms(lambda: net(one_w), SINGLE_REPEATS)
                    row["batch_ms_per_sample"] = median_ms(lambda: net(many_w), BATCH_REPEATS) / BATCH
            rows.append(row)

        rows.append(int8_row(horizon))

    out = pd.DataFrame(rows)
    out["feature_ms"] = feature_ms
    out["end_to_end_single_ms"] = out["feature_ms"] + out["single_ms"]
    return out


def measure_int8(horizon):
    """Size, accuracy and latency of the int8 LSTM (runs in a child process)."""
    table = evaluate.load_table(horizon)
    table = table[table["split"] == "test"]
    bench = evaluate.benchmark(table, horizon)
    rated = evaluate.rated_kW(table)
    net, scaler, info = lstm.load(evaluate.MODELS / f"lstm_{horizon}.pt")
    quantized = quantize_lstm(net)
    path = evaluate.MODELS / f"lstm_{horizon}_int8.pt"
    torch.save(quantized.state_dict(), path)

    window = lstm.WINDOW[horizon]
    scaled = lstm.scale(table, scaler)
    ends = lstm.window_ends(table, window, evaluate.eval_mask(table, "test", "daytime"))
    one_w = lstm.make_batch(scaled, ends[:1], window)
    many_w = lstm.make_batch(scaled, ends[:BATCH], window)

    row = {"horizon": horizon, "model": "lstm_int8", "train_seconds": None, "size_kB": size_kb(path),
           "note": "torch dynamic quantization, int8 (qnnpack)"}
    pred = lstm.predict(quantized, scaler, table, horizon)
    row["test_nmae_pct"], row["test_rmse_norm"], row["test_skill"] = accuracy(table, pred, bench, rated)
    with torch.no_grad():
        row["single_ms"] = median_ms(lambda: quantized(one_w), SINGLE_REPEATS)
        row["batch_ms_per_sample"] = median_ms(lambda: quantized(many_w), BATCH_REPEATS) / BATCH
    return row


def int8_row(horizon, attempts=3):
    """Run `measure_int8` in a child process.

    On this CPU (no AVX2) quantized inference sometimes stops with an
    "illegal instruction" signal, which cannot be caught inside Python. In a
    child process the crash is seen as a failed exit code, so it can be
    retried and, if it keeps failing, written down instead of ending the run.
    """
    out_file = evaluate.MODELS / f"int8_{horizon}.json"
    for attempt in range(1, attempts + 1):
        out_file.unlink(missing_ok=True)
        done = subprocess.run([sys.executable, "-W", "ignore", "-m", "src.benchmark",
                               "--int8", horizon], cwd=evaluate.REPO)
        if done.returncode == 0 and out_file.exists():
            row = json.loads(out_file.read_text())
            if attempt > 1:
                row["note"] += f"; crashed {attempt - 1} time(s) before this run"
            return row
        print(f"   int8 {horizon}: attempt {attempt} failed (exit code {done.returncode})")
    return {"horizon": horizon, "model": "lstm_int8",
            "note": f"int8 inference crashed {attempts} times (illegal instruction, CPU without AVX2)"}


def add_export_change(results):
    """Accuracy change of every export against its original model."""
    original = {"lgbm_onnx": "lgbm", "xgb_onnx": "xgb", "lstm_onnx": "lstm", "lstm_int8": "lstm"}
    base = results.set_index(["horizon", "model"])["test_nmae_pct"]
    change = []
    for row in results.itertuples():
        if row.model in original:
            change.append(row.test_nmae_pct - base[(row.horizon, original[row.model])])
        else:
            change.append(np.nan)
    results["nmae_change_vs_original_pct_points"] = change
    return results


def plot(results):
    """Test nMAE against the end-to-end latency of one forecast."""
    horizons = [h for h in evaluate.HORIZONS if h in set(results["horizon"])]
    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, horizon in zip(axes[0], horizons):
        d = results[(results["horizon"] == horizon) & results["single_ms"].notna()]
        ax.scatter(d["single_ms"], d["test_nmae_pct"], color="#2a78d6", s=40, zorder=3)
        for r in d.itertuples():
            ax.annotate(r.model, (r.single_ms, r.test_nmae_pct), fontsize=8,
                        xytext=(5, 4), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_facecolor("#fcfcfb")
        ax.set_title(f"{horizon} (features add {d['feature_ms'].iloc[0]:.0f} ms)")
        ax.set_xlabel("model-only latency, one sample (ms, log scale)")
        ax.grid(color="#e6e5e1", lw=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("test nMAE, daytime (% of rated)")
    fig.suptitle("Accuracy vs latency (down and left is better)", fontsize=11)
    plt.tight_layout()
    plt.savefig(evaluate.FIGURES / "06_accuracy_vs_latency.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    results = pd.concat([bench_horizon(h) for h in evaluate.HORIZONS], ignore_index=True)
    results = add_export_change(results)
    for key, value in hardware().items():
        results[key] = value
    results["commit"] = evaluate.git_commit()
    results["date"] = pd.Timestamp.today().date().isoformat()
    results.to_csv(evaluate.TABLES / "06_efficiency.csv", index=False, float_format="%.5f")
    plot(results)
    show = ["horizon", "model", "train_seconds", "single_ms", "batch_ms_per_sample",
            "end_to_end_single_ms", "size_kB", "test_nmae_pct", "nmae_change_vs_original_pct_points", "note"]
    print(results[show].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6: efficiency benchmark.")
    parser.add_argument("--int8", choices=sorted(lstm.WINDOW),
                        help="internal: measure the int8 LSTM of one horizon and write a json file")
    args = parser.parse_args()
    if args.int8:
        torch.set_num_threads(4)
        result = measure_int8(args.int8)
        (evaluate.MODELS / f"int8_{args.int8}.json").write_text(json.dumps(result))
    else:
        main()
