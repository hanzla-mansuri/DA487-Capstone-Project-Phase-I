"""Evaluation metrics (Stage 3; PLAN §3 and §6).

All predictions and targets are power / rated (`power_norm`). Errors in kW
multiply by the station's rated power.

- MAE, RMSE in kW;
- nMAE = MAE / rated capacity, in % (= mean of |error in power/rated| x 100);
- skill = 1 - RMSE_model / RMSE_benchmark, on power/rated, so every station
  counts equally (a large station does not dominate);
- 95 % confidence intervals by a block bootstrap over DAYS: whole days are
  drawn with replacement (all stations of a day together), because errors
  within a day are strongly correlated.

The model and the benchmark are always scored on the SAME rows.
"""
import numpy as np
import pandas as pd

SEED = 42
N_BOOT = 1000
METRICS = ["mae_kW", "rmse_kW", "nmae_pct", "rmse_norm", "skill"]


def daily_sums(target, pred, bench, rated_kW):
    """Sum the errors per day. All inputs are Series on the same time index.

    Working with day sums makes the bootstrap fast: a resample only has to
    add up the sums of the drawn days.
    """
    err = pred - target
    bench_err = bench - target
    parts = pd.DataFrame({
        "n": 1,
        "abs_norm": err.abs(),
        "sq_norm": err ** 2,
        "abs_kW": (err * rated_kW).abs(),
        "sq_kW": (err * rated_kW) ** 2,
        "bench_sq_norm": bench_err ** 2,
    }, index=target.index)
    return parts.groupby(parts.index.normalize()).sum()


def metrics_from_sums(sums):
    """MAE, RMSE, nMAE and skill from (summed) day sums."""
    n = sums["n"]
    # skill is undefined when the benchmark has no error at all
    if sums["bench_sq_norm"] > 0:
        skill = 1 - np.sqrt(sums["sq_norm"] / sums["bench_sq_norm"])
    else:
        skill = float("nan")
    return {
        "mae_kW": sums["abs_kW"] / n,
        "rmse_kW": np.sqrt(sums["sq_kW"] / n),
        "nmae_pct": 100 * sums["abs_norm"] / n,
        "rmse_norm": np.sqrt(sums["sq_norm"] / n),
        "skill": skill,
    }


def bootstrap_ci(days, n_boot=N_BOOT, seed=SEED):
    """95 % interval of every metric, resampling whole days.

    `days` is the output of `daily_sums`. Returns {metric: (low, high)}.
    """
    rng = np.random.default_rng(seed)
    values = days.to_numpy()
    results = {m: [] for m in METRICS}
    for _ in range(n_boot):
        drawn = rng.integers(0, len(values), len(values))
        total = pd.Series(values[drawn].sum(axis=0), index=days.columns)
        for m, v in metrics_from_sums(total).items():
            results[m].append(v)
    return {m: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for m, v in results.items()}


def score(target, pred, bench, rated_kW, n_boot=N_BOOT, seed=SEED):
    """All metrics with 95 % CIs, on the rows where target, model and
    benchmark all have a value.

    Returns a dict: n_rows, n_days, each metric, and <metric>_low/_high.
    """
    ok = target.notna() & pred.notna() & bench.notna()
    days = daily_sums(target[ok], pred[ok], bench[ok], rated_kW[ok])

    out = {"n_rows": int(ok.sum()), "n_days": len(days)}
    out.update(metrics_from_sums(days.sum()))
    if n_boot:
        for m, (low, high) in bootstrap_ci(days, n_boot, seed).items():
            out[f"{m}_low"] = low
            out[f"{m}_high"] = high
    return out


def per_station(target, pred, bench, rated_kW, station):
    """Metrics per station (no CIs). Returns one row per station."""
    rows = []
    for name in sorted(station.unique()):
        mine = station == name
        result = score(target[mine], pred[mine], bench[mine], rated_kW[mine], n_boot=0)
        result["station"] = name
        rows.append(result)
    return pd.DataFrame(rows).set_index("station")


def mean_station_skill(target, pred, bench, rated_kW, station):
    """Mean of the per-station skill scores (PLAN §3: reported next to the
    pooled skill)."""
    return per_station(target, pred, bench, rated_kW, station)["skill"].mean()
