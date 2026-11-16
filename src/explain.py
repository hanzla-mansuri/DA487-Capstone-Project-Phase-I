"""SHAP explanations of the best GBM and their stability (Stage 7).

    python -m src.explain

For every horizon the better of LightGBM / XGBoost is used (lower VALIDATION
RMSE in `configs/`). SHAP values are TreeSHAP values computed by the model
library itself (`pred_contrib`), which is the same exact algorithm the `shap`
package uses for trees. Units: power / rated. For one row, the SHAP values
of all features plus the base value add up to the prediction.

Sample: 20,000 random test rows (seen stations, usable targets, daytime).

Stability (PLAN Stage 7): Spearman rank correlation of the importance
(mean |SHAP|) vectors
- between stations,
- between seasons,
- between 5 retrained models (bootstrap resample of the training rows and a
  different seed each).

Outputs:
    results/tables/07_shap_importance.csv, 07_shap_groups.csv,
    results/tables/07_shap_stability.csv, 07_physical_checks.csv
    results/figures/07_shap_global.png, 07_shap_groups.png,
    results/figures/07_shap_dependence_1h.png, 07_shap_local_1h.png
"""
import itertools

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from src import evaluate
from src.models import gbm
from src.models.ridge import feature_columns
from src.splits import season
from src.train_gbm import load_config, model_path

SEED = 42
N_SAMPLE = 20_000
N_RETRAIN = 5
TOP_K = 5

GROUP_COLORS = {
    "own power (lags)": "#2a78d6",
    "measured irradiance / clear-sky index": "#e0a526",
    "solar geometry": "#d6602a",
    "calendar": "#6f5bd3",
    "other weather": "#3aa37a",
    "station metadata": "#7a7974",
}


def group_of(feature):
    """Feature group, from the column name."""
    if feature.startswith("power_"):
        return "own power (lags)"
    if feature.startswith(("irr_", "csi_")) or feature in ("irradiance", "csi"):
        return "measured irradiance / clear-sky index"
    if feature in ("sun_elevation", "sun_azimuth", "clearsky_ghi"):
        return "solar geometry"
    if feature in ("is_optimizer", "tilt_deg", "multi_orientation"):
        return "station metadata"
    if feature.startswith(("hour", "doy", "day_of_year", "month")):
        return "calendar"
    return "other weather"


def best_model_name(horizon):
    """LightGBM or XGBoost: the lower validation RMSE wins."""
    scores = {m: load_config(m, horizon)["val_rmse"] for m in ["lgbm", "xgb"]}
    return min(scores, key=scores.get)


def load_booster(model_name, horizon):
    path = str(model_path(model_name, horizon))
    if model_name == "lgbm":
        return lgb.Booster(model_file=path)
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


def shap_values(booster, X):
    """TreeSHAP values, one column per feature (the base value is dropped)."""
    if isinstance(booster, lgb.Booster):
        values = booster.predict(X, pred_contrib=True)
    else:
        values = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    return pd.DataFrame(values[:, :-1], index=X.index, columns=X.columns), float(values[0, -1])


def importance(shap):
    """Global importance: mean absolute SHAP value per feature."""
    return shap.abs().mean()


def test_sample(table):
    """Random daytime test rows of seen stations (fixed seed)."""
    mask = evaluate.eval_mask(table, "test", "daytime")
    rows = table[mask].reset_index()
    return rows.sample(min(N_SAMPLE, len(rows)), random_state=SEED).reset_index(drop=True)


def pairwise_spearman(vectors):
    """Spearman correlation of every pair of importance vectors."""
    return [spearmanr(a, b)[0] for a, b in itertools.combinations(vectors, 2)]


def top_overlap(vectors, k=TOP_K):
    """Mean share of the top-k features that two vectors have in common."""
    tops = [set(v.nlargest(k).index) for v in vectors]
    return float(np.mean([len(a & b) / k for a, b in itertools.combinations(tops, 2)]))


def stability_row(horizon, kind, vectors):
    rho = pairwise_spearman(vectors)
    return {"horizon": horizon, "across": kind, "n_groups": len(vectors),
            "mean_spearman": np.mean(rho), "min_spearman": np.min(rho),
            f"top{TOP_K}_overlap": top_overlap(vectors)}


def retrained_importances(table, model_name, horizon, sample_X):
    """Importance vectors of N_RETRAIN models, each fitted on a bootstrap
    resample of the train + validation rows with its own seed."""
    config = load_config(model_name, horizon)
    X, y = gbm.data(table, ["train", "val"])
    vectors = []
    for seed in range(N_RETRAIN):
        rng = np.random.default_rng(seed)
        rows = rng.integers(0, len(X), len(X))
        if model_name == "lgbm":
            model = lgb.LGBMRegressor(n_estimators=config["n_trees"], subsample_freq=1,
                                      random_state=seed, n_jobs=4, verbose=-1, **config["params"])
            model.fit(X.iloc[rows], y.iloc[rows])
            booster = model.booster_
        else:
            model = xgb.XGBRegressor(n_estimators=config["n_trees"], tree_method="hist",
                                     random_state=seed, n_jobs=4, **config["params"])
            model.fit(X.iloc[rows], y.iloc[rows])
            booster = model.get_booster()
        shap, _ = shap_values(booster, sample_X)
        vectors.append(importance(shap))
        print(f"      retrain {seed + 1}/{N_RETRAIN} done")
    return vectors


def physical_checks(horizon, X, shap):
    """Does a higher feature value push the prediction up or down?
    Spearman correlation between the feature value and its SHAP value."""
    expected = {"power_lag1": "+", "power_d1": "+", "irr_lag1": "+", "csi_lag1": "+",
                "csi_d1_daymean": "+", "clearsky_ghi": "+", "sun_elevation": "+",
                "temperature_lag1": "-", "temp_d1_mean": "-"}
    rows = []
    for feature, sign in expected.items():
        if feature not in X.columns:
            continue
        ok = X[feature].notna()
        rho = spearmanr(X[feature][ok], shap[feature][ok])[0]
        rows.append({"horizon": horizon, "feature": feature, "expected_direction": sign,
                     "spearman_value_vs_shap": rho,
                     "agrees": (rho > 0) == (sign == "+"),
                     "mean_abs_shap": shap[feature].abs().mean()})
    return rows


def explain_horizon(horizon):
    """All tables of one horizon. Returns a dict of results."""
    table = evaluate.load_table(horizon)
    name = best_model_name(horizon)
    booster = load_booster(name, horizon)
    sample = test_sample(table)
    X = sample[feature_columns(table)]
    shap, base = shap_values(booster, X)
    print(f"== {horizon}: {name}, {len(X)} rows, base value {base:.4f}")

    imp = importance(shap).sort_values(ascending=False)
    imp_table = pd.DataFrame({
        "horizon": horizon, "model": name, "feature": imp.index,
        "group": [group_of(f) for f in imp.index],
        "mean_abs_shap": imp.to_numpy(),
        "share_pct": 100 * imp.to_numpy() / imp.sum(),
        "rank": range(1, len(imp) + 1)})

    stations = [importance(shap[sample["station"] == s]) for s in sorted(sample["station"].unique())]
    seasons_of = season(pd.DatetimeIndex(sample["time"])).to_numpy()
    seasons = [importance(shap[seasons_of == s]) for s in ["winter", "spring", "summer", "autumn"]]
    retrained = retrained_importances(table, name, horizon, X)
    stability = [stability_row(horizon, "stations", stations),
                 stability_row(horizon, "seasons", seasons),
                 stability_row(horizon, "retrained models (bootstrap + seed)", retrained)]

    return {"importance": imp_table, "stability": stability,
            "checks": physical_checks(horizon, X, shap),
            "sample": sample, "X": X, "shap": shap, "base": base, "booster": booster}


# ------------------------------------------------------------ figures

def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="x", color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def plot_global(importances):
    fig, axes = plt.subplots(1, len(importances), figsize=(5 * len(importances), 4.6))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (horizon, imp) in zip(axes, importances.items()):
        top = imp.head(12).iloc[::-1]
        ax.barh(top["feature"], top["mean_abs_shap"], color=[GROUP_COLORS[g] for g in top["group"]])
        ax.set_title(f"{horizon} ({imp['model'].iloc[0]})")
        ax.set_xlabel("mean |SHAP| (power / rated)")
        ax.tick_params(labelsize=8)
        style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    fig.legend(handles, GROUP_COLORS.keys(), ncol=3, fontsize=8, frameon=False, loc="lower center")
    fig.suptitle("Global feature importance, 2023 test sample, daytime", fontsize=11)
    plt.tight_layout(rect=(0, 0.09, 1, 1))
    plt.savefig(evaluate.FIGURES / "07_shap_global.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_groups(groups):
    """Share of the total importance per feature group and horizon."""
    wide = groups.pivot(index="horizon", columns="group", values="share_pct").loc[list(evaluate.HORIZONS)]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_facecolor("#fcfcfb")
    left = np.zeros(len(wide))
    for group, color in GROUP_COLORS.items():
        if group not in wide:
            continue
        values = wide[group].fillna(0).to_numpy()
        ax.barh(wide.index, values, left=left, color=color, label=group, height=0.6)
        for y, (v, l) in enumerate(zip(values, left)):
            if v >= 6:
                ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white")
        left += values
    ax.set_xlabel("share of total mean |SHAP| (%)")
    ax.invert_yaxis()
    style(ax)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    ax.set_title("Grouped importance per horizon", fontsize=11)
    plt.tight_layout()
    plt.savefig(evaluate.FIGURES / "07_shap_groups.png", dpi=130, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)


def plot_dependence(result, horizon):
    """Feature value against its SHAP value for the top 3 features and temperature."""
    features = list(result["importance"]["feature"].head(3))
    for extra in ["temperature_lag1", "temp_d1_mean"]:
        if extra in result["X"].columns and extra not in features:
            features.append(extra)
            break
    rows = result["X"].sample(min(4000, len(result["X"])), random_state=SEED).index
    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 3.4))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, feature in zip(axes, features):
        ax.scatter(result["X"].loc[rows, feature], result["shap"].loc[rows, feature],
                   s=4, alpha=0.35, color="#2a78d6", linewidths=0)
        ax.axhline(0, color="#52514e", lw=0.6)
        ax.set_xlabel(feature)
        ax.set_facecolor("#fcfcfb")
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("SHAP value (power / rated)")
    fig.suptitle(f"Dependence plots, {horizon} horizon", fontsize=11)
    plt.tight_layout()
    plt.savefig(evaluate.FIGURES / f"07_shap_dependence_{horizon}.png", dpi=130,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def local_examples(result):
    """Three test rows to explain: a clear hour, a cloudy midday hour, and
    the largest change since the last known value (a ramp)."""
    sample, X = result["sample"], result["X"]
    midday = sample["hour"].between(11, 13)
    lag = "power_lag1" if "power_lag1" in X else "power_d1"
    csi = "csi_lag1" if "csi_lag1" in X else "csi_d1_daymean"
    return {
        "clear hour (highest output)": sample["target"].idxmax(),
        "cloudy midday hour (lowest clear-sky index)": X.loc[midday, csi].idxmin(),
        "ramp (largest change since the last value)": (sample["target"] - X[lag]).abs().idxmax(),
    }


def plot_local(result, horizon):
    examples = local_examples(result)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (title, row) in zip(axes, examples.items()):
        shap = result["shap"].loc[row]
        top = shap.reindex(shap.abs().sort_values(ascending=False).index).head(8).iloc[::-1]
        labels = [f"{f} = {result['X'].loc[row, f]:.3g}" for f in top.index]
        ax.barh(labels, top, color=["#2a78d6" if v >= 0 else "#d6602a" for v in top])
        ax.axvline(0, color="#52514e", lw=0.6)
        prediction = result["base"] + shap.sum()
        s = result["sample"].loc[row]
        ax.set_title(f"{title}\n{s['station']}, {s['time']:%Y-%m-%d %H:%M}\n"
                     f"base {result['base']:.2f} + SHAP = {prediction:.2f}, actual {s['target']:.2f}",
                     fontsize=9)
        ax.tick_params(labelsize=8)
        style(ax)
    fig.suptitle(f"Local explanations, {horizon} horizon (blue pushes the forecast up, orange down)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(evaluate.FIGURES / f"07_shap_local_{horizon}.png", dpi=130,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    importances, stability, checks = {}, [], []
    for horizon in evaluate.HORIZONS:
        result = explain_horizon(horizon)
        importances[horizon] = result["importance"]
        stability += result["stability"]
        checks += result["checks"]
        if horizon == "1h":
            plot_dependence(result, horizon)
            plot_local(result, horizon)

    commit = evaluate.git_commit()
    imp = pd.concat(importances.values())
    groups = imp.groupby(["horizon", "group"], as_index=False)[["mean_abs_shap", "share_pct"]].sum()
    tables = {"07_shap_importance.csv": imp, "07_shap_groups.csv": groups,
              "07_shap_stability.csv": pd.DataFrame(stability),
              "07_physical_checks.csv": pd.DataFrame(checks)}
    for name, frame in tables.items():
        frame["commit"] = commit
        frame.to_csv(evaluate.TABLES / name, index=False, float_format="%.5f")
        print(f"saved {name}")

    plot_global(importances)
    plot_groups(groups)
    print(pd.DataFrame(stability).to_string(index=False))
    print(pd.DataFrame(checks).to_string(index=False))
    print(groups.to_string(index=False))


if __name__ == "__main__":
    main()
