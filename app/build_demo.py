"""Build the dashboard demo bundle (Stage 10).

    python -m app.build_demo

The dashboard reads ONLY `app/demo/` (git-ignored). No data or model is
committed because the dataset licences are unverified (DATA_SOURCES.md), so
this script rebuilds the bundle from the outputs of Stages 2-8:

    app/demo/features_<horizon>.parquet   2023 test rows of 10 stations:
                                          features, target, benchmark and
                                          the predictions of every model
    app/demo/gbm_<model>_<horizon>.*      the final GBMs (for live SHAP)
    app/demo/fault_daily.parquet          daily sums for the fault detectors
    app/demo/stations.csv                 rated power and role of the stations
    app/demo/manifest.json                what is inside, and from which commit

Run the stage scripts first (see README, "Reproduce").
"""
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate

DEMO = Path(__file__).resolve().parent / "demo"

# 8 seen stations and 2 random hold-outs (never trained on); all have 15-min data
STATIONS = ["Indoor Sports Centre", "LSK North", "SQ1", "SQ567", "UG Hall4",
            "Wong Check She Research Centre", "Zone A1", "Zone D",
            "LSK South", "SQ3"]


def bundle_horizon(horizon):
    """Test rows of the demo stations with every model's prediction."""
    table = evaluate.load_table(horizon)
    keep = (table["split"] == "test").to_numpy()
    part = table[keep].copy()

    gbm = pd.read_parquet(evaluate.PROCESSED / f"pred_gbm_{horizon}.parquet")
    models = ["lgbm", "xgb"]
    for column in ["benchmark"] + models:
        part[column] = gbm[column].to_numpy()

    lstm_file = evaluate.PROCESSED / f"pred_lstm_{horizon}.parquet"
    if lstm_file.exists():
        part["lstm"] = pd.read_parquet(lstm_file)["lstm"].to_numpy()
        models.append("lstm")

    part = part[part["station"].isin(STATIONS)]
    numbers = part.select_dtypes("float64").columns
    part[numbers] = part[numbers].astype(np.float32)
    part.to_parquet(DEMO / f"features_{horizon}.parquet")
    print(f"   {horizon}: {len(part)} rows, models {models}")
    return models


def main():
    DEMO.mkdir(exist_ok=True)
    manifest = {"built": date.today().isoformat(), "commit": evaluate.git_commit(),
                "stations": STATIONS, "models": {}}

    for horizon in evaluate.HORIZONS:
        manifest["models"][horizon] = bundle_horizon(horizon)
        for name in [f"gbm_lgbm_{horizon}.txt", f"gbm_xgb_{horizon}.json", f"features_{horizon}.json"]:
            shutil.copy(evaluate.MODELS / name, DEMO / name)

    daily = pd.read_parquet(evaluate.PROCESSED / "fault_daily.parquet")
    daily.to_parquet(DEMO / "fault_daily.parquet")

    inventory = pd.read_csv(evaluate.INVENTORY).set_index("station")
    sets = pd.read_csv(evaluate.TABLES / "02_station_sets.csv").set_index("station")
    stations = pd.DataFrame({"rated_kW": inventory["rated_value"], "role": sets["role"]})
    stations.to_csv(DEMO / "stations.csv")

    (DEMO / "manifest.json").write_text(json.dumps(manifest, indent=2))
    size = sum(f.stat().st_size for f in DEMO.iterdir()) / 1024 ** 2
    print(f"saved {DEMO} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
