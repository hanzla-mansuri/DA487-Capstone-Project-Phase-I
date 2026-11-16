# DECISIONS: choices made during the autonomous run

Short log of technical choices made without asking (see `PROGRESS.md` for the run rules).
Anything that changes the plan itself is also in the `PLAN.md` Change log.

| Date | Decision | Why |
|---|---|---|
| 2026-11-16 | `PLAN.md` checkboxes stay the task tracker; `PROGRESS.md` only lists milestones and the resume checkpoint. | One source of truth; avoids two task lists drifting apart. |
| 2026-11-16 | `WORKLOG.md` is not updated. | The user dropped it earlier. |
| 2026-11-16 | Dashboard lives at `app/streamlit_app.py`, not `app.py`. | `PLAN.md` Stage 10 names this path. |
| 2026-11-16 | No demo data or models are committed; `app/build_demo.py` rebuilds the bundle from raw data. | Dataset licences are unverified ("skip licences"). |
| 2026-11-16 | Untracked `test.py` in the repo root is never staged. | Not part of the project; commits add files by name. |
| 2026-11-16 | Irradiance is divided by the sensor's own trailing clear-day level (30 days, causal) instead of dropping irradiance features or applying a fixed correction factor for 2023. | The sensor scale jumps by about 1.6 on 2023-02-15, so the test year is on a different scale from training. A fixed factor would be estimated from test-year data; dropping the features loses skill. The trailing rule uses only earlier irradiance and the sun position. Details in the `PLAN.md` Change log. |
| 2026-11-16 | After the irradiance fix the feature tables were rebuilt and tuning was repeated for all horizons; the first (uncalibrated) test results are kept in `results/tables/04_gbm_results_uncalibrated.csv` as an ablation. | Tuned settings and validation scores must belong to the features that are actually used; the old numbers show what the sensor step costs. |
| 2026-11-16 | Heavy jobs (tuning, training, benchmark) run one at a time. | The laptop CPU has 2 cores / 4 threads; two multi-threaded jobs at once slow each other down many times over. |
| 2026-11-16 | LSTM input = the last 16 (15 min) or 24 (1 h) rows of the same feature table the GBMs use; one LSTM layer, hidden size 32 or 64 chosen on validation. | Same features and splits as the other models (PLAN Stage 5), no new leakage surface, and small enough for CPU training. |
| 2026-11-16 | SHAP values come from the GBM libraries' own TreeSHAP (`pred_contrib`), plots are plain matplotlib. | Same exact algorithm as the `shap` package for trees, without depending on its plotting code under Python 3.14. |
| 2026-11-16 | int8 quantization uses the `qnnpack` engine. | The CPU (i5-3437U) has no AVX2; the default engine crashes with an illegal instruction. |
| 2026-11-16 | Fault detectors work on daily energy ratios with a 3-day rolling mean; thresholds are the pooled 1 % validation quantile after dividing by each station's validation median. | Daily sums remove hour-to-hour cloud noise; the per-station normal level removes fixed differences (shading, orientation); one pooled quantile keeps the rule simple and gives both detectors the same nominal false-alarm rate. |
| 2026-11-16 | Kaggle capacity proxy: the site's 99th-percentile daytime power is assumed to sit at the same share of rated power as the median HKUST station in training. | The Kaggle files give no rated power; some scale is needed to compare normalised errors. Stated as a caveat. |
