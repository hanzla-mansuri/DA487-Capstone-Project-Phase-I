# Work Log

| Date | Hours | Work done |
|---|---|---|

## 2026-11-06 — hours: _TBD (fill in)_
- Set up repo scaffold, Python venv (pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm, shap, jupyter), `.gitignore`, git init.
- Built `notebooks/01_eda.ipynb` (Kaggle India dataset audit); tested on synthetic data only, never run on real data.
- Scope change: project moved to multi-horizon forecasting on the HKUST rooftop PV dataset (Lin et al., Sci. Data 2025).
- Inspected `data/raw/` (structure, timestamps, units, missing data) and reviewed git state. No models trained.
- Updated `.gitignore`, `README.md` (new title/scope/citation) and created this log.

## 2026-11-06 (later): Stage 0 git setup, hours: _TBD (fill in)_
- Inspected GitHub remote: 1 old commit (Kaggle-India EDA, unrelated history); no secrets or large files.
- Backups: tag `backup-before-capstone-cleanup`, branches `archive/old-scope`, `archive/github-old-main`.
- Merged remote `main` with `--allow-unrelated-histories` (no force push); archived old notebook/figures under `archive/`.
- Hardened `.gitignore`, added `.gitkeep`, `CLAUDE.md`, `DATA_SOURCES.md`, `PLAN.md`; updated README.
- Documented both datasets (primary HKUST Dryad in `data/hkust/`, secondary Kaggle in `data/raw/`).
- Pushed backup tag, both archive branches and `main` (fast-forward, no force). Stage 0 DONE.
