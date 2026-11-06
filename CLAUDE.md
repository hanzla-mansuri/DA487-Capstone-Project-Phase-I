# CLAUDE.md: DA487 capstone

**Project:** *Explainable and Efficient Machine Learning for Multi-Horizon Solar Power Forecasting* (B.Sc. Data Science capstone, 12 credits, ~10-12 h/week for 12 weeks).
It forecasts rooftop PV power at 15 min, 1 h and day-ahead horizons, using a ladder of models (persistence/linear → XGBoost/LightGBM → LSTM). It also covers an efficiency benchmark, SHAP explainability, residual-based fault flagging, an unseen-station test and a Streamlit dashboard.

- **Primary data:** HKUST Dryad dataset in `data/hkust/`.
- **Secondary data:** the Kaggle dataset in `data/raw/`, used for an out-of-time check only.
- See `DATA_SOURCES.md`.

## Session rule
**At the start of every session, read PLAN.md, WORKLOG.md and DATA_SOURCES.md, state the current stage and the next unchecked task, and wait for my go-ahead.**

## Working rules
Follow the Resume Protocol in `PLAN.md` §8. In short:
- One stage at a time.
- Tick tasks, log them in `WORKLOG.md`, and make small commits.
- Ask for approval at the end of each stage.
- Never push without approval, and never commit `data/` or `models/`.
- No force push, rebase, `reset --hard` or amend on pushed commits.
- Never fabricate numbers, citations or licences.
- Sample or chunk large files.
- Reusable code goes in `src/`; notebooks are for exploration and figures. Explain decisions in markdown so they can be defended in the viva.
