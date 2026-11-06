# Project instructions (DA487 capstone)

Project: *Explainable and Efficient Machine Learning for Multi-Horizon Solar Power Forecasting*.
Primary data: HKUST Dryad dataset in `data/hkust/`. Secondary: Kaggle HK-Rooftop-Microclimate-PV in `data/raw/`. See `DATA_SOURCES.md` and `README.md`.

## Rules
- Work stage by stage. Do not start the next stage (EDA, preprocessing, modelling) until the user approves.
- Keep it simple, clean and reproducible. The user has limited time and must defend choices in a viva: explain decisions briefly.
- Never load the full datasets into memory; sample large files. Data and models are never committed.
- Git: small commits with clear messages (`chore:`, `docs:`, `feat:`). Never push without explicit approval. Never force push, rebase, `reset --hard` or amend pushed commits.
- Chronological train/validation/test splits only (no shuffling). Verify facts from the files; do not assume.
- Log work in `WORKLOG.md` (date, hours, what was done).

## Layout
`data/` (git-ignored) · `notebooks/` · `src/` · `models/` (git-ignored) · `results/{figures,tables}` · `app/` · `report/` · `archive/` (old Kaggle-India scope, kept for reference).

## Environment
Python venv in `venv/` (git-ignored); `pip install -r requirements.txt`. Run notebooks from inside `notebooks/`.
