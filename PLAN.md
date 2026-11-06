# Plan

Stages are approved one at a time. Later stages are added only after approval.

## Stage 0: Git setup. READY (awaiting push approval)
**Acceptance criteria**
- [x] Working tree clean (only the user's untracked `test.py` remains).
- [x] Old code archived with normal commits (`archive/`); no history rewrite, no force push.
- [x] Ignore rules verified with `git check-ignore` (`data/**`, `models/`, `.env`, `venv/`, large file types; `.gitkeep` files stay trackable).
- [x] Backup tag `backup-before-capstone-cleanup` and branches `archive/old-scope`, `archive/github-old-main` exist locally.
- [x] No tracked file over 50 MB; no secrets or data tracked.
- [x] Local git identity set for this repo only.
- [ ] Remote synced (`main`, both archive branches and the tag pushed): **pending user approval of the push**.
