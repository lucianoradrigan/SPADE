# FE-2 — avatar resources and their license

- **Command:** `PYTHONPATH=src python -m pytest -q tests/assistant/test_assistant_avatar_assets.py`; wheel check: `uv build --wheel -o <scratch>` and listing the `.webp` entries of the wheel.
- **Files:** `src/driveflow/viz/assets/assistant/base.webp` (34 908 B) and `hand.webp` (10 950 B), provisional (FE-D2), copied unchanged from the package Luciano supplied. Total 45.9 KB (< 1 MB). The original caricature (JPG) is not committed (test).
- **Packaging:** the wheel (`packages = ["src/driveflow"]`) includes them: `driveflow/viz/assets/assistant/base.webp`, `.../hand.webp` (and the existing `motor_base.webp`). The Windows installer stages a `git archive` of the commit, so it includes them too. No file list needed a change.
- **License (FE-D3):** `licenses/avatar/LICENSE` (all rights reserved, not MIT, redistribution only unmodified and as part of SPADE) and `licenses/avatar/README.md` (how to replace it). README, License section: one line pointing there. `LICENSE` (MIT) not modified.
- **Persona (`configs/assistant.yaml`):** `name: "PhD. Radrigan"`, `tagline: "AI assistant · local model"`, `avatar_dir: src/driveflow/viz/assets/assistant`. `assistant.config.avatar_assets()` returns the layers, or `ok: False` with the missing files and a logged warning (the UI then shows a neutral silhouette); it never raises (test).
