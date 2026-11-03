# F1.7 — catalog of SPADE for the assistant (`explain_spade`, `search_docs`)

- **Command:** `PYTHONPATH=src python -m pytest -q tests/assistant/test_assistant_catalog.py` (47 tests).
- **Code:** `configs/assistant_catalog.yaml` (17 pages, 5 plants, 5 controller types, 9 job kinds, one index entry referencing the 13 algorithm-guide families; plus `not_simulable`), `src/driveflow/assistant/catalog.py` (`explain_spade`, `search_docs`, `CATALOG_TOOLS`).
- **Sources of the texts:** the README page table, `docs/` and the code they name; nothing from general knowledge. Algorithm families are **referenced** (`algorithm_guide: [id]`), not copied: `explain_spade` returns the guide's own entries.
- **Coverage test:** every page of `viz/app.py PAGES`, every plant of `NATIVE_PLANTS`, every `controller_type` of `Scenario._VALID_PAIRS` and every job kind of `jobs.handlers.HANDLERS` has an entry; every algorithm-guide family is referenced; every `implemented_by` module imports; every `docs` file and `page` exists; every cited finding ID (S-07, S-20, S-21, AI-N13, AI-N24, AI-N26, AI-N27) is a row of the findings tables. Adding a page, plant, controller or job without documenting it fails the test.
- **"Not documented":** a topic with no entry returns `documented: false` with the instruction to say so (e.g. "quantum annealing"); "induction motor" also returns the `not_simulable` reason.
- **`search_docs`:** plain-text search over `docs/**/*.md` and the README, up to 5 snippets with file and section (e.g. "one-quadrant converter brake" finds `docs/SIMULATION_STABILITY_REPORT.md`).
- **Not executed:** answers by a real local model (F1.1 / F1.11).
