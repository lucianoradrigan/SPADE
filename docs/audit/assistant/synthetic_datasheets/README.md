# Synthetic motor datasheets (B6, 2026-11-16)

Fictitious motors from fictitious manufacturers, each sheet marked "SYNTHETIC SHEET … Not a real product". **No LLM and no extractor was run on them** (instructions 2026-11-16). They exist so that a future Phase 2 iteration can be measured on documents whose truth is known exactly, without touching the golden set or its held-out third.

- **Generator:** `experiments/generate_synthetic_datasheets.py`, seed **20261116**, 30 sheets: 5 per style A–F (styles and traps from `docs/assistant/RELEVAMIENTO_DATASHEETS_2026-11-02.md` §2–§3).
- **Command:** `python experiments/generate_synthetic_datasheets.py --pdf-python <python with playwright>` → `data/synthetic_datasheets/` (outside git): `S01…S30.html`, `.pdf` (Chromium print), `expected/S*.json`, `trap_log.json`.
- **Expected files:** schema `golden-expected/1` (as the golden set), `must_ask` as `{field, reason, item, text}` with the closed reason list of decision D8, plus `traps` and `truth_si` (the exact SI values). A perfect output scores perfectly with `experiments/score_golden_extraction.py` (`tests/assistant/test_synthetic_datasheets.py`).
- **Traps injected and logged** (counts over the 30 sheets, `trap_log.json`): missing inductance 11, multiple ratings 10, continuous vs peak 10, curve-only value 10, R at two temperatures 9, Kt–Ke inconsistent (+12 %) 9, wrong label ("Thermal resistance (Ra)") 7, Ke units (mV/rpm, V/rpm, rpm/V) 7, unitless inertia 6, decimal comma 5, US units (oz-in, lb-ft²) 5, manufacturer unnamed 5, Kt missing 4, Ke rms/peak undeclared 4, R line-line 2.
- **Questions expected** (`must_ask` reasons): missing 45, variant_unspecified 20, ambiguous_meaning 13, inconsistent 9, manufacturer_unnamed 5, ambiguous_waveform 4. Not covered yet: ambiguous_basis, condition_unspecified, machine_type_unspecified.
- **Examples committed here:** `S01` (style A, DC, variants in columns), `S05` (style E, PMSM, catalogue page with a curve), `S06` (style F, DC rating plate): HTML, PDF and expected file each.

Limits: the "photo" of style F is an HTML plate with a slight blur and rotation, not a photograph; the sheets are much cleaner than real scans. All values are fictitious.
