# X2 — colour contrast (WCAG 2.1 AA), both themes

- **Command:** `pytest tests/test_contrast.py` (20 tests). Raw values: [`x2_contrast.json`](x2_contrast.json).
- **Sources:** tokens from `src/driveflow/viz/theme/tokens.py`.
- **Criteria:** normal text ≥ 4.5:1; essential UI elements ≥ 3:1.
- **Provenance labels covered:** SIMULATED, SIMULATED · PLECS, SIMULATED · SURROGATE, SIMULATED · OPENMODELICA (new, D5), EXTERNAL FILE, MEASURED · EXTERNAL DATASET.

## Finding and fix

**Before.** The components painted status and provenance tones (badge borders and glyphs, KPI rails, check glyphs, the motor ring) with the **dark** palette in both themes (`DARK[...]` in `page_header.badge_html`, `metric_cards`, `state_panel`, `motor_status`, `css.inject_base_css`). On the light theme's white surface those tones give:

| Tone | Dark tone on the light surface | Required |
|---|---|---|
| critical | 3.76:1 | 3:1 |
| healthy | 2.28:1 | 3:1 |
| primary | 2.14:1 | 3:1 |
| training | 2.72:1 | 3:1 |
| unknown | 2.56:1 | 3:1 |
| warning | 2.15:1 | 3:1 |

**Fix.** The components now take each tone from the active theme's palette (`tokens.active_palette()`, from `st.context.theme.type`). The light palette already existed and meets every criterion, so **no colour was changed**. The test `test_components_use_the_active_theme` fails on the old code.

**Not covered.** Two Plotly figures still draw with dark tones in both themes (`views/system_diagram.py` connector lines, `views/vsc_dpc.py` requirement tint). They are chart decoration, not text or state carriers; noted for a later pass.

## Results (minimum ratio over the four backgrounds)

| Theme | Pair | Min ratio | Required |
|---|---|---|---|
| dark | text on backgrounds | 10.37:1 | 4.5:1 |
| dark | text_muted on backgrounds | 5.29:1 | 4.5:1 |
| dark | primary as text (eyebrow) | 5.97:1 | 4.5:1 |
| dark | critical tone as UI element | 3.4:1 | 3.0:1 |
| dark | badge label on critical tint | 9.48:1 | 4.5:1 |
| dark | healthy tone as UI element | 5.61:1 | 3.0:1 |
| dark | badge label on healthy tint | 8.16:1 | 4.5:1 |
| dark | primary tone as UI element | 5.97:1 | 3.0:1 |
| dark | badge label on primary tint | 7.94:1 | 4.5:1 |
| dark | training tone as UI element | 4.7:1 | 3.0:1 |
| dark | badge label on training tint | 8.36:1 | 4.5:1 |
| dark | unknown tone as UI element | 4.99:1 | 3.0:1 |
| dark | badge label on unknown tint | 8.18:1 | 4.5:1 |
| dark | warning tone as UI element | 5.95:1 | 3.0:1 |
| dark | badge label on warning tint | 8.28:1 | 4.5:1 |
| light | text on backgrounds | 14.39:1 | 4.5:1 |
| light | text_muted on backgrounds | 5.56:1 | 4.5:1 |
| light | primary as text (eyebrow) | 5.06:1 | 4.5:1 |
| light | critical tone as UI element | 5.52:1 | 3.0:1 |
| light | badge label on critical tint | 11.46:1 | 4.5:1 |
| light | healthy tone as UI element | 4.28:1 | 3.0:1 |
| light | badge label on healthy tint | 12.1:1 | 4.5:1 |
| light | primary tone as UI element | 5.06:1 | 3.0:1 |
| light | badge label on primary tint | 11.82:1 | 4.5:1 |
| light | training tone as UI element | 6.06:1 | 3.0:1 |
| light | badge label on training tint | 11.51:1 | 4.5:1 |
| light | unknown tone as UI element | 4.06:1 | 3.0:1 |
| light | badge label on unknown tint | 12.28:1 | 4.5:1 |
| light | warning tone as UI element | 4.29:1 | 3.0:1 |
| light | badge label on warning tint | 11.98:1 | 4.5:1 |
