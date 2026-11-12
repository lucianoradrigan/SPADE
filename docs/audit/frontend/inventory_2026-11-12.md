# Frontend inventory (2026-11-12, M6.1)

Read-only: no code was changed for this inventory. Compared against `ESPEC_FRONTEND_PHD_RADRIGAN_2026-11-02.md`, `ADENDA_FRONTEND_FASE1_2026-11-02.md` and V1 of `PLAN_CONSOLIDADO_2026-11-08.md` §3. App at `dev` `33e1e29`. Everything shown is **simulated**.

Page check: `python scripts/oneoff/frontend_inventory_pages.py http://127.0.0.1:8765 <out>`, Playwright with the system Chrome, `OLLAMA_HOST` closed. Raw: [`inventory_pages_2026-11-12.json`](inventory_pages_2026-11-12.json). Screenshots: `docs/screenshots/frontend/desktop_NN_<page>.png` (1440 px) and `narrow_NN_<page>.png` (390 px). The screenshots that show the avatar are covered by `licenses/avatar/LICENSE`.

## 1. Elements of the specification

**Summary:** 22 done, 10 partial, 8 missing, 1 not applicable (of 41).

| Area | Element | Specified in | Status | File | Screenshot | Note |
|---|---|---|---|---|---|---|
| Assistant | 11 avatar states | ESPEC §1.3; ADENDA FE-3 | **done** | `viz/components/radrigan_avatar/avatar.js (CLASS)` | `desktop_14_assistant.png` |  |
| Assistant | Event contract and event→state mapping with priority | ESPEC §2; FE-1 | **done** | `assistant/events.py; viz/adapters/assistant_adapter.py` | — |  |
| Assistant | Readable label per tool | FE-1.4 | **done** | `assistant_adapter.TOOL_LABELS` | — |  |
| Assistant | Three columns: avatar, chat, draft with Run | ESPEC §1.2; FE-4.1 | **done** | `viz/views/assistant.py` | `desktop_14_assistant.png` |  |
| Assistant | Figure sticky, bottom-anchored, overlapping the chat column by ~8 % | ESPEC §1.2 | **partial** | `viz/views/assistant.py` | `desktop_14_assistant.png` | inside its own column (spec fallback, fe3_spike.md (b)) |
| Assistant | Bubble tail toward the figure; last message at head height | ESPEC §1.2 | **missing** | `—` | `desktop_14_assistant.png` |  |
| Assistant | Status chip with text, aria-live=polite | ESPEC §1.3 | **done** | `avatar.html` | `desktop_14_assistant.png` |  |
| Assistant | Narrow screens (<900 px): compact avatar in the header | ESPEC §1.2 | **done** | `assistant.py NARROW_CSS` | `narrow_14_assistant.png` |  |
| Assistant | Narrow screens: draft as a tab under the chat | ESPEC §1.2 | **partial** | `assistant.py` | `narrow_14_assistant.png` | columns stack; draft is a stacked column, not a tab |
| Assistant | prefers-reduced-motion turns animations off | ESPEC §1.3 | **done** | `avatar.css @media; avatar.js` | — |  |
| Assistant | Animations pause when the tab is hidden | ESPEC §1.3 | **done** | `avatar.js visibilitychange` | — |  |
| Assistant | alt text of the figure | ESPEC §1.3 | **done** | `avatar.html` | — |  |
| Assistant | Chip contrast ≥ 4.5:1 in both themes | ESPEC §1.3, §7 | **done** | `avatar.css` | — | the component has its own dark panel: text 15.3:1, amber 10.5:1, tagline 6.9:1 |
| Assistant | Visible keyboard focus | M6.3 | **partial** | `theme/css.py :focus-visible` | — | SPADE elements have a 3 px focus ring; the avatar is not focusable (decorative); native widgets use Streamlit's |
| Assistant | A1 figure with alpha | ESPEC §1.1 | **partial** | `assets/assistant/base.webp` | — | provisional asset (FE-D2) |
| Assistant | A2 name as CSS neon text | ESPEC §1.1 | **done** | `avatar.css .rad-name` | — |  |
| Assistant | A3 head crop in the chat bubbles | ESPEC §1.1 | **missing** | `—` | `desktop_14_assistant.png` | chat bubbles use Streamlit's default icon |
| Assistant | A4 poses (hand on chin, paper, wink) | ESPEC §1.1 | **missing** | `—` | — | needs assets (FE-D2) |
| Assistant | A5 overlay layers (chip glow, scan line, data lines) | ESPEC §1.1 | **partial** | `avatar.css` | — | glow, ring and scan line in CSS; no data lines from the glasses |
| Assistant | A6 light-theme variant | ESPEC §1.1 | **missing** | `—` | — | known limitation of the provisional assets |
| Assistant | Tagline 'AI assistant · local model' | FE-D5 | **done** | `avatar.html` | `desktop_14_assistant.png` |  |
| Assistant | Tooltip with model and digest | ESPEC §1.4 | **partial** | `assistant.py caption` | `desktop_14_assistant.png` | tag and status shown; digest not shown |
| Assistant | Fixed first message; footer 'All results are SIMULATED.' | FE-4.3 | **done** | `assistant.py` | `desktop_14_assistant.png` |  |
| Assistant | No approval gesture on results; 'confirmed' only on the user's confirmation | FE-D6 | **done** | `avatar.css .s-results; tests` | — |  |
| Assistant | Job progress every 2 s | ESPEC §2 | **done** | `assistant.py _jobs_panel` | — |  |
| Assistant | Minimum 400 ms of 'thinking' | ESPEC §1.3 | **missing** | `—` | — | the adapter delegates it to the UI; the UI does not enforce it |
| Assistant | Visual streaming with a beat per few words | FE-4.2 | **done** | `assistant.py _turn` | — |  |
| Assistant | Fact cards with their source | CLAUDE.md assistant rules | **done** | `assistant.py _render_cards` | — |  |
| Assistant | proposal_card: experiments, assumptions, cost, Run / Edit | ESPEC §3 | **partial** | `assistant.py; assistant/cards.py` | — | cards and Run; no Edit button |
| Assistant | draft_panel: origin chips with colours and a legend | ESPEC §3, §5 | **partial** | `assistant.py _draft_panel` | `desktop_14_assistant.png` | origin as plain text column; no colours, no legend |
| Assistant | job_progress with a link to Experiment Runs | ESPEC §3 | **partial** | `assistant.py _jobs_panel` | — | progress bar, no link |
| Assistant | results_summary with KPIs, Plotly charts and SIMULATED | ESPEC §3 | **partial** | `assistant/cards.py` | — | KPIs in a card; no charts |
| Assistant | Visible response times | M6.3 | **missing** | `—` | — | not shown per turn |
| Assistant | Assets < 1 MB | ESPEC §7 | **done** | `assets/assistant + component` | — | 58 KB in total |
| Assistant | extraction_review (datasheets) | ESPEC §5 | **not applicable** | `—` | — | phase 2 |
| V1 | DC Motor: plant mode (v2/v1 with S-21 warning), load N·m, supply V, limits, nominals | PLAN V1 | **done** | `viz/views/dc_motor.py` | `desktop_03_dc_motor.png` | round B (90943b5) |
| V1 | DC Motor: motor parameters r_a, l_a, ψ_e | PLAN V1 | **done** | `dc_motor.py` | — |  |
| V1 | DC Motor: rotor inertia j_rotor | PLAN V1 (Scenario.motor_parameter_overrides) | **missing** | `—` | — | supported by Scenario, not exposed |
| V1 | PMSM: p, l_d, l_q, r_s, ψ_p, supply | PLAN V1 | **done** | `viz/views/pmsm.py` | `desktop_04_pmsm.png` |  |
| V1 | PMSM: rotor inertia j_rotor | PLAN V1 (Scenario.pmsm_motor_parameters) | **missing** | `—` | — | supported by Scenario, not exposed |
| V1 | Scenario validation (incl. S-24) as a clear message, not an exception | M6.2 | **done** | `dc_motor.py, pmsm.py st.error` | — |  |

## 2. Coherence across the 18 pages

| # | Page | Action run | SIMULATED | AI SUGGESTION label | Numeric controls without an explicit unit | Banned words | Narrow: horizontal scroll | Exceptions |
|---|---|---|---|---|---|---|---|---|
| 1 | Overview | — | yes | — | — | — | no | 0 |
| 2 | Experiment Runs | — | yes | — | — | — | no | 0 |
| 3 | DC Motor | Run simulation | yes | — | Setpoint weight b | — | no | 0 |
| 4 | PMSM / FOC | Generate PMSM cloud | yes | — | — | — | no | 0 |
| 5 | VSC / DPC | Generate DPC scenario | yes | — | Bootstrap refits | — | no | 0 |
| 6 | DC-DC Converters | Generate Buck scenario | yes | — | Primary inner shift D1, Secondary inner shift D2 | — | no | 0 |
| 7 | PLECS Plant Studio | — | yes | — | — | — | no | 0 |
| 8 | Tuning Studio | Start tuning | yes | — | Speed-loop damping (–), Setpoint weight b (–) | — | no | 0 |
| 9 | RL Lab | — | yes | — | — | — | no | 0 |
| 10 | AI Diagnostics | Generate | yes | yes | Levels, PI setpoint weight b, PI setpoint weight b, ψ_e factor, J factor, Simulation processes | — | no | 0 |
| 11 | Edge Deployment | Download ESP32 model (.tflite) | yes | — | — | live | no | 0 |
| 12 | Transfer Learning | Prepare dataset | **no** | — | Max macro-F1 drop vs. base, Frozen layers, Learning rate | — | no | 0 |
| 13 | Model Registry | — | **no** | see notes | — | hardware verified | no | 0 |
| 14 | Assistant | — | yes | — | — | — | no | 0 |
| 15 | Monitoring | Run monitoring | yes | see notes | — | — | no | 0 |
| 16 | Alert History | — | yes | — | — | — | no | 0 |
| 17 | Documentation | — | yes | see notes | — | live | no | 0 |
| 18 | Algorithm Guide | — | **no** | — | — | — | no | 0 |

**Reading of the flags (each one checked in context):**

- **Spanish:** none found in the visible text of any page.
- **Exceptions:** none; **horizontal scroll at 390 px:** none.
- **SIMULATED missing** on Transfer Learning (after *Prepare dataset*), Model Registry and Algorithm Guide.
  - The Algorithm Guide shows no results (not applicable).
  - **Transfer Learning** and **Model Registry** show model metrics trained and measured on simulated data without the label (**partial**).
- **AI SUGGESTION — REQUIRES VERIFICATION:**
  - exact on AI Diagnostics (and in DC-DC Converters' code);
  - **DC Motor** splits it: heading 'Fault classification — AI suggestion' + badge 'REQUIRES VERIFICATION', in a tab the check does not read;
  - **Monitoring** says 'AI suggestion … (not an alarm)' without 'REQUIRES VERIFICATION' (**partial**);
  - the 'prediction' hits on Model Registry and Documentation are metric names and explanations, not predictions.
- **Banned words:**
  - 'live' on Edge Deployment is a variable name in the generated C code, and on Documentation it explains a status reserved for real telemetry ('CONNECTED / LIVE — reserved for real telemetry'). Both are not claims.
  - **Model Registry** has a counter tile titled **'Hardware verified'** (value: the runs with deployment evidence; caption 'requires deployment evidence'). It claims nothing, but it uses the banned phrase: **to review** (decision for Luciano: rename, e.g. 'Board-measured').
- **Units:** the controls flagged are dimensionless (setpoint weight b, damping, inner shift D1/D2, ψ_e factor, levels, refits, frozen layers, learning rate). One lacks a unit it needs: **'Max macro-F1 drop vs. base'** (points).
- **Version in the sidebar:** the screenshots show 'v0.1.0'. This is the stale editable-install metadata of this machine's `SPADE-dev/.venv`; a fresh install reports 0.2.0 (`fresh_dev`). It is not a code defect. Fix locally with `uv pip install -e ".[dev,viz]"`.

## 3. What M6.2 and M6.3 take from this

- **M6.2 (V1, integrable):** rotor inertia `j_rotor` on DC Motor and PMSM. It is the only V1 capability missing; everything else of V1 was done in round B.
- **M6.3 (polish branch, not integrated):**
  - visible response time per turn;
  - the 400 ms minimum of `thinking`;
  - origin chips with colours and a legend in the draft panel;
  - the model digest in the header tooltip;
  - a link from job progress to Experiment Runs;
  - the draft as a tab on narrow screens.
- **Not taken (need assets or decisions):** A3 head crop, A4 poses, A6 light variant, bubble tail / overlap, proposal Edit button (it changes the assistant's flow), results charts, the 'Hardware verified' tile and the missing SIMULATED labels on other pages (outside M6.3's assistant scope).
