# E12: Help / Algorithm Guide

Task: `TODO_EXPANSION_CLAUDE_CODE.md` E12. Spec: `docs/design/SPADE_GUI_Complete_Specification.md` §8.6.
Branch: `feat/E12-algorithm-guide` from `dev` (0342573, after E10a), rebased onto 48d6617 (E10b). Date: 2026-10-21. Machine: Linux x86-64, Python 3.12.13, Streamlit 1.66.0.

## Origin and license

- The idea comes from Sel. (`XinzeLee/AI_for_PE_Algorithm_Selector`): a What → Which → How assistant that goes from the life-cycle phase and the task to an algorithm family.
- That repository has no license, so **none of its code or text is copied**. I read only its public page (2026-10-21) for the structure and the paper it accompanies.
- The paper: X. Li, F. Lin, J. J. Rodríguez-Andina, S. Vazquez, H. A. Mantooth, L. G. Franquelo, "Fundamentals of Artificial Intelligence for Power Electronics," *IEEE Transactions on Industrial Electronics*, 2026.
  - The selector's page gives no DOI, so none is cited.
  - Sel. groups its data question by modality (tabular, signal, field…). The spec asks for **Data source**, so SPADE's third question is the data source: simulation, PLECS, or measured files.
- Every recommendation, with its why, assumptions and limitations, is SPADE's own text, written for what SPADE implements today.

## What was built

| File | Content |
|---|---|
| `configs/algorithm_guide.yaml` | The citation; 3 phases; 3 data sources, each with an assumption and a limitation added to every recommendation; 13 tasks, each with a family, why, assumptions, limitations, `page` (a registered page or `null`) and `accepts` (the data sources that page can work with). |
| `src/driveflow/viz/adapters/guide_adapter.py` | `load_guide()` validates the YAML on load. It checks required fields, unique ids, known phases and sources, that every `page` is registered in `viz/app.py` `PAGES`, and that a task with a page lists its sources. An invalid table fails loudly instead of rendering a broken link. It also provides `recommend(phase, task, source)` and `citation_text()`. |
| `src/driveflow/viz/views/algorithm_guide.py`, `viz/pages/algorithm_guide.py` | The page: three radio questions on the left; the recommended family, why, assumptions, limitations, and a `st.page_link` to the page or "Not available in this version" on the right; the citation. |
| `src/driveflow/viz/app.py` | "Algorithm Guide" added under Help. |

### When a recommendation has no page

- **4 tasks have no page in this version:**
  - controller tuning (E7);
  - circuit surrogate (E5);
  - model-free RL control (E8; the Environments page is linked from its own task, "Expose a plant as a training environment");
  - remaining useful life (E11).
- **The chosen data source is not accepted by the page.** No page reads PLECS data yet, and the simulation-only pages (DC Motor, VSC control, Monitoring, Environments, DC-DC) do not take measured files. These show "Not available in this version for circuit simulation (plecs)" or "… for measured data (your own files)".

## Acceptance

| Criterion | Evidence | Result |
|---|---|---|
| AppTest walks every combination without broken links | `tests/test_algorithm_guide_page.py::test_apptest_walks_every_combination`. It sets every phase × task × source (39 combinations) on the real page. Each one: no exception; the family shown is the table's; either a `page_link` whose route is that page and every link is a registered page, or the "Not available in this version" message. | PASS |
| Every recommendation resolves; every linked page file exists | `test_every_recommendation_resolves` (39 parametrized cases) | PASS |
| The table is valid and cites Li et al. (IEEE TIE, 2026) | `test_table_is_valid_and_cites_the_paper`; `test_validation_catches_broken_entries` (unregistered page, duplicate id, unknown phase); `test_no_page_reads_plecs_data_yet` | PASS |
| Page in the UI contract | `algorithm_guide` added to `tests/test_ui_contract.py` `PAGES` (renders, header, no LIVE) | PASS |

Screenshots in Chrome, dark and light (not committed).

## Tests

`tests/test_algorithm_guide_page.py`: 43 tests. Full suite on the branch after rebasing onto `dev` with E10b (48d6617): **949 passed, 0 failed** (688 s; 905 after E10b + 43 + 1 UI-contract case) — `e12_branch_pytest.txt`. Before the rebase: 934 passed on 0342573. Run from a second worktree of the repo with `PYTHONPATH=src` and the SPADE-dev virtual environment.

## Limits

- The guide is curated text, not a learned or validated recommender. It is a starting point, and the page says so (badge "GUIDANCE").
- When E5, E7, E8 or E11 add their pages, their `page` field and `accepts` must be filled in. The loader then checks that the page is registered.
