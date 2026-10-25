# SPADE — notes for Claude Code

Created 2026-10-25 (round PLECS/OpenModelica, task R1). Until then the repository had no CLAUDE.md; the working rules lived in the round instruction documents. This file keeps the rules that hold across rounds.

## Evidence rules

- Evidence goes under `docs/audit/<round>/` as `.md` + `.json`, with the command that produced each number. Anything not executed is written as "no ejecutado / not executed: <reason>".
- Negative results are deliverables. Findings get the next free ID (`S-…` in `docs/SIMULATION_STABILITY_REPORT.md`, `AI-N…` in `docs/AI_TECHNICAL_AUDIT.md`).
- Every xfail is `strict=True`, cites a finding, and is listed in `docs/audit/expansion/xfail_inventory.md` (`tests/test_xfail_inventory.py`).
- Pre-registered criteria are committed before any measurement and are not changed after results are seen; post-hoc analyses are labelled as such and never change a pre-registered verdict.
- Numerical-agreement GO/NO-GO criteria must be measured against a converged reference solution (fine solver with demonstrated convergence), not against another discretization. (Lesson G7, 2026-10-25, from B3 / S-16.)
