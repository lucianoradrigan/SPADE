# GEM issue: published (decision C8, round C, T3)

- **URL:** https://github.com/upb-lea/gym-electric-motor/issues/292
- **Published:** 2026-11-09 18:24:54 UTC (15:24 local), from Luciano Radrigan's account (`lucianoradrigan`), with `gh issue create`. State at publication: open.
- **Text:** the "Description" section of [`gem_cont1qc_issue_draft.md`](gem_cont1qc_issue_draft.md) at commit `b627d1a`, verbatim. It includes:
  - the context of #138;
  - the full reproduction script with its exact output;
  - the three fix levels;
  - the verified note on `FiniteOneQuadrantConverter`.
- **Duplicate check, the same day,** with `gh issue list --repo upb-lea/gym-electric-motor --state all --search …`:

  | Query | Hits | Same problem? |
  |---|---|---|
  | "OneQuadrant" | none | — |
  | "one quadrant" | #83, a discrete-converter action scaling, closed 2020 | no |
  | "Cont1QC" | none | — |
  | "negative current" | #6 and #30 | no, unrelated |
- **Corrected before publishing:** the draft's short snippet printed only the final values, while the quoted output came from the full script. The issue now carries the full script.
  - The output's `u = 0.00 V` lines are explained: the current alternates sign step by step, and the printed steps start with a positive current.
  - This was checked by instrumenting `convert` on GEM `b474daf9`: in 20 steps, 8 start with a negative current, and all 8 get u = 1 (normalized).
- **Not done (decision C8):** the PR has not been opened and the fork `/mnt/datos/upstream/gym-electric-motor` has not been pushed. They wait for the maintainers' answer.
- **Follow-up:** `gh issue view 292 --repo upb-lea/gym-electric-motor --comments`.
