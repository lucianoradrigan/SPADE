# C1 (round C, T1, 2026-11-09): CI on every push to dev, and the manual runs on dev and release/v0.2.0

- **Trigger change:** `.github/workflows/tests.yml` now runs on `push` to `[main, dev]` (commit `5f1b1b3`). Pull requests to `main` and manual dispatch are unchanged.
- **Push of `release/v0.2.0`:** only that branch was pushed (`git push -u origin release/v0.2.0`), to run CI on it. **Nothing was pushed to `main`.**

| Run | Branch | Event | Commit | Jobs (pytest 3.11 / 3.12, smoke Ubuntu / Windows / macOS) | Result |
|---|---|---|---|---|---|
| [42672978993](https://github.com/lucianoradrigan/SPADE/actions/runs/42672978993) | dev | workflow_dispatch | `5f1b1b3` | 3 of 5 created, all 3 passed | failure (2 jobs never created) |
| [42672947040](https://github.com/lucianoradrigan/SPADE/actions/runs/42672947040) | dev | push | `5f1b1b3` | 2 of 5 created | failure (jobs never created) |
| [42673477575](https://github.com/lucianoradrigan/SPADE/actions/runs/42673477575) | release/v0.2.0 | workflow_dispatch | `9795509` | 3 of 5 created, all 3 passed | failure (2 jobs never created) |
| 42673760439 / 42673902500 | dev | push | `b627d1a` / `c484818` | 0 jobs ("workflow file issue") | failure (no job ran) |
| [42674141779](https://github.com/lucianoradrigan/SPADE/actions/runs/42674141779) | dev | push | `21caadc` | **5 of 5 passed** | **success** |
| [42674712266](https://github.com/lucianoradrigan/SPADE/actions/runs/42674712266) | dev | push | `3cb4232` | **5 of 5 passed** | **success** |
| [42682780248](https://github.com/lucianoradrigan/SPADE/actions/runs/42682780248) | **release/v0.2.0** | workflow_dispatch | **`9795509`** | **5 of 5 passed** | **success** |

## Reading

- **No job failed on its own steps.** Every job that started passed.
- **Why some runs were marked failure:** while about six runs were queued at once (5 jobs each), GitHub did not create some of their jobs, and those runs ended as "failure" with 0 jobs or only part of their jobs. Once the queue was short, complete runs passed: `dev` at `21caadc` and `3cb4232`, and `release/v0.2.0` at `9795509`.
- **The likely cause is the account's limit on concurrent jobs.** It was not confirmed: GitHub gave no message beyond "workflow file issue".
- **Practical rule:** avoid bursts of pushes to `dev`. A later push does not cancel the earlier runs; a `concurrency` group in `tests.yml` would, but that is not added here because it is a workflow-policy choice.

## Not executed

- A `concurrency: cancel-in-progress` setting in the workflow (a decision for Luciano).
