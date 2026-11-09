# Assistant benchmarks: how to run them by hand

The assistant's benchmarks call the real local LLM, so they need a GPU and Ollama. They are **not
part of CI** and are not run by `pytest`: the normal suite replays recorded answers (`ReplayClient`
cassettes) and needs no GPU or network. Tests that call the real model carry `@pytest.mark.llm` and
run only with `pytest --run-llm`.

Everything here is simulated, and no data leave the machine.

## Prerequisites

- Ollama serving at `127.0.0.1:11434`, with the pinned model pulled. Tag and digest are in
  `configs/assistant.yaml` (`qwen3.5:9b`, digest `56671c2a…3c90`):
  `ollama pull qwen3.5:9b`, then check the digest with `ollama show qwen3.5:9b --modelfile` or
  `curl -s 127.0.0.1:11434/api/tags`.
- The repository environment: `.venv/bin/python`, `PYTHONPATH=src:.`, run from the repository root.
- A free GPU. On the reference machine (RTX 5060 Ti, 16 GB) nothing else should be using it; the
  latency criteria are measured on that machine.
- No worker needed: the harness never executes a launch.

## F1.1: model choice (descriptive since round 2)

Bench: `docs/audit/assistant/f1_1_prompts.json` (12 one-turn requests). This bench was used to design
the fixes, so its results are **descriptive**, not an independent test.

```bash
python experiments/report_f1_1_model_choice.py --models qwen3.5:9b --visibility state \
    --think true --guards config --out-stem f1_1_round3 --label "round 3" \
    --traces <scratch dir>
```

Useful options: `--models` (several Ollama tags), `--seeds 0 1 2` (default),
`--visibility all state`, `--think true|false|config`, `--guards on|off|config`.
Output: `docs/audit/assistant/<out-stem>.{md,json}`; the raw traces go to `--traces` and are not
committed.

## F1.11, F1.12, F1.13: the pre-registered benchmarks

One script runs all three. Each bench has its criteria (`f1_1X_criteria.md`) and requests
(`f1_1X_requests.json`), committed before the run, under `docs/audit/assistant/`.

```bash
python experiments/report_f1_11_benchmark.py --bench f1_13            # f1_11 (default), f1_12, f1_13
python experiments/report_f1_11_benchmark.py --bench f1_13 --resume   # continue an interrupted run
python experiments/report_f1_11_benchmark.py --bench f1_13 --rescore  # re-score the raw file, no model calls
```

Other options: `--seeds 0 1 2` (default), `--only c01 q03 ...` (a subset of conversations).

**Time.** The sum of the turn latencies recorded in the raw files is about 52 min for F1.11, 54 min
for F1.12 and 47 min for F1.13 (168–173 turns each, 3 seeds); count about one hour per bench.
Launch a long run detached (for example `setsid nohup ... &`) and use `--resume` if it is cut.

**Output** (in `docs/audit/assistant/`):
- `f1_1X_benchmark_raw.json`: every turn, with its text, cards, tool calls and latency;
- `f1_1X_benchmark.json`: the automatic scores;
- `f1_1X_review.md`: the visible texts for the false-statement review.

**False statements.** This row is a review, not an automatic score. Since F1.13 it is done by a
separate reviewer that sees only the texts, the cards, the docs and the code, and its verdicts are
attached unedited (`f1_13_false_statement_review.json`). The verdict goes into `f1_1X_benchmark.md`.

**Rules.**
- The thresholds are committed before the requests are written.
- A bench whose results were used to design a fix is no longer an independent test: new requests are needed for the next one.
- Phase 1 closed with F1.13 (no F1.14).

## Other model checks

- `experiments/report_f1_2_llm_client.py`, `report_f1_3_loop_policy.py`, `report_f1_10_demo.py`:
  checks from the phase-1 tasks. See the header of each script for its command and output.
- `pytest --run-llm -m llm`: the tests that call the real model.
