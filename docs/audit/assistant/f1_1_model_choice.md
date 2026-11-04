# F1.1 — comparison of local models (measured)

Command: `python experiments/report_f1_1_model_choice.py --models qwen3:14b ornith:9b --traces /tmp/claude-1000/-home-seven-post-doc-plataform-proposed/3b387af5-16b9-4a3f-a6b0-f810f07a05d0/scratchpad/f1_1_traces`. Host: Linux-7.0.0-38-generic-x86_64-with-glibc2.43; Ollama 0.40.13. Settings: {'temperature': 0, 'context_tokens': 16384, 'think': False, 'limits': {'max_tool_calls_per_turn': 12, 'max_experiments_per_launch': 50, 'max_total_sim_seconds': 600, 'max_model_calls_per_turn': 16, 'max_tokens_per_turn': 60000}}. Bench: [`f1_1_prompts.json`](f1_1_prompts.json) (12 prompts × 3 seeds × 2 visibilities, written before measuring).

| Model | Size | Quant. | License | VRAM loaded | Visibility | Valid calls | Correct tool (conf. / obj. / question) | Unconfirmed launch attempts / executed | Invented numbers refused / accepted | Latency median / p95 | tok/s | Errors / limit stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen3:14b` | 14.8B (9.3 GB) | Q4_K_M | Apache-2.0 | 10.9 GiB | all | 97.4 % (114) | 33.3 % (67 / 0 / 0) | 6 / 0 | 0 / 0 | 3.4 / 11.9 s | 31 | 0 / 3 |
| `qwen3:14b` | 14.8B (9.3 GB) | Q4_K_M | Apache-2.0 | 10.9 GiB | state | 92.0 % (75) | 33.3 % (67 / 0 / 0) | 0 / 0 | 0 / 0 | 4.5 / 10.1 s | 35 | 0 / 0 |
| `ornith:9b` | 9.0B (5.6 GB) | Q4_K_M | MIT | 5.5 GiB | all | 100.0 % (195) | 66.7 % (83 / 33 / 67) | 0 / 0 | 0 / 0 | 8.1 / 14.2 s | 47 | 0 / 12 |
| `ornith:9b` | 9.0B (5.6 GB) | Q4_K_M | MIT | 5.5 GiB | state | 98.6 % (207) | 83.3 % (67 / 100 / 100) | 0 / 0 | 0 / 0 | 8.3 / 189.8 s | 49 | 3 / 15 |

Digests: `qwen3:14b` bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8; `ornith:9b` a75697c145891910e312c95e4a9fc1ccb8653e5ef543b23b0403a4665b82fd91.

## `qwen3:14b` per prompt (correct of 3)

| Prompt | all | state |
|---|---|---|
| c1 | 3 | 3 |
| c2 | 3 | 3 |
| c3 | 3 | 3 |
| c4 | 0 | 0 |
| c5 | 3 | 0 |
| c6 | 0 | 3 |
| o1 | 0 | 0 |
| o2 | 0 | 0 |
| o3 | 0 | 0 |
| q1 | 0 | 0 |
| q2 | 0 | 0 |
| q3 | 0 | 0 |

## `ornith:9b` per prompt (correct of 3)

| Prompt | all | state |
|---|---|---|
| c1 | 3 | 3 |
| c2 | 3 | 0 |
| c3 | 3 | 3 |
| c4 | 0 | 0 |
| c5 | 3 | 3 |
| c6 | 3 | 3 |
| o1 | 3 | 3 |
| o2 | 0 | 3 |
| o3 | 0 | 3 |
| q1 | 0 | 3 |
| q2 | 3 | 3 |
| q3 | 3 | 3 |

## Post-hoc reading of the traces (descriptive; not part of the pre-defined metrics)

| Model | Visibility | Turns ending with a non-empty answer | Empty final answer | Stopped by a limit | Runtime timeout |
|---|---|---|---|---|---|
| `qwen3:14b` | all | 18 / 36 | 15 | 3 | 0 |
| `qwen3:14b` | state | 24 / 36 | 12 | 0 | 0 |
| `ornith:9b` | all | 24 / 36 | 0 | 12 | 0 |
| `ornith:9b` | state | 18 / 36 | 0 | 15 | 3 |

- **"Correct tool" overstates both models:**
  - `ornith:9b` (state) calls `search_docs` correctly on the three questions, but repeats it until the token limit without answering.
  - `qwen3:14b` sets the right mode on the objective prompts but then answers in prose without calling any objective tool.
- **`qwen3:14b`:**
  - It rarely calls `set_mode` for questions; it answers directly instead.
  - It ends many configuration turns with an empty message after `update_draft`.
  - In "all" it tried `run_experiments` 6 times without a proposal; the policy refused every attempt.
  - One answer stated something false: "SPADE no tiene una documentación específica sobre las plantas" (q1, all, seed 0).
- **`ornith:9b`:**
  - It calls `update_objective` before choosing a plant (18 refusals) and passes `ops` to `propose_experiments` (3).
  - On o2 (state), Ollama did not answer within `timeout_s` = 180 s in all 3 seeds. That is the p95 of 190 s.
- **Fixed criteria:** 0 launches executed without confirmation and 0 invented numbers accepted, for both models.
- **Limits of this comparison:** two candidates only. The disk had 20 GB free before the downloads (6 GB after), so a third, e.g. `qwen3:8b` (5.2 GB), did not fit.
