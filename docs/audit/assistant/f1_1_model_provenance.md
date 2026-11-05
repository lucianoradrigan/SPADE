# F1.1 — provenance and license check of the chosen model (2026-11-04)

Luciano chose `ornith:9b` (tools by state) **on condition** that its publisher, base model and license be confirmed from the original source. Checked on 2026-11-04.

## `ornith:9b`: not confirmed

| Item | What the original source says | Confirmed? |
|---|---|---|
| Publisher | Hugging Face repository `ornith-ai/Ornith-1.0-9B`. The model card credits the "DeepReinforce Team" (blog deep-reinforce.com). Its own code samples and its `license_link` use another namespace, `deepreinforce-ai/Ornith-1.0-9B`. | **No:** inconsistent namespaces |
| Base model | Not declared: no `base_model` in the README front matter and no `_name_or_path` in `config.json`. The architecture is `Qwen3_5ForConditionalGeneration` (`model_type` `qwen3_5`). The model card says the family is "post-trained on top of Gemma 4 and Qwen 3.5" without saying which applies to the 9B. | **No** |
| License | Front matter `license: mit`; the model card says "MIT licensed". `license_link` points to `deepreinforce-ai/Ornith-1.0-9B/blob/main/LICENSE`, which returns **HTTP 404**. The repository has no LICENSE or NOTICE file. If the 9B derives from Qwen3.5-9B (Apache-2.0, as its architecture suggests), Apache-2.0 requires keeping its license and notices; none are present. | **No** |

**Decision rule (Luciano):** if it cannot be confirmed, stop and propose an official model of similar size. Therefore `ornith:9b` is **not** pinned in `configs/assistant.yaml`.

## Proposed alternative: `qwen3.5:9b` (official)

| Item | Source | Value |
|---|---|---|
| Publisher | Hugging Face `Qwen/Qwen3.5-9B` (Qwen Team) | Qwen Team |
| Base model | Model tree of `Qwen/Qwen3.5-9B` | `Qwen/Qwen3.5-9B-Base`; the repository is the post-trained model |
| License | License tag of `Qwen/Qwen3.5-9B` | Apache-2.0 |
| Tool calling | Model card ("excels in tool calling capabilities") and Ollama (`tools`) | Yes |
| Ollama tag / digest | `ollama list`, 2026-11-04 | `qwen3.5:9b`, Q4_K_M, 9.0 B, 6.6 GB, digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90` |

**Disk.**
- `qwen3:14b` (the weakest candidate of F1.1, Apache-2.0) was removed with `ollama rm` to make room; it can be pulled again.
- `ornith:9b` is kept until Luciano decides.

**Measurement.** The original F1.1 bench was run once on `qwen3.5:9b`, under the same conditions as the first two candidates: `f1_1_model_choice_qwen3.5_9b.md`. The original F1.1 report is unchanged.

### Result of the original F1.1 bench on `qwen3.5:9b` (same conditions as the first two candidates)

| Model | Visibility | Correct tool (conf. / obj. / question) | Valid calls | Turns ending with a non-empty answer | Limit stops / timeouts | Latency median / p95 |
|---|---|---|---|---|---|---|
| `qwen3:14b` | state | 33 % (67 / 0 / 0) | 92 % | 24 / 36 | 0 / 0 | 4.5 / 10.1 s |
| `ornith:9b` | state | 83 % (67 / 100 / 100) | 99 % | 18 / 36 | 15 / 3 | 8.3 / 190 s |
| `qwen3.5:9b` | state | 33 % (33 / 33 / 33) | 100 % | **36 / 36** | **0 / 0** | 3.6 / 12.5 s |

- All three: 0 launches without confirmation and 0 invented numbers accepted.
- **Reading of the `qwen3.5:9b` conversations (descriptive):** it always answers and never loops, but it rarely uses the tools.
  - On c1, c2 and c6 it sets the mode and then restates the request in prose, without building the draft.
  - On o1 and o3 it does not call the objective tools.
  - On q3 it answers from general knowledge, with no search and no citation.
  - On q2 it reads the search results as "S-21 parece ser una referencia a un conjunto de decisiones…". That is inaccurate: S-21 is a simulator finding, and it would count as a false statement under the F1.11 criteria.
- The renamed outputs are `f1_1_model_choice_qwen3.5_9b.{md,json}`. The script's `Path.with_suffix` cut the stem at "3.5", so the files were renamed after the run.

## Decision (Luciano, 2026-11-04): `ornith:9b` discarded

`ornith:9b` is not used because its provenance and license could not be confirmed from the original source:
- inconsistent publisher namespaces;
- the license link returns 404, and there is no LICENSE or NOTICE file;
- no declared base model, while the architecture suggests an Apache-2.0 base whose notices are absent.

It was uninstalled (`ollama rm ornith:9b`) on 2026-11-04. Its F1.1 numbers stay in `f1_1_model_choice.md` as measured. The assistant continues with the official `qwen3.5:9b` (round 2, `f1_1_model_choice_round2_rule.md`).
