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
| Ollama tag / digest | `ollama list`, 2026-11-04 | `qwen3.5:9b`, Q4_K_M, 9.0 B, 6.6 GB, ID `56671c2ab938` (full digest in `f1_1_model_choice_qwen3.5_9b.json`) |

**Disk.**
- `qwen3:14b` (the weakest candidate of F1.1, Apache-2.0) was removed with `ollama rm` to make room; it can be pulled again.
- `ornith:9b` is kept until Luciano decides.

**Measurement.** The original F1.1 bench was run once on `qwen3.5:9b`, under the same conditions as the first two candidates: `f1_1_model_choice_qwen3.5_9b.md`. The original F1.1 report is unchanged.
