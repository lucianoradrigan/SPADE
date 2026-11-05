# F1.1 — comparison of local models (measured)

Command: `python experiments/report_f1_1_model_choice.py --models qwen3.5:9b --visibility state --think false --guards on --out-stem f1_1_round2_b_think_false --label (b) qwen3.5:9b think=false + guards --traces /tmp/claude-1000/-home-seven-post-doc-plataform-proposed/3b387af5-16b9-4a3f-a6b0-f810f07a05d0/scratchpad/f1_1_traces`. Host: Linux-7.0.0-38-generic-x86_64-with-glibc2.43; Ollama 0.40.13. Settings: {'temperature': 0, 'context_tokens': 16384, 'think': False, 'limits': {'max_tool_calls_per_turn': 12, 'max_experiments_per_launch': 50, 'max_total_sim_seconds': 600, 'max_model_calls_per_turn': 16, 'max_tokens_per_turn': 60000}, 'num_predict': 2048, 'guards': {'reject_repeats': True, 'per_turn_caps': {'search_docs': 2, 'explain_spade': 2}, 'final_answer': True, 'objective_needs_plant': True, 'nudge_draft': True, 'nudge_docs': True}}. Bench: [`f1_1_prompts.json`](f1_1_prompts.json) (12 prompts × 3 seeds × 2 visibilities, written before measuring).

| Model | Size | Quant. | License | VRAM loaded | Visibility | Valid calls | Correct tool (conf. / obj. / question) | Unconfirmed launch attempts / executed | Invented numbers refused / accepted | Latency median / p95 | tok/s | Errors / limit stops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen3.5:9b` | 9.0B (6.6 GB) | Q4_K_M | Apache-2.0 | 5.7 GiB | state | 100.0 % (126) | 66.7 % (83 / 67 / 33) | 0 / 0 | 0 / 0 | 9.4 / 31.1 s | 55 | 0 / 0 |

Digests: `qwen3.5:9b` 56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90.

## `qwen3.5:9b` per prompt (correct of 3)

| Prompt | state |
|---|---|
| c1 | 3 |
| c2 | 3 |
| c3 | 3 |
| c4 | 0 |
| c5 | 3 |
| c6 | 3 |
| o1 | 0 |
| o2 | 3 |
| o3 | 3 |
| q1 | 0 |
| q2 | 3 |
| q3 | 0 |
