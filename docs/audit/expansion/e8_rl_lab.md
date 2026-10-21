# E8: RL Lab (DQN, DDPG in Keras) and benchmark

Round 2026-10-21, task T5 (decision D4: E8 does not depend on E7; no policy becomes a default). Branch `feat/E8-rl-lab` from `dev` (6ef9d13), worked in a parallel worktree. Machine: Linux x86-64, Python 3.12.13, TensorFlow 2.21 (CPU).

Benchmark report: [`e8_rl_benchmark.md`](e8_rl_benchmark.md) / `.json`.

## What was built

### Agents and benchmark (`control/rl/`)

- **`common.py`:**
  - `seed_everything` (Python, NumPy, Keras seeds and deterministic TF ops);
  - replay buffer, small MLPs;
  - `ActionDiscretizer`, the grid of action levels saved with a DQN policy;
  - `TrainingHistory`;
  - `run_episode`, which records the playback of an executed episode.
- **`dqn.py`:** a DQN reimplemented in Keras from Fund. `RL_buck_control.ipynb`.
  - Replay, a soft-updated target network, Huber loss, ε-greedy.
  - Absolute duty levels instead of Δduty, because SPADE's observation is Markov in the duty level, not in Δduty.
  - Only discrete actions; a Box action needs an explicit discretiser.
- **`ddpg.py`:** a DDPG reimplemented in Keras from Fund. `DDPG_buck_control.ipynb`.
  - Actor with tanh output scaled to the action bounds, critic Q(s, a), soft target updates, Ornstein-Uhlenbeck noise, random warm-up.
  - The duty-smoothness penalty is SPADE's `action_rate` reward term, not hard-coded.
- **`policy_controller.py`:** `PolicyController(policy)` runs a saved policy with a numpy forward pass, with the `reset()`/`control()` shape of the other controllers. Its input is the observation; the docstring explains the difference.
- **`benchmark.py`:** every controller runs through the same environment and the same grid:
  - buck: the PI cascade on 4 reference/load/load-step conditions;
  - VSC: DPC v3 and v4 at 50 V on the stable loads of `measure_dpc_rmse_by_r_range` (R ≥ 4 Ω).
  - Metrics: episode reward, steady-state tracking RMSE, peak current, termination.

### Job

`rl_training` (in `jobs/handlers.py`):
- validates the environment, the algorithm, the discretisation for DQN, the episodes and the reward terms;
- reports progress per episode and supports cancellation;
- writes a checkpoint (`checkpoints/policy_epNNNN.json`) and an evaluation playback (`playback/eval_epNNNN.json`) at every evaluation, then `policy.json`, `history.json` and `benchmark.json`;
- the status is always "policy candidate -- not a default".

### GUI: `Design & Control / RL Lab` (spec §5)

1. **Environment:**
   - registered IDs, with the contract and a Python snippet;
   - DDPG for continuous actions;
   - DQN only after choosing a discretisation (levels per dimension, saved with the policy).
2. **Reward:**
   - **RewardEditor** component (`viz/components/reward_editor.py`): enable, weight and threshold per term, with formula and signal, validated by `envs.reward.parse_terms`.
   - **"Preview on baseline episode"** runs one real episode of the plant's classical controller with those terms and shows the per-term contributions. It is recorded as an evaluation run. No curve exists before that.
3. **Training:**
   - episodes, seed, evaluation/checkpoint interval; Start training (a job);
   - curves of episode reward and greedy evaluation reward, critic/actor loss, and the constraint-violation rate;
   - **Playback actual evaluation episode**: the job's recorded traces, selected by episode.
4. **Benchmark:**
   - the same grid for every controller, with per-condition rows including terminations;
   - a warning when RL is worse;
   - **Save policy candidate**: stores a file next to the run and deploys nothing;
   - **Use as default** through ApprovalGate, always blocked by D4 (and by "loses on the grid" when it does).

ApprovalGate already existed (E0, `viz/components/approval_gate.py`) and was reused.

## Results (summary of `e8_rl_benchmark.md`)

Five jobs of 100 episodes, run in parallel: buck DDPG with seeds 0/1/2, buck DQN with 21 duty levels, VSC DDPG.

| Plant | Controller | Median tracking RMSE, 2nd half | Mean episode reward |
|---|---|---|---|
| Buck | PI | **0.19 V** | −5.03 |
| Buck | DQN (seed 0) | 0.49 V | **−3.10** |
| Buck | DDPG, seeds 0 / 1 / 2 | 2.25 / 0.93 / 4.62 V | −6.12 / −4.66 / −10.62 |
| VSC | DPC v4 | **2.8 V** | **−0.82** |
| VSC | DPC v3 | 13.2 V | −15.0 |
| VSC | DDPG (seed 0) | 21.5 V | −169.0 |

**What this shows:**
- **RL loses on steady-state tracking everywhere.**
- **On the buck, DQN and DDPG seed 1 score a better reward than the PI.** The reward includes start-up and action rate; this was not decomposed. Both numbers are reported, and a better reward is not presented as a better regulator.
- **DDPG is very seed-sensitive.**
- No condition terminated.

## Acceptance

| Criterion | Evidence | Result |
|---|---|---|
| DQN and DDPG in Keras, seeded, with checkpoints, as E0 jobs; DQN discrete only, with the transformation stored | `tests/control/test_rl_agents.py` (9 tests); `test_dqn_needs_a_discretizer_in_the_job_spec`; `test_training_job_writes_checkpoints_playback_and_benchmark` (slow) | PASS |
| Reproducibility: same seed, same curve, within a declared tolerance | `test_same_seed_same_curve[dqn, ddpg]`: episode and evaluation rewards equal to **1e-9** (declared `REPRO_TOL`) on CPU with deterministic TF ops; not checked across machines | PASS |
| PolicyController with the controller interface | `test_policy_controller_matches_the_agent[dqn, ddpg]` (numpy forward equals the Keras policy to 1e-6) | PASS |
| GUI: environment → reward (preview only after a real episode) → training (reward, losses, violations, playback of executed episodes) → benchmark on the same grid | `tests/test_rl_lab_page.py` (3 AppTests: empty states and preview, DQN discretisation, a real job through the worker with every chart, the benchmark table and the blocked default); `rl_lab` in `tests/test_ui_contract.py` | PASS |
| Cases: buck vs. PI; VSC vs. DPC v3 and v4 on the `measure_dpc_rmse_by_r_range` grid | `e8_rl_benchmark.md` | PASS (RL loses; published) |
| No policy as default | ApprovalGate blocked; the backend refuses blocked approvals; saved file `status: policy candidate -- not a default` | PASS |

Full suite (rebased on 5288a76): **1063 passed, 4 xfailed (strict, findings), 0 failed** (830 s) — `e8_branch_pytest.txt`. Run from a second worktree with `PYTHONPATH=src` and the SPADE-dev virtual environment.

## Limits

- 100 episodes per run, one seed for DQN and for the VSC, so the budget is small. Longer training and hyper-parameter search were not tried.
- The reward weights are the environment defaults and were not tuned.
- Simulated plants only.
