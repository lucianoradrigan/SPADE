# B4 — SPADE/OmBuck-v0 (FmuGymEnv over the BuckAveraged FMU)

Command: `python experiments/report_b4_fmu_env.py `  
Host: Linux-7.0.0-38-generic-x86_64-with-glibc2.43, Python 3.12.13.  
FMU: `/home/seven/post_doc/plataform_proposed/SPADE-dev/data/fmu/BuckAveraged.fmu`, SHA-256 `48d52b6ac081cd7aab19da6576e7a8d20941597b22db9f617c9c70d97e1d6320` (the one B3 evaluated: `48d52b6a…`).

- `gymnasium.utils.env_checker.check_env`: passed.
- Reset reproducibility (seed 11, 200 random actions, twice): bit-identical.

| Case | v_out NRMS vs SPADE/Buck-v0 (native PI actions) | i_L NRMS | ≤ 1 % (B3 criterion 4) | PI on the FMU env: final v_out | Step p50 / p99 |
|---|---|---|---|---|---|
| nominal 24 V / 6 ohm | 6.0e-06 | 1.1e-05 | yes | 24.000 V (native 24.000 V) | 0.025 / 0.044 ms |
| 12 V / 20 ohm | 6.5e-06 | 6.3e-06 | yes | 12.000 V (native 12.000 V) | 0.025 / 0.044 ms |
| 36 V / 3 ohm | 1.1e-05 | 3.0e-05 | yes | 36.000 V (native 36.000 V) | 0.025 / 0.047 ms |

Scope (decisions G1/G2): only BuckAveraged is registered. It replicates the native buck: it validates the FMU integration, not other physics. The FMU's load R is a fixed parameter, so it is set at reset and SPADE/OmBuck-v0 has no load step inside an episode.
Not executed: Windows and macOS (the FMU has linux64 binaries only).
