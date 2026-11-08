# DC Motor page: wall time of a typical run, v1 vs v2 (decision E3)

**Simulated.** Command: `python experiments/report_dc_motor_page_wall_time.py`. Raw: `dc_motor_page_wall_time.json`. Host: Linux-7.0.0-43-generic-x86_64-with-glibc2.43, Python 3.12.13.
Call: viz.adapters.simulation_adapter.run_dc (the DC Motor page's Run button). Page defaults (PI, speed mode, 300 rad/s, 1.0 s, seed 0, no noise, stock motor). 1 warm-up, then 5 timed calls per plant, alternating.

| Case | v1 median (min–max) | v2 median (min–max) | v2 / v1 | Final speed v1 / v2 (rad/s) | Min current v1 / v2 (A) |
|---|---|---|---|---|---|
| healthy_300rad_s_1s | 2.91 s (2.90–2.93) | 2.97 s (2.93–2.98) | 1.02 | 300.00 / 300.00 | 84.0 / 80.6 |
| outer_race_300rad_s_1s | 3.05 s (3.02–3.07) | 3.13 s (3.13–3.17) | 1.03 | 300.31 / 300.30 | 83.3 / 80.6 |
