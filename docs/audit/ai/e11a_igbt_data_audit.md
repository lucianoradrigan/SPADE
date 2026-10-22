# E11a — IGBT data audit (round 2026-10-22, decision D3)

- **Command:** `python experiments/report_e11a_igbt_data_audit.py --archive data/upstream/fund/igbt/extracted/IGBTAgingData_04022009 --log-csv data/upstream/fund/igbt/device_aging_log`. The aging log was converted once with uv run --with xlrd --with pandas python scripts/oneoff/convert_igbt_aging_log.py (throwaway environment; xlrd not added to pyproject.toml).
- **Data:** the NASA PCoE archive `IGBTAgingData_04022009` (U.S. Government work, see DATA.md). It is not committed; the CSVs stay in `data/upstream/fund/igbt/`.
- **What was not done:** no model and no Prognostics page.

| Device | Protocol | End of test | Run to failure | Recorded hours | Transients (with a turn-off edge) | Source |
|---|---|---|---|---|---|---|
| DC-gate (20080429T135531) | DC gate 10 V; package held at 268-270 °C (stepped 220/240/270 °C), runaway cut-off 305 °C; low-speed data only | failed: latch-up and thermal runaway | yes | 4.19 | 0 (0) | DC folder Readme: 'Experiments stopped after latch-up and thermal runaway'; ~4 h |
| A17 (square gate) | square gate 10 V, 10 kHz, 40 % duty; package band 329-340 °C (Autotestcon 2008); low-speed + oscilloscope transients | failed: latch-up (recovered at room temperature) | yes | 2.86 | 418 (418) | Sonnenfeld, Goebel, Celaya, Autotestcon 2008, Fig. 8 and text: 'latch up … The experiment was ended after the failure and the IGBT was found to be functional when returned to room temperature'; the folder Readme says this file is that paper's experiment |
| Device 2 | square gate + SMU set-up; package set point stepped 100→280 °C and supply 2.5→6 V by hand within 20-40 min sessions (log) | no failure recorded | no | 1.11 | 175 (175) | Device Aging log: sessions 2 and 2b end with no failure note, then a low-temperature check (Device2check) |
| Device 3 | same as Device 2 | failed: latch-up | yes | 0.96 | 152 (152) | Device Aging log, session 3b, 04:02: 'end experiment…device latchup' |
| Device 4 | same as Device 2 | no failure recorded | no | 1.61 | 255 (255) | Device Aging log: sessions 4 and 4b, no failure note |
| Device 5 | same as Device 2 | failed: no longer able to draw current | yes | 0.66 | 106 (106) | Device Aging log, after 12:54: 'The device is no longer able to draw current.' |

## Reading

- **Run to failure:** 4 devices (DC-gate (20080429T135531), A17 (square gate), Device 3, Device 5), spread over 3 protocols:
  - DC gate: DC-gate (20080429T135531)
  - square gate, fixed 329-340 °C band: A17 (square gate)
  - square gate + SMU, stepped: Device 3, Device 5
- **On one comparable protocol:** at most **2** (Device 3, Device 5).
  - Their failure modes differ: latch-up for Device 3; loss of conduction for Device 5.
  - The steady-state data of that set-up are not scaled, and the collector current drifts by ~600 mA (folder Readme).
- **Turn-off current tail** (the precursor named by NASA for latch-up), computable from high-speed transients: A17 (square gate), Device 3, Device 5.
  - The DC-gate device has low-speed data only.
  - A17 ran a different thermal protocol (a fixed 329–340 °C band against a stepped 140→280 °C ramp) and recovered after latch-up.
  - The tail is a latch-up precursor, which is not Device 5's failure mode.
- **On-state V_CE:** not comparable across set-ups. The Device 2–5 steady-state voltages are unscaled.

## Conclusion

**E11 closes as a negative result.** Finding: "the NASA IGBT accelerated-aging archive supports at most 2 run-to-failure devices on comparable protocols; a validated RUL model is not feasible".

- **Looser reading, stated so it can be checked.** If protocol and failure mode are ignored, three run-to-failure devices have a computable turn-off tail (A17, Device 3, Device 5).
- **Why that pooled set is rejected.** Their times to failure come from different thermal schedules (hours at 329–340 °C against tens of minutes on a manual ramp), and one of them fails by another mechanism. A leave-one-device-out RUL model on that set would measure the protocol, not the degradation.
- **Option 3** (another NASA PCoE dataset, e.g. MOSFET thermal-overstress aging) stays **postponed**: not rejected, not done in this round.
