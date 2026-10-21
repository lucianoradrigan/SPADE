# T7 (E11): IGBT RUL with the NASA dataset — stopped at the feasibility check

Round 2026-10-21, task T7 (decision D9). Date 2026-10-21. Branch `feat/E11-igbt-rul`.

## Done

1. **License and provenance checked first, as D9 requires.** The NASA catalogue marks the dataset as a U.S. Government work (https://www.usa.gov/government-works), with public access. Documented in `DATA.md`, with the URL, size, SHA-256 and citation. The original NASA archive was used, not Fund.'s `.mat` (CC BY-NC 4.0).
2. **Downloaded** to `data/upstream/fund/igbt/` (not committed), extracted, and every `.mat` file inspected (`scipy.io.loadmat`, data only).

## Why I stopped

T7 asks for a probabilistic RUL model (Gaussian BiLSTM, NLL), **split by device or series**, with the **empirical coverage of the 90 % interval** reported. That needs several devices that were aged to failure, with a comparable aging indicator. The archive does not provide that.

| Device | Protocol and measurements | Usable for a common RUL target? |
|---|---|---|
| DC-gate device (`20080429T135531.mat`) | Continuous slow signals (V_CE, I_C, V_G, temperatures), 301,680 samples over 4.2 h. The Readme says it failed by latch-up; the precursor is the collector-current tail at turn-off. | Yes for itself, with its own protocol |
| A17 (`april22nd-23rd…A17.mat`) | Square-wave gate; 41,752 steady-state records and 418 transients over 2.9 h; the device of the Autotestcon paper | Probably, but with a different protocol from the DC device |
| Devices 2–5 (two to three files each, 0.4–0.9 h per file) | Square-wave gate. The Readme says transients are missing, the collector current drifts by ~600 mA, and the steady-state data "are not scaled properly". Sensor values seen: heat sink 718, internal temperature −2254, epoch dates in 1903. | Not without the aging log (failure times, how the files relate). The devices were sent to Padova, so they may not have been aged to failure. |

**At most two run-to-failure devices, measured with different protocols.** A by-device split leaves one device to train and one to test. The 90 % coverage would rest on a single device's windows. Any number would be dominated by which device is held out, and the GUI would have to say "Coverage not validated" anyway.

Fund.'s notebook works on one device. A split in time within one unit is exactly what D9 forbids ("sin ventanas de una misma unidad en entrenamiento y prueba").

## Proposed split (needs your decision)

1. **E11a, data audit (small).**
   - Read `Device Aging log.xls`. This needs `xlrd`, a new dependency (not added), or a one-off manual conversion to CSV.
   - Settle which devices ran to failure, their failure times, and a precursor that can be computed under each protocol (collector-current tail at turn-off; on-state V_CE).
   - Result: a table of usable devices.
2. **E11b, the model, only if E11a finds ≥ 3 run-to-failure devices with a common precursor.**
   - BiLSTM-Gaussian in Keras (YAML via `train_model.py`), leave-one-device-out, coverage per held-out device.
   - GUI `Intelligence / Prognostics` labelled MEASURED · EXTERNAL DATASET.
3. **Alternative.** Use the other NASA PCoE power-semiconductor aging data (e.g. MOSFET thermal-overstress aging), which has more devices on one protocol. That needs its own license check and your approval of the change of dataset.

Nothing was trained, so there is no `igbt_rul.{md,json}` and no model run. **No ejecutado: fuera del alcance verificable con este dataset.**
