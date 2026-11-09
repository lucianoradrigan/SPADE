# Golden set of phase 2: downloaded, catalogued, read by hand (round C, T8, decision C10)

**Phase 2 has not started.** No extractor and no LLM was run on these files. "Used" means that Claude Code read each document or photo itself and wrote the expected values by hand.

- **Raw values:** [`golden_set.json`](golden_set.json).
- **Catalogue:** `tests/assistant/golden/manifest.yaml` (committed).
- **Expected values:** `tests/assistant/golden/expected/<id>.json` (committed).
- **The third-party files** are in `data/upstream/assistant_golden/` and are **never committed**. The exception is the 8 photos under CC0 or CC BY, which are in `tests/assistant/golden/photos/` with `ATTRIBUTION.md`.
- **Commands:** `python experiments/fetch_golden_datasheets.py`, `python experiments/fetch_golden_photos.py search` and `… download <titles>`, `python experiments/fetch_golden_ingenium.py 1992.2220.001 1992.2292.001`, `python experiments/report_golden_set.py`.

## Documents (annex A)

**17 of 21 downloaded** on 2026-11-09, each with its URL, date, Content-Type and SHA-256 in the manifest. **13** have hand-written expected values: D1, D2, D3, D5, P1, P2, P3, P5, P6, P7, P9, P10, I1.

| ID | Manufacturer (as printed) | Type | Style | Status | Expected | Note / reason |
|---|---|---|---|---|---|---|
| D1 ★ | perm-motor (Heinzmann PMG 132 in annex A; 'Heinzmann' not pr | PM DC | A | downloaded | yes |  |
| D2 ★ | Parvalux | PM DC | C | downloaded | yes |  |
| D3 ★ | Motenergy | PM DC | D | downloaded | yes | the title reads ME1513 |
| D4 | Bodine Electric | PM DC | D | downloaded | no | web page, printed to PDF |
| D5 ★ | Portescap | PM DC (coreless) | A | downloaded | yes | DEVIATION: one catalog page (16N78, 6 windings) instead of the ~120-variant catalog of annex A |
| D6 ★ | not named in the document (annex A: do not invent it) | PM DC | C | unavailable | no | HTTPError: HTTP Error 403: Forbidden |
| D7 | Portescap | PM DC (coreless) | A | unavailable | no | not a PDF (Content-Type 'text/html; charset=UTF-8', starts b'<!DOCTYPE html>\r\n<html lang="en">\r\n<head'); n |
| P1 ★ | Siemens | PMSM servo | C | downloaded | yes |  |
| P2 ★ | Kollmorgen | PMSM servo | A | downloaded | yes |  |
| P3 ★ | Industrial Devices Corporation (annex A: Kollmorgen B23) | PMSM servo | C | downloaded | yes |  |
| P4 ★ | Delta Electronics | PMSM servo (ECM-B3) | A | unavailable | no | TimeoutError: The read operation timed out; HTTPError: HTTP Error 403: Forbidden; URLError: <urlopen error [Er |
| P5 ★ | Nanotec | BLDC | C | downloaded | yes |  |
| P6 | Nanotec | BLDC | C | downloaded | yes | expected values written although not starred (to reach >= 12) |
| P7 ★ | EMRAX | axial-flux PMSM | A | downloaded | yes |  |
| P8 | maxon | BLDC | A | downloaded | no |  |
| P9 | IMTT (annex A: TI BLDC17F; hosted on TI E2E) | BLDC | B | downloaded | yes | expected values written although not starred (to reach >= 12) |
| P10 ★ | Bosch Rexroth (reseller sheet, IBA-Retzer GbR) | PMSM servo | D | downloaded | yes |  |
| P11 | Yaskawa | PMSM servo | D | downloaded | no | web page, printed to PDF |
| I1 ★ | WEG | induction (squirrel cage) | E | downloaded | yes | WEG: reprinting not allowed without authorisation |
| I2 | Baldor (ABB) | induction (inverter duty) | E | unavailable | no | TimeoutError: The read operation timed out |
| R1 | Kollmorgen | reference (not extracted) | - | downloaded | no | source of configs/assistant_conversions.yaml |

**Unavailable:**
- **D6 and P4:** both starred. Each URL refused the automated download (HTTP 403) or timed out. The download was not retried with a browser identity: that would bypass the site's refusal.
- **D7 and I2:** D7 has only third-party viewer pages, no PDF; I2 timed out.
- **To add D6 or P4,** Luciano can save the file by hand into `data/upstream/assistant_golden/<id>/document.pdf`. The script then records its hash.

**Declared deviations:**
- **D5** is one page of the 2024 Portescap catalog (16N78, 6 windings), not the ~120-variant catalog of annex A, which was found only in DirectIndustry viewers.
- **P6 and P9** are not starred; their expected values were written so that, without D6 and P4, there are still at least 12 documents with expected values.
- **D5, P2 and P7** have many variants. Two variants were chosen before reading the values: the first and the last column (or row).

**Counts:**
- by type: {'PM DC': 5, 'PM DC (coreless)': 2, 'PMSM servo': 5, 'PMSM servo (ECM-B3)': 1, 'BLDC': 4, 'axial-flux PMSM': 1, 'induction (squirrel cage)': 1, 'induction (inverter duty)': 1, 'reference (not extracted)': 1};
- by style: {'A': 7, 'C': 6, 'D': 4, 'B': 1, 'E': 2, '-': 1}.

## Photos of rating plates

- **Search:** Wikimedia Commons API. There are no "motor nameplate" categories; Commons files them under "manufacturing tablets". The licence of each file was read from its own information page (`extmetadata`).
- **23 records:**
  - 21 Commons photos: 8 under CC0 or CC BY, committed and resized to at most 1600 px; 13 under CC BY-SA, in `data/upstream/`;
  - 2 Ingenium records (non-commercial licence, `data/upstream/` only).
- **P5 not triggered:** 21 photos have an open licence, against the threshold of 10.

| ID | Machine (as read on the plate) | Licence | Where |
|---|---|---|---|
| W01 | three-phase induction motor | CC0 | tests/ (committed) |
| W02 | AC/universal brushed motor (Electrolux, brushes visible); type not pri | CC BY 2.0 | tests/ (committed) |
| W03 | three-phase synchronous hydro-generator | CC0 | tests/ (committed) |
| W04 | generator (hydro power station) | CC0 | tests/ (committed) |
| W05 | generator (hydro power station) | CC0 | tests/ (committed) |
| W06 | machine type not printed (museum label: generator) | CC BY 3.0 | tests/ (committed) |
| W07 | machine type not printed (museum label: DC generator) | CC BY 3.0 | tests/ (committed) |
| W08 | shunt-wound machine (DC by its winding type) | CC BY 3.0 | tests/ (committed) |
| W09 | asynchronous (induction) machine | CC BY-SA 2.0 de | data/upstream/ |
| W10 | six-phase motor with excitation (synchronous by its 'Erregung' line);  | CC BY-SA 2.0 de | data/upstream/ |
| W11 | not stated | CC BY-SA 2.0 fr | data/upstream/ |
| W12 | not stated | CC BY-SA 3.0 | data/upstream/ |
| W13 | three-phase induction motor | CC BY-SA 4.0 | data/upstream/ |
| W14 | three-phase motor (3 ~Mot; IE3) | CC BY-SA 4.0 | data/upstream/ |
| W15 | three-phase motor (D Mot) | CC BY-SA 4.0 | data/upstream/ |
| W16 | NOT an electric machine: a 1906 Diesel engine ('Wärme-Motor Patent Die | CC BY-SA 4.0 | data/upstream/ |
| W17 | DC motor (Marine D.C. Motor) | CC BY-SA 4.0 | data/upstream/ |
| W18 | traction three-phase asynchronous (induction) motor | CC BY-SA 4.0 | data/upstream/ |
| W19 | three-phase asynchronous motor (Moteur asynchrone triphasé) | CC BY-SA 4.0 | data/upstream/ |
| W20 | three-phase generator (3 ~ GEN.) | CC BY-SA 4.0 | data/upstream/ |
| W21 | machine type not printed (file title: exciter) | CC BY-SA 4.0 | data/upstream/ |
| IG1 | AC induction motor, 10 HP, General Electric Co., 1901 | CC BY-NC 4.0 (non-commercial, per the record page) | data/upstream/ |
| IG2 | DC bipolar motor, 7.5 HP, Crocker-Wheeler, 1891 | CC BY-NC 4.0 (non-commercial, per the record page) | data/upstream/ |

**Variety:**
- **Machines:** DC machines (Westinghouse marine DC motor, Lancashire shunt-wound, Greenwood & Batley, an AEG exciter), induction motors (Malmbergs, ABB, TECO, VEM, AEG, Marelli, a ZXTM traction motor), synchronous and other generators, and a universal motor.
- **Traps:**
  - two plates that are not electric-machine ratings: W16 is a 1906 Diesel engine and W12 a property tag;
  - illegible plates: W04, W06, W13, W19;
  - Cyrillic (W03), Chinese (W18), German and French labels;
  - decimal commas and an apostrophe as decimal mark (W21);
  - internal inconsistencies: W20 is 115 kW or 115 kVA, W05's current does not match its voltage, W15 prints Y/Δ 220/380.
- **No open-licence servo or PMSM plate was found.**
- **Ingenium:** the official transcription of each record is in the expected file. The public record photo (750 px) does not let the plate's numbers be read, so they are marked "illegible". This is the difference noted against the official text.

## Conventions (Kollmorgen WP RevD, R1): ratio checks on P1-P5

Rules and pages: `configs/assistant_conversions.yaml`, and a new dated section of `docs/assistant/RELEVAMIENTO_DATASHEETS_2026-11-02.md`.

| Doc | Variant | Ke printed | Kt printed | Ratio | Fits |
|---|---|---|---|---|---|
| P1 | — | 89.0 V/1000*min¯¹ | 1.40 Nm/A | 63.57 | kb_over_kt_sine_rms |
| P2 | AKM62H | 142 V/krpm | 2.2 Nm/Arms | 64.55 | kb_over_kt_sine_rms |
| P2 | AKM65P | 80.5 V/krpm | 1.3 Nm/Arms | 61.92 | kb_over_kt_sine_rms |
| P3 | — | 45.5 Vp-p/kRPM | 57.6 [0.41] oz-in/A [N-m/A] | 110.98 | ke_over_kt_six_step |
| P4 | — | — | — | — | not executed: document unavailable (no expected values) |
| P5 | — | 3.7 Vrms/krpm | 0.044 Nm/A | 84.09 | none |

- **P1:** the ratio fits RMS, so Siemens' kE is already line-to-line RMS. The earlier reading of the survey, which multiplied it by √3, is corrected in its new section.
- **P5:** no convention fits, so the agent must ask.
- **The test does not fail on these values;** it only reports them (`tests/assistant/test_golden_set.py::test_the_ratio_checks_report_without_failing`).

## Not done

- **No model or extractor was run** on the golden set (phase 2 has not started).
- **D6, D7, P4 and I2** are unavailable (see above). No servo or PMSM plate photo under an open licence was found.
- **Luciano's own photos (D6):** not added yet. They do not exist yet.
