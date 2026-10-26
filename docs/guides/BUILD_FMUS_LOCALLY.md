# OpenModelica FMUs: download or build locally

## Download (Linux x86_64)

```bash
pip install -e ".[fmu]"
python -m driveflow.fmu fetch     # latest published release; or: --release fmu-v1 --model BuckAveraged
```

- `fetch` checks each FMU's SHA-256.
- An FMU built from another version of `modelica/SPADE/package.mo` is not installed unless you pass `--allow-mismatch`; that choice is recorded in `data/fmu/fetch_log.json`.
- The same download is in Plant Studio → Use as a Gymnasium plant → 2. Engine → "Download prebuilt FMUs", which runs it as a job.
- Releases carry only FMUs with a B3 GO verdict on their own build, with the licence texts of everything compiled into them (`licenses/fmu/`; review in `docs/audit/plecs_om/b8_license_review.md`).

## Build locally

For any other platform, or to rebuild from your own model version.

## Requirements

- Docker, able to pull `openmodelica/openmodelica:v1.27.1-minimal`. The image is pinned by digest in `scripts/fmu/OPENMODELICA_VERSION`. The first run also installs the Modelica Standard Library 4.1.0+maint.om into `data/fmu/openmodelica_home/`.
- Python with SPADE and the `fmu` extra: `pip install -e ".[fmu]"`. This installs fmpy, used to load the FMUs.
- Linux x86_64. That is the platform the build has been run on. Windows and macOS were not tried.

## Build

```bash
scripts/fmu/build_fmus.sh            # from the repository root
# a git worktree whose data/ lives elsewhere:
SPADE_DATA_DIR=../SPADE/data scripts/fmu/build_fmus.sh
```

**Output:**
- `data/fmu/{BuckAveraged,BuckSwitching,DcMotorAveraged,DcMotorSwitching}.fmu`;
- `data/fmu/manifest.json`, with the SHA-256 of each FMU and of `modelica/SPADE/package.mo`, the toolchain versions and the commit;
- the compiler log, `data/fmu/build/omc.log`.

Nothing under `data/` is committed.

### Model Exchange variants (optional, P1)

```bash
mkdir -p data/fmu/variants && scripts/fmu/om.sh scripts/fmu/build_fmu_variants.mos
```

This builds `BuckAveraged_fmi2_me.fmu` and `BuckSwitching_fmi2_me.fmu` in `data/fmu/variants/`. It exercises the Model Exchange path, which is how a PLECS Model Exchange export runs. The switching buck aborts in Co-Simulation (S-15) and runs in Model Exchange.

## Check

```bash
PYTHONPATH=src python -m pytest -m fmu -q -rxX
```

The two switching FMUs are expected to show as `XFAIL` (finding S-15: CVODE aborts at the PWM period). An `XPASS` on your build fails the run on purpose, so that the change gets documented.

## What you can use

| FMU | B3 verdict | Use in SPADE |
|---|---|---|
| BuckAveraged | GO | `SPADE/OmBuck-v0` (Environments, RL Lab, Tuning Studio) and Plant Studio → Use as a Gymnasium plant |
| DcMotorAveraged | NO-GO (criterion 4; see the B3 post-hoc analysis) | Reference use only in studies. Not publishable |
| BuckSwitching, DcMotorSwitching | NO-GO (S-15) | Not publishable |

The verdicts are those of `docs/audit/plecs_om/b3_om_fmu_spike.md`, for the builds evaluated there. Plant Studio re-checks each FMU you build before it can be published (stage 6, Validate & publish).
