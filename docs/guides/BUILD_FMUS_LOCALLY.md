# Build the OpenModelica FMUs locally

SPADE does not distribute prebuilt FMUs.

**Why.** The licence review of the OpenModelica runtime that ships inside every exported FMU (`docs/audit/plecs_om/b8_license_review.md`, 2026-10-25) did not clear redistribution. Two points are open:
- some bundled solver sources (f2c-translated LAPACK, MINPACK) carry no licence header;
- the OpenModelica developers have said that part of the runtime's third-party code is LGPL.

`python -m driveflow.fmu fetch` therefore says so and exits. Build the FMUs on your own machine instead: the result is the same model, for your platform.

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
