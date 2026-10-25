# B7 — Weekly CI for the OpenModelica FMUs

## What changed

**Per-commit CI (`tests.yml`): unchanged.**
- It installs `.[dev,viz]`, without fmpy. Every `fmu` test skips (the tests call `pytest.importorskip("fmpy")` or check `fmu_env.availability()`), so the suite stays green without OpenModelica.
- The tests that check the "no FMU / no fmpy" behaviour have no `fmu` marker and run there:
  - `tests/envs/test_fmu_env.py::test_without_the_fmu_the_environment_is_not_offered`;
  - `tests/fmu/test_plant_studio_fmu.py::test_without_fmpy…` and `…without_fmus…`;
  - `tests/test_b6_capabilities.py`.

**New job `fmu` in `validation-scheduled.yml`: weekly (Monday 04:00 UTC) and `workflow_dispatch`.** Linux only. Steps:
1. `pip install -e ".[dev,viz,fmu]"`. The `viz` extra is needed by the Plant Studio AppTests marked `fmu` (B5).
2. `bash scripts/fmu/build_fmus.sh`. It builds the four FMUs in `openmodelica/openmodelica:v1.27.1-minimal@sha256:ac9a0223…`, the image pinned by digest in `scripts/fmu/OPENMODELICA_VERSION`.
3. `pytest -m fmu -v -rxX`, with JUnit XML and a text log.
4. `python experiments/report_b3_om_fmu_spike.py --out ci_out/fmu/b3`, on the fresh build. It writes to `ci_out/`, never to `docs/`.
5. It uploads the artifact `fmu-evidence`: the pytest log and XML, the B3 JSON and figures, `manifest.json` (SHA-256 of each FMU and of the `.mo`) and `omc.log`.

**S-15.**
- The switching FMUs are built and run like the others.
- Their strict xfail (`tests/fmu/test_om_fmus.py::test_switching_fmus_simulate_10_ms_at_the_pwm_period`) keeps documenting the CVODE abort.
- Because the outcome depends on the build, an XPASS on a new CI build fails the job on purpose. They are not excluded.

## Verification here

| Check | Command | Result |
|---|---|---|
| Workflow parses, jobs listed | `python -c "import yaml; print(list(yaml.safe_load(open('.github/workflows/validation-scheduled.yml'))['jobs']))"` | `['slow-and-reports', 'fmu']` |
| Full suite without fmpy (as per-commit CI) | `PYTHONPATH=src:<dir with sitecustomize.py setting sys.modules["fmpy"] = None> python -m pytest -q` | see `b7_suite_without_fmpy.txt` |
| `fmu` tests with fmpy and the local FMUs | `PYTHONPATH=src python -m pytest -m fmu -q -rxX` | see `b7_pytest_fmu_local.txt` |
| Workflow dispatched once | `gh workflow run validation-scheduled.yml --ref dev` | **not executed: `gh` is not installed or authenticated on this machine.** To dispatch it: install `gh`, run `gh auth login`, then run that command. The run link goes here afterwards. Note: the new `fmu` job exists only on `dev` until the PR to `main`; dispatching with `--ref dev` runs dev's version of the file. |
