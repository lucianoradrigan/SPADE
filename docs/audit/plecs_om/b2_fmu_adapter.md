# B2 — Generic FMU adapter (round PLECS/OM, D6)

`src/driveflow/integrations/fmu/adapter.py` drives one FMI 2.0 Co-Simulation FMU step by step through fmpy. Nothing in it is specific to SPADE's models: it reads every name, causality, unit and range from the FMU's `modelDescription.xml`.

## API

| Call | What it does |
|---|---|
| `inspect_fmu(path)` → `FmuInfo` | Reads the model description and the binary platforms in the archive, without loading the binary. It returns the model name, GUID, FMI version, FMU type, SHA-256 and the variables (causality, variability, unit, min/max, start) |
| `FmuAdapter.load(path)` | Requires FMI 2.0 Co-Simulation and a binary for this platform, then extracts the archive |
| `reset(params, initial_state, seed, start_time)` | Frees the instance, re-instantiates it, sets parameters and start values, and runs initialisation. A fresh instance is more reproducible across FMUs than `fmi2Reset`. `seed` is recorded only, because the FMUs are deterministic |
| `set_inputs(dict)` · `do_step(t, dt)` · `get_outputs(check_range)` · `close()` | One communication step. `do_step` refuses a non-monotone t |
| `stats()` | Number of steps, p50 and p99 wall time per doStep (ms), and the number of solver failures |

**Typed errors**, `FmuError.code`:

| Code | Raised when |
|---|---|
| `load` | missing file, not FMI 2.0 Co-Simulation, unknown variable or input, fmpy missing |
| `platform` | no binary for this platform; the message says "Not supported on this machine" |
| `solver` | initialisation or doStep failure, or a non-monotone step |
| `nan` | non-finite input or output |
| `out_of_range` | an output outside its declared min/max |

**Dependency:** fmpy is imported lazily. Without the extra `[fmu]`, the adapter raises `FmuError("load")` with the install command; it never fails at import time.

## Tests (`tests/fmu/test_fmu_adapter.py`)

**Unit tests**, with a fake slave injected through `slave_factory`; fmpy only reads the model description:
- steps, inputs, outputs and statistics;
- reproducible reset that re-instantiates;
- each typed error;
- out-of-range output, unknown input, non-monotone time;
- platform mismatch.

**Integration test** (marker `fmu`; it skips when the FMU is missing): the real `BuckAveraged.fmu` at a constant duty of 0.5 for 500 steps of 20 µs. The test checks two things:
- two episodes are bit-identical;
- the maximum difference from the native `BuckSystem` is below 1 % of the native range.

The full comparison is B3.

**Results:** see `b2_branch_pytest.txt`.
