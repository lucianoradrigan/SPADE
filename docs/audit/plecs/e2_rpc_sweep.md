# E2: PLECS RPC sweep on a real PLECS installation

**Not run: no PLECS available.** No machine with PLECS Standalone and a license was available for this task (2026-10-21), so there is no log, manifest or hash from a real PLECS run.

Everything in E2 was tested against a fake XML-RPC server that imitates the documented commands (`tests/integrations/plecs/fake_plecs.py`). That server checks the client, the job, the dataset format and the GUI. It says nothing about how a real circuit behaves, about PLECS's actual fault strings, or about the orientation of `Values` in PLECS's real reply. The client accepts both orientations.

## To produce the evidence (on a machine with PLECS)

1. **Enable RPC in PLECS:** Preferences → General → RPC interface. It is disabled by default. Use port 1080 or another port.
2. **Prepare the circuit:** a buck whose top-level outports are, in order, `v_out` and `i_L`, with ModelVars `Vin`, `Rload` and `Vref`. This matches the example manifest in the page.
3. **Run SPADE:**
   - `python -m driveflow.jobs.worker serve`
   - `streamlit run src/driveflow/viz/app.py`
4. **In PLECS Plant Studio:**
   - Connection & Diagnostics → Test connection;
   - Create Plant → Generate simulation datasets → import the circuit with its manifest;
   - a 3×3 grid over `Vin` × `Rload`;
   - Start sweep.
5. **Save the record here:**
   - the worker log;
   - `manifest.json` and `conditions.csv` from `data/plecs/<profile>/<run_id>/`;
   - the circuit SHA-256;
   - the PLECS version, read from PLECS itself (the RPC interface does not report it);
   - the machine.
