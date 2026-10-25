"""B1 evidence: runs each switching FMU (built with CVODE, as scripts/fmu/build_fmus.sh builds it) at
constant duties for 2 ms with communication steps Ts/k (Ts = PWM period, k = 1, 2, 4, 5, 10, 20) and
records whether fmi2DoStep fails and at which simulated time. Finding S-15. The FMU's error and fatal
messages are kept.
    SPADE_FMU_DIR=... python experiments/report_b1_switching_cvode.py --out docs/audit/plecs_om/b1_switching_cvode.json"""

import argparse
import json
import os
from pathlib import Path

from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave, fmi2CallbackFunctions, fmi2CallbackLoggerTYPE, fmi2CallbackAllocateMemoryTYPE, fmi2CallbackFreeMemoryTYPE
from fmpy import calloc, free

FMU_DIR = Path(os.environ.get("SPADE_FMU_DIR", Path(__file__).resolve().parents[1] / "data" / "fmu"))
MODELS = {"BuckSwitching": 2e-5, "DcMotorSwitching": 1e-4}


def run(name, ts, divisor, stop=2e-3, duty=0.5):
    path = FMU_DIR / f"{name}.fmu"
    md = read_model_description(str(path))
    vr = {v.name: v.valueReference for v in md.modelVariables}
    unzip = extract(str(path))
    msgs = []

    def log(_env, _inst, status, _cat, message):
        if status >= 3:  # fmi2Error, fmi2Fatal
            msgs.append(message.decode(errors="replace"))

    cb = fmi2CallbackFunctions()
    cb.logger = fmi2CallbackLoggerTYPE(log)
    cb.allocateMemory = fmi2CallbackAllocateMemoryTYPE(calloc)
    cb.freeMemory = fmi2CallbackFreeMemoryTYPE(free)
    slave = FMU2Slave(guid=md.guid, unzipDirectory=unzip, modelIdentifier=md.coSimulation.modelIdentifier, instanceName="b1")
    slave.instantiate(loggingOn=True, callbacks=cb)
    h, t, n, err = ts / divisor, 0.0, 0, None
    try:
        slave.setupExperiment(startTime=0.0)
        slave.enterInitializationMode()
        slave.setReal([vr["duty"]], [duty])
        slave.exitInitializationMode()
        while t < stop - 1e-15:
            slave.setReal([vr["duty"]], [duty])
            slave.doStep(currentCommunicationPoint=t, communicationStepSize=h)
            n += 1
            t = n * h
    except Exception as exc:  # fmpy raises on fmi2Error / fmi2Fatal
        err = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            slave.terminate()
        except Exception:
            pass
        slave.freeInstance()
    return {"duty": duty, "step_s": h, "divisor": divisor, "steps_completed": n, "simulated_s": t, "failed": err is not None, "error": err, "fmu_log": msgs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    man = json.loads((FMU_DIR / "manifest.json").read_text())
    res = {m: [run(m, ts, k, duty=d) for d in (0.3, 0.5, 0.8) for k in (1, 2, 4, 5, 10, 20)] for m, ts in MODELS.items()}
    doc = {"fmu_manifest": {k: man[k] for k in ("openmodelica", "msl", "solver", "source_mo_sha256")},
           "fmu_sha256": {m: man["fmus"][m]["sha256"] for m in MODELS}, "stop_s": 2e-3, "results": res}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=2) + "\n")
    for m, rows in res.items():
        for r in rows:
            print(m, r["duty"], f"Ts/{r['divisor']}", "FAIL" if r["failed"] else "ok", r["steps_completed"], r["error"] or "")


if __name__ == "__main__":
    main()
