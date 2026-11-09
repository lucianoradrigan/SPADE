"""CPU load for the S-22 check: a DC-motor simulation in a loop until killed."""
import time
from driveflow.datagen.scenario import Scenario
from driveflow.datagen.runner import run_scenario
while True:
    run_scenario(Scenario(scenario_id="s22_load", duration_s=0.5, seed=int(time.time()) % 1000))
