"""Tests for datagen/{scenario,runner,export_parquet}.py -- the A.4 orchestration layer -- and the
A.5 closeout criterion: generate normal + >=2 fault types under PI control, with synthetic
vibration populated, and verify the injected fault frequencies show up in BOTH the current (MCSA)
and the synthetic vibration spectra.
"""

import numpy as np
import pandas as pd
import pytest

from driveflow.datagen import Scenario, export_parquet, healthy_and_faulted_grid, run_flow, run_scenario, run_scenarios
from driveflow.datagen.runner import TAU
from driveflow.datagen.scenario import CALIBRATED_MECHANICAL_SEVERITY
from driveflow.sim.vibration import bearing_frequencies as bf


class TestScenario:
    def test_label_is_normal_when_no_fault(self):
        assert Scenario(scenario_id="s1").label == "normal"

    def test_label_matches_fault_type(self):
        assert Scenario(scenario_id="s1", fault_type="inner_race").label == "inner_race"

    def test_rejects_unimplemented_controller(self):
        with pytest.raises(NotImplementedError):
            Scenario(scenario_id="s1", controller_type="not_a_real_controller")

    def test_accepts_mpc_dc_motor_pairing(self):
        """MPC was implemented in Patch 10 (docs/patch10_implementacion_mpc.md) -- this used to
        raise NotImplementedError (see git history of test_rejects_unimplemented_controller
        above), which is exactly the Fase A / INSTRUCTIONS.md discrepancy that patch resolved."""
        Scenario(scenario_id="s1", controller_type="MPC")  # default plant_config_id is dc_perm_ex_v1

    def test_rejects_mpc_vsc_pairing(self):
        with pytest.raises(NotImplementedError):
            Scenario(scenario_id="s1", controller_type="MPC", plant_config_id="vsc_dpc_v1")

    def test_motor_parameter_override_changes_simulated_behavior(self):
        """viz/dashboard.py's editable motor-characteristics panel relies on this actually
        reaching the simulated plant, not just being stored inert on the Scenario.

        Uses psi_e, not r_a: PICascadeController retunes its own current-loop gains from the
        motor's r_a/l_a at construction (magnitude-optimum, see pi_controller.py) specifically so
        closed-loop *current tracking* is invariant to r_a/l_a -- an r_a override would correctly
        show ~no difference in current_r, which would make this test look broken when the
        feature isn't. psi_e (the torque constant) isn't part of that retuning, so a change there
        shows up directly in how fast the motor accelerates for the same current."""
        baseline = run_scenario(Scenario(scenario_id="s1", duration_s=0.02))
        overridden = run_scenario(Scenario(scenario_id="s2", duration_s=0.02, motor_parameter_overrides=dict(psi_e=0.3)))
        assert baseline[-1]["rpm"] != overridden[-1]["rpm"]

    def test_torque_ref_nm_switches_runner_to_torque_control(self):
        """dashboard.py's "Torque (τ_ref)" control mode relies on this: setting torque_ref_nm
        must actually route runner.py to PICascadeController.control_torque(), not just be stored
        inert on the Scenario (same class of bug the motor_parameter_overrides test above guards
        against). duration_s=0.3 gives the current loop time to settle (it's much faster than the
        speed loop, see pi_controller.py's TestPICascadeControllerTorqueMode)."""
        psi_e = 0.165  # DcPermanentlyExcitedMotor's stock psi_e -- torque = psi_e * i_a for this motor
        torque_ref = 10.0
        records = run_scenario(Scenario(scenario_id="s1", duration_s=0.3, torque_ref_nm=torque_ref))
        settled_torque_nm = [r["torque_nm"] for r in records[-500:]]
        assert sum(settled_torque_nm) / len(settled_torque_nm) == pytest.approx(torque_ref, rel=0.05)

    def test_torque_ref_nm_none_is_a_byte_identical_no_op(self):
        """The default (None) must not perturb any existing caller/test's output -- torque control
        is opt-in, same guarantee as the noise_pct fields above."""
        a = run_scenario(Scenario(scenario_id="s1", duration_s=0.02))
        b = run_scenario(Scenario(scenario_id="s2", duration_s=0.02, torque_ref_nm=None))
        assert [r["current_r"] for r in a] == [r["current_r"] for r in b]

    def test_fault_order_override_injects_a_user_defined_fault_type(self):
        """dashboard.py's "Custom fault types": fault_type doesn't need to be a real
        bearing_frequencies key when fault_order_override is set -- runner.py must actually reach
        both BearingFaultLoad and the vibration synthesizer with it (not just accept it inertly on
        Scenario), producing a distinguishable trace from a healthy run on both paths."""
        healthy = run_scenario(Scenario(scenario_id="s1", duration_s=0.1))
        custom = run_scenario(
            Scenario(scenario_id="s2", duration_s=0.1, fault_type="gear_mesh", electrical_severity=8.0, mechanical_severity=0.05, fault_order_override=6.5)
        )
        assert [r["current_r"] for r in healthy] != [r["current_r"] for r in custom]
        assert [r["acc_x"] for r in healthy] != [r["acc_x"] for r in custom]
        assert custom[-1]["label"] == "gear_mesh"

    def test_extra_faults_combine_with_the_primary_fault(self):
        """dashboard.py's "Combine with additional fault types": extra_faults must actually reach
        both excitation paths (not just be stored inert on the Scenario), and must produce a
        result distinguishable from the primary fault alone on both paths."""
        primary_only = run_scenario(Scenario(scenario_id="s1", duration_s=0.1, fault_type="outer_race", electrical_severity=8.0, mechanical_severity=0.05))
        combined = run_scenario(
            Scenario(
                scenario_id="s2",
                duration_s=0.1,
                fault_type="outer_race",
                electrical_severity=8.0,
                mechanical_severity=0.05,
                extra_faults=[("gear_mesh", 6.5)],
            )
        )
        assert [r["current_r"] for r in primary_only] != [r["current_r"] for r in combined]
        assert [r["acc_x"] for r in primary_only] != [r["acc_x"] for r in combined]
        assert combined[-1]["label"] == "outer_race+gear_mesh"

    def test_extra_faults_work_even_with_a_healthy_primary(self):
        """A "healthy" primary (fault_type=None) plus extras must still inject those extras --
        extras aren't gated on the primary being active."""
        healthy = run_scenario(Scenario(scenario_id="s1", duration_s=0.1))
        healthy_plus_extra = run_scenario(
            Scenario(scenario_id="s2", duration_s=0.1, electrical_severity=8.0, mechanical_severity=0.05, extra_faults=[("gear_mesh", 6.5)])
        )
        assert [r["current_r"] for r in healthy] != [r["current_r"] for r in healthy_plus_extra]
        assert healthy_plus_extra[-1]["label"] == "gear_mesh"

    def test_noise_pct_zero_is_a_byte_identical_no_op(self):
        """The default (0%) must not perturb any existing caller/test's output -- noise is opt-in."""
        a = run_scenario(Scenario(scenario_id="s1", duration_s=0.02))
        b = run_scenario(Scenario(scenario_id="s2", duration_s=0.02, electrical_noise_pct=0.0, mechanical_noise_pct=0.0))
        assert [r["current_r"] for r in a] == [r["current_r"] for r in b]
        assert [r["acc_x"] for r in a] == [r["acc_x"] for r in b]

    def test_electrical_noise_pct_perturbs_current_but_not_vibration(self):
        clean = run_scenario(Scenario(scenario_id="s1", duration_s=0.02))
        noisy = run_scenario(Scenario(scenario_id="s2", duration_s=0.02, electrical_noise_pct=5.0))
        assert [r["current_r"] for r in clean] != [r["current_r"] for r in noisy]
        assert [r["acc_x"] for r in clean] == [r["acc_x"] for r in noisy]

    def test_mechanical_noise_pct_perturbs_vibration_but_not_current(self):
        clean = run_scenario(Scenario(scenario_id="s1", duration_s=0.02))
        noisy = run_scenario(Scenario(scenario_id="s2", duration_s=0.02, mechanical_noise_pct=5.0))
        assert [r["acc_x"] for r in clean] != [r["acc_x"] for r in noisy]
        assert [r["current_r"] for r in clean] == [r["current_r"] for r in noisy]

    def test_noise_is_reproducible_for_a_fixed_seed(self):
        a = run_scenario(Scenario(scenario_id="s1", duration_s=0.02, seed=7, electrical_noise_pct=5.0, mechanical_noise_pct=5.0))
        b = run_scenario(Scenario(scenario_id="s2", duration_s=0.02, seed=7, electrical_noise_pct=5.0, mechanical_noise_pct=5.0))
        assert [r["current_r"] for r in a] == [r["current_r"] for r in b]
        assert [r["acc_x"] for r in a] == [r["acc_x"] for r in b]

    def test_rejects_mismatched_controller_plant_pairing(self):
        """DPC targets the VSC (vsc_dpc_v1), not the DC motor -- see runner.py's module
        docstring. Pairing DPC with the DC motor's plant_config_id (or PI with the VSC's) must
        be rejected the same way an unimplemented controller is."""
        with pytest.raises(NotImplementedError):
            Scenario(scenario_id="s1", controller_type="DPC")  # default plant_config_id is dc_perm_ex_v1
        with pytest.raises(NotImplementedError):
            Scenario(scenario_id="s1", controller_type="PI", plant_config_id="vsc_dpc_v1")

    def test_accepts_dpc_vsc_pairing(self):
        Scenario(scenario_id="s1", controller_type="DPC", plant_config_id="vsc_dpc_v1")

    def test_electrical_and_mechanical_severity_are_independent(self):
        """Regression for the finding in docs/patch2_retiro_modulo_C.md Sec. 4: a single shared
        severity conflated two unrelated physical scales (Nm of torque ripple vs. Module B's own
        impulse-amplitude units)."""
        scenario = Scenario(scenario_id="s1", fault_type="outer_race", electrical_severity=8.0, mechanical_severity=0.05)
        assert scenario.electrical_severity == 8.0
        assert scenario.mechanical_severity == 0.05

    def test_mechanical_severity_defaults_to_calibrated_value_per_fault_type(self):
        for fault_type, expected in CALIBRATED_MECHANICAL_SEVERITY.items():
            assert Scenario(scenario_id="s1", fault_type=fault_type).mechanical_severity == expected

    def test_mechanical_severity_defaults_to_zero_when_healthy(self):
        assert Scenario(scenario_id="s1", fault_type=None).mechanical_severity == 0.0

    def test_explicit_mechanical_severity_overrides_calibrated_default(self):
        scenario = Scenario(scenario_id="s1", fault_type="outer_race", mechanical_severity=123.0)
        assert scenario.mechanical_severity == 123.0

    def test_healthy_and_faulted_grid_covers_all_conditions_and_seeds(self):
        scenarios = healthy_and_faulted_grid("run", fault_types=("outer_race", "inner_race"), seeds=(0, 1))
        labels = sorted((s.label, s.seed) for s in scenarios)
        assert labels == sorted(
            [
                ("normal", 0),
                ("normal", 1),
                ("outer_race", 0),
                ("outer_race", 1),
                ("inner_race", 0),
                ("inner_race", 1),
            ]
        )


class TestRunFlow:
    """viz/dashboard.py's "Advanced Flow" mode: an ordered sequence of Scenarios concatenated
    into one continuous timeline -- see run_flow's own docstring for why each segment is an
    independent simulation, not a continuation of the previous segment's exact instantaneous
    state."""

    def test_concatenates_records_in_order(self):
        seg_a = Scenario(scenario_id="a", duration_s=0.01, fault_type=None)
        seg_b = Scenario(scenario_id="b", duration_s=0.02, fault_type="outer_race", mechanical_severity=0.05)
        records = run_flow([seg_a, seg_b])
        assert len(records) == round(0.01 / TAU) + round(0.02 / TAU)
        assert all(r["segment_index"] == 0 for r in records[: round(0.01 / TAU)])
        assert all(r["segment_index"] == 1 for r in records[round(0.01 / TAU) :])
        assert records[0]["segment_label"] == "normal"
        assert records[-1]["segment_label"] == "outer_race"

    def test_timestamps_are_continuous_across_segments(self):
        seg_a = Scenario(scenario_id="a", duration_s=0.01, fault_type=None)
        seg_b = Scenario(scenario_id="b", duration_s=0.01, fault_type=None)
        records = run_flow([seg_a, seg_b])
        n = round(0.01 / TAU)
        # last sample of segment a and first sample of segment b are exactly one TAU apart, with
        # segment b's timestamps offset by segment a's full duration -- one unbroken timeline.
        assert records[n]["timestamp_s"] == pytest.approx(records[n - 1]["timestamp_s"] + TAU)
        assert records[-1]["timestamp_s"] == pytest.approx(0.02, abs=1e-9)

    def test_each_segment_keeps_its_own_control_mode_and_setpoint(self):
        """Different segments can use different control modes -- speed for one, torque for the
        next -- each resolved independently by its own Scenario, same as running them standalone."""
        seg_speed = Scenario(scenario_id="a", duration_s=0.02, omega_ref_rad_s=200.0)
        seg_torque = Scenario(scenario_id="b", duration_s=0.02, torque_ref_nm=10.0)
        records = run_flow([seg_speed, seg_torque])
        speed_segment = [r for r in records if r["segment_index"] == 0]
        torque_segment = [r for r in records if r["segment_index"] == 1]
        assert speed_segment[-1]["rpm"] > 0
        assert torque_segment[-1]["torque_nm"] == pytest.approx(10.0, rel=0.1)


class TestRunScenario:
    def test_returns_one_record_per_step(self):
        scenario = Scenario(scenario_id="short", duration_s=0.01, fault_type=None)
        records = run_scenario(scenario)
        assert len(records) == round(0.01 / TAU)

    def test_healthy_run_has_no_fault_frequencies(self):
        records = run_scenario(Scenario(scenario_id="s", duration_s=0.01, fault_type=None))
        assert all(np.isnan(r["bpfo_hz"]) for r in records)
        assert all(r["label"] == "normal" for r in records)
        assert all(r["mechanical_severity"] == 0.0 and r["electrical_severity_nm"] == 0.0 for r in records)

    def test_faulted_run_has_fault_frequencies_and_label(self):
        records = run_scenario(
            Scenario(scenario_id="s", duration_s=0.01, fault_type="outer_race", mechanical_severity=5.0)
        )
        assert all(not np.isnan(r["bpfo_hz"]) for r in records)
        assert all(r["label"] == "outer_race" for r in records)
        assert all(r["mechanical_severity"] == 5.0 for r in records)

    def test_reproducible_with_fixed_seed(self):
        scenario = Scenario(scenario_id="s", duration_s=0.01, fault_type="ball", mechanical_severity=3.0, seed=42)
        records_a = run_scenario(scenario)
        records_b = run_scenario(scenario)
        for a, b in zip(records_a, records_b):
            assert a["acc_x"] == b["acc_x"]
            assert a["current_r"] == b["current_r"]

    def test_vibration_source_is_always_synthetic_b(self):
        records = run_scenario(Scenario(scenario_id="s", duration_s=0.005))
        assert all(r["vibration_source"] == "synthetic_b" for r in records)

    def test_all_signals_finite(self):
        records = run_scenario(Scenario(scenario_id="s", duration_s=0.02, fault_type="inner_race", mechanical_severity=6.0))
        for col in ("current_r", "acc_x", "acc_y", "acc_z", "rpm", "torque_nm", "voltage_v"):
            assert all(np.isfinite(r[col]) for r in records)

    def test_voltage_v_populated_in_both_control_modes(self):
        """u_a is the controller's actual applied voltage, not the setpoint -- it should be a real,
        non-degenerate signal whether the loop is tracking a speed or a torque reference."""
        speed_records = run_scenario(Scenario(scenario_id="s", duration_s=0.02, omega_ref_rad_s=100.0))
        torque_records = run_scenario(Scenario(scenario_id="s", duration_s=0.02, torque_ref_nm=5.0))
        for records in (speed_records, torque_records):
            voltages = [r["voltage_v"] for r in records]
            assert all(np.isfinite(v) for v in voltages)
            assert len(set(voltages)) > 1  # not flat-lined/degenerate


class TestRunScenarioMpc:
    """controller_type="MPC" through the full Scenario/run_scenario pipeline, against the SAME
    dc_perm_ex_v1 plant PI already runs against (see control/mpc/controller.py and
    docs/patch10_implementacion_mpc.md) -- short durations since MpcController solves a real QP
    every step (~7ms/step), unlike PI's O(1) update."""

    def test_returns_one_record_per_step(self):
        records = run_scenario(Scenario(scenario_id="s", controller_type="MPC", duration_s=0.01))
        assert len(records) == round(0.01 / TAU)

    def test_all_signals_finite(self):
        records = run_scenario(
            Scenario(scenario_id="s", controller_type="MPC", duration_s=0.01, fault_type="inner_race", mechanical_severity=6.0)
        )
        for col in ("current_r", "acc_x", "acc_y", "acc_z", "rpm", "torque_nm", "voltage_v"):
            assert all(np.isfinite(r[col]) for r in records)

    def test_torque_ref_nm_switches_runner_to_torque_control(self):
        records = run_scenario(Scenario(scenario_id="s", controller_type="MPC", duration_s=0.01, torque_ref_nm=10.0))
        assert records[-1]["torque_nm"] == pytest.approx(10.0, rel=0.2)


class TestDpcVscScenario:
    """Macro-fase B.2 (docs/macro_fase_B2_dpc_deployment.md): controller_type="DPC" against
    plant_config_id="vsc_dpc_v1", a Voltage Source Converter -- a genuinely different plant from
    the DC motor above, not just a different controller on the same system (see runner.py)."""

    @staticmethod
    def _dpc_scenario(**kwargs):
        return Scenario(scenario_id="dpc_test", controller_type="DPC", plant_config_id="vsc_dpc_v1", **kwargs)

    def test_returns_one_record_per_step(self):
        records = run_scenario(self._dpc_scenario(duration_s=0.01))
        assert len(records) == round(0.01 / TAU)

    def test_dc_motor_columns_are_nan_vsc_columns_are_populated(self):
        records = run_scenario(self._dpc_scenario(duration_s=0.005))
        for r in records:
            for col in ("current_r", "rpm", "torque_nm", "voltage_v", "acc_x", "acc_y", "acc_z", "bpfo_hz"):
                assert np.isnan(r[col])
            assert r["vibration_source"] is None
            assert r["label"] == "normal"
            for col in ("v_ref_real", "v_ref_imag", "vc_real", "vc_imag", "i_f_real", "i_f_imag"):
                assert np.isfinite(r[col])

    def test_reproducible_with_fixed_seed(self):
        records_a = run_scenario(self._dpc_scenario(duration_s=0.005, seed=7))
        records_b = run_scenario(self._dpc_scenario(duration_s=0.005, seed=7))
        for a, b in zip(records_a, records_b):
            assert a["vc_real"] == b["vc_real"]
            assert a["vc_imag"] == b["vc_imag"]

    def test_different_seeds_give_different_reference_phase(self):
        records_a = run_scenario(self._dpc_scenario(duration_s=0.001, seed=0))
        records_b = run_scenario(self._dpc_scenario(duration_s=0.001, seed=90))
        assert records_a[0]["v_ref_real"] != pytest.approx(records_b[0]["v_ref_real"])

    def test_closed_loop_tracking_quality_regression(self):
        """Guards the checkpoint integrated here (dpc_trained_v3_closed_loop.weights.h5): if a
        future change swaps in a worse checkpoint, a wrong R, or breaks the reference/controller
        wiring, settled RMSE should blow well past this bound -- v3's own measured settled RMSE
        was ~1.18V against a 50V reference (docs/macro_fase_B2_dpc_deployment.md); this asserts a
        generous multiple of that, not the exact figure, to tolerate run-to-run/version noise."""
        records = run_scenario(self._dpc_scenario(duration_s=0.1))
        settled = records[len(records) // 4 :]  # discard the initial transient
        errors = [np.hypot(r["vc_real"] - r["v_ref_real"], r["vc_imag"] - r["v_ref_imag"]) for r in settled]
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        assert rmse < 5.0

    def test_no_drift_over_a_long_run(self):
        """The original v1/v2/v3 validation (docs/macro_fase_B2_dpc_deployment.md) only ever ran
        2000 steps (200ms). Found no regression at 3s (30000 steps, ~150 reference rotations) in
        an ad-hoc check -- this test uses 1s (10000 steps, still 5x the original coverage) to
        keep the regular suite fast while still confirming settled RMSE doesn't grow over time: a
        memoryless controller re-predicting from scratch every step could in principle still
        accumulate a slow bias the original short runs wouldn't reveal."""
        records = run_scenario(self._dpc_scenario(duration_s=1.0))
        errors = np.array([np.hypot(r["vc_real"] - r["v_ref_real"], r["vc_imag"] - r["v_ref_imag"]) for r in records])
        assert np.isfinite(errors).all()

        settled = errors[len(errors) // 20 :]  # discard the initial transient (first 5%)
        quarter = len(settled) // 4
        rmse_per_quarter = [float(np.sqrt(np.mean(np.square(settled[i * quarter : (i + 1) * quarter])))) for i in range(4)]
        assert max(rmse_per_quarter) < 2.0 * min(rmse_per_quarter)  # bounded, not growing quarter to quarter
        assert max(rmse_per_quarter) < 5.0

    def test_load_resistance_ohm_default_is_a_byte_identical_no_op(self):
        """viz/dashboard.py's Fase B "Load resistance" control defaults to None -- must reproduce
        the hardcoded _VSC_R_OHM behavior exactly, not silently drift."""
        a = run_scenario(self._dpc_scenario(duration_s=0.01))
        b = run_scenario(self._dpc_scenario(duration_s=0.01, load_resistance_ohm=None))
        assert [r["vc_real"] for r in a] == [r["vc_real"] for r in b]

    def test_load_resistance_ohm_reaches_both_plant_and_controller(self):
        """Must actually change the simulated trace (not stored inert on the Scenario) -- same
        class of check as motor_parameter_overrides/torque_ref_nm above. A different R changes
        both i_load = vc/R in the plant AND the network's own r_ohm input feature (see
        control/dpc/controller.py), so tracking should visibly differ, not just current."""
        default_r = run_scenario(self._dpc_scenario(duration_s=0.01))
        different_r = run_scenario(self._dpc_scenario(duration_s=0.01, load_resistance_ohm=4.0))
        assert [r["vc_real"] for r in default_r] != [r["vc_real"] for r in different_r]
        assert [r["i_f_real"] for r in default_r] != [r["i_f_real"] for r in different_r]

    def test_reference_magnitude_and_omega_default_is_a_byte_identical_no_op(self):
        a = run_scenario(self._dpc_scenario(duration_s=0.01))
        b = run_scenario(self._dpc_scenario(duration_s=0.01, reference_magnitude_v=None, reference_omega_rad_s=None))
        assert [r["v_ref_real"] for r in a] == [r["v_ref_real"] for r in b]

    def test_reference_magnitude_v_changes_the_reference_and_tracking(self):
        default = run_scenario(self._dpc_scenario(duration_s=0.01))
        bigger = run_scenario(self._dpc_scenario(duration_s=0.01, reference_magnitude_v=100.0))
        # the commanded reference itself must scale...
        assert max(abs(r["v_ref_real"]) for r in bigger) == pytest.approx(2 * max(abs(r["v_ref_real"]) for r in default), rel=0.05)
        # ...and reach the network (not stored inert on the Scenario) -- different tracking behavior.
        assert [r["vc_real"] for r in default] != [r["vc_real"] for r in bigger]

    def test_reference_omega_rad_s_changes_the_reference_rotation_rate(self):
        default = run_scenario(self._dpc_scenario(duration_s=0.01))
        faster = run_scenario(self._dpc_scenario(duration_s=0.01, reference_omega_rad_s=2 * 2 * np.pi * 50.0))
        assert [r["v_ref_real"] for r in default] != [r["v_ref_real"] for r in faster]


class TestExportParquet:
    def test_export_and_reload_round_trip(self, tmp_path):
        scenarios = [Scenario(scenario_id="a", duration_s=0.005, fault_type=None), Scenario(scenario_id="b", duration_s=0.005, fault_type="ball", mechanical_severity=2.0)]
        runs = run_scenarios(scenarios)
        out_path = tmp_path / "dataset.parquet"
        df = export_parquet(runs, out_path)

        assert out_path.exists()
        reloaded = pd.read_parquet(out_path)
        assert list(reloaded.columns) == list(df.columns)
        assert len(reloaded) == sum(len(r) for r in runs)
        assert set(reloaded["label"]) == {"normal", "ball"}

    def test_flat_record_list_also_works(self, tmp_path):
        records = run_scenario(Scenario(scenario_id="a", duration_s=0.005))
        out_path = tmp_path / "flat.parquet"
        df = export_parquet(records, out_path)
        assert len(df) == len(records)

    def test_missing_column_raises(self):
        with pytest.raises(ValueError):
            export_parquet([{"timestamp_s": 0.0}], "/tmp/unused_should_not_be_written.parquet")

    def test_mixed_dc_motor_and_vsc_scenarios_export_together(self, tmp_path):
        """The two plant types (see runner.py) share one wide schema -- a real dataset would mix
        PI/dc_perm_ex_v1 and DPC/vsc_dpc_v1 runs in the same export."""
        dc_records = run_scenario(Scenario(scenario_id="dc", duration_s=0.005))
        vsc_records = run_scenario(Scenario(scenario_id="vsc", controller_type="DPC", plant_config_id="vsc_dpc_v1", duration_s=0.005))
        df = export_parquet([dc_records, vsc_records], tmp_path / "mixed.parquet")

        assert set(df["plant_config_id"]) == {"dc_perm_ex_v1", "vsc_dpc_v1"}
        assert df.loc[df["plant_config_id"] == "dc_perm_ex_v1", "vc_real"].isna().all()
        assert df.loc[df["plant_config_id"] == "vsc_dpc_v1", "vc_real"].notna().all()
        assert df.loc[df["plant_config_id"] == "vsc_dpc_v1", "current_r"].isna().all()
        assert df.loc[df["plant_config_id"] == "dc_perm_ex_v1", "current_r"].notna().all()


class TestA5CloseoutCriterion:
    """Reproduces INSTRUCTIONS.md's A.5 validation: normal + >=2 fault types under PI control,
    vibration populated, injected fault frequencies visible in BOTH the current (MCSA) and the
    synthetic-vibration spectra."""

    @staticmethod
    def _settled_spectrum(df, column):
        settled = df.iloc[len(df) // 2 :]
        signal = settled[column].to_numpy() - settled[column].mean()
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), d=TAU)
        return freqs, spectrum

    @staticmethod
    def _band_energy(freqs, spectrum, center_hz, half_width_hz=15.0):
        mask = np.abs(freqs - center_hz) <= half_width_hz
        return float(np.sum(spectrum[mask] ** 2))

    @pytest.mark.parametrize("fault_type", ["outer_race", "inner_race"])
    def test_fault_frequency_visible_in_current_and_vibration_spectra(self, fault_type):
        """The fault frequency must show up as a raw-spectrum peak in the current (MCSA) --
        that path has no resonant amplification distorting it. For vibration it must show up as
        *elevated energy in that frequency band relative to a healthy baseline*, not necessarily
        as the spectrum's single largest peak: Module B's calibrated structural resonances can
        (and, physically, should be expected to) dominate the raw peak amplitude more than the
        fault's own fundamental -- exactly why real bearing-fault diagnostics read an envelope
        spectrum rather than pick the raw spectrum's global maximum (paper_federative's own
        pipeline works this way too, per docs/propuesta_consolidacion.pdf Sec. 2.3's
        envelope_forecaster / windowing.py)."""
        common_kwargs = dict(duration_s=0.4, omega_ref_rad_s=150.0, seed=0)
        healthy_records = run_scenario(Scenario(scenario_id=f"a5_{fault_type}_healthy_ref", fault_type=None, **common_kwargs))
        faulty_records = run_scenario(
            Scenario(scenario_id=f"a5_{fault_type}", fault_type=fault_type, mechanical_severity=8.0, **common_kwargs)
        )
        df_healthy = pd.DataFrame.from_records(healthy_records)
        df_faulty = pd.DataFrame.from_records(faulty_records)

        settled_faulty = df_faulty.iloc[len(df_faulty) // 2 :]
        f_r_hz = settled_faulty["rpm"].mean() / 60.0
        expected_hz = bf.fault_order(fault_type, bf.KAT_DATACENTER_6203_GEOMETRY) * f_r_hz

        current_freqs, current_spectrum = self._settled_spectrum(df_faulty, "current_r")
        current_peak = current_freqs[np.argmax(current_spectrum)]
        assert current_peak == pytest.approx(expected_hz, rel=0.15)

        acc_freqs_faulty, acc_spectrum_faulty = self._settled_spectrum(df_faulty, "acc_x")
        acc_freqs_healthy, acc_spectrum_healthy = self._settled_spectrum(df_healthy, "acc_x")
        faulty_band_energy = self._band_energy(acc_freqs_faulty, acc_spectrum_faulty, expected_hz)
        healthy_band_energy = self._band_energy(acc_freqs_healthy, acc_spectrum_healthy, expected_hz)
        assert faulty_band_energy > 2.0 * healthy_band_energy

    def test_generates_normal_plus_two_fault_types_dataset(self, tmp_path):
        scenarios = healthy_and_faulted_grid(
            "a5_dataset", fault_types=("outer_race", "inner_race"), mechanical_severity=8.0, seeds=(0,), duration_s=0.1
        )
        runs = run_scenarios(scenarios)
        df = export_parquet(runs, tmp_path / "a5_dataset.parquet")

        assert set(df["label"]) == {"normal", "outer_race", "inner_race"}
        assert df["acc_x"].notna().all()
        assert df["acc_y"].notna().all()
        assert df["acc_z"].notna().all()
        assert (df.loc[df["label"] != "normal", "bpfo_hz"].notna() | df.loc[df["label"] != "normal", "bpfi_hz"].notna()).all()
        assert df.loc[df["label"] == "normal", "bpfo_hz"].isna().all()
