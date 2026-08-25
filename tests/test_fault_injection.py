"""Tests for datagen/fault_injection.py's BearingFaultLoad, standalone and wired into a real
SCMLSystem (mirrors how datagen/runner.py, Macro-fase A.4, will use it)."""

import numpy as np
import pytest

from driveflow.datagen.fault_injection import BearingFaultLoad
from driveflow.sim import (
    ContOneQuadrantConverter,
    DcMotorSystem,
    DcPermanentlyExcitedMotor,
    EulerSolver,
    IdealVoltageSupply,
    PolynomialStaticLoad,
)
from driveflow.sim.vibration import bearing_frequencies as bf

GEOMETRY = bf.KAT_DATACENTER_6203_GEOMETRY
TAU = 1e-4


def build_base_load():
    return PolynomialStaticLoad(load_parameter=dict(a=0.01, b=0.05, c=0.0, j_load=0.0025))


def build_system(load):
    return DcMotorSystem(
        converter=ContOneQuadrantConverter(),
        motor=DcPermanentlyExcitedMotor(),
        load=load,
        supply=IdealVoltageSupply(u_nominal=560.0),
        ode_solver=EulerSolver(),
        tau=TAU,
    )


class TestBearingFaultLoadStandalone:
    def test_state_names_append_fault_phase(self):
        base = build_base_load()
        fault_load = BearingFaultLoad(base, fault_type="outer_race", geometry=GEOMETRY, severity=1.0)
        assert fault_load.state_names == list(base.state_names) + ["fault_phase"]

    def test_limits_carried_over_from_base_load(self):
        base = PolynomialStaticLoad(load_parameter=dict(a=0.01, b=0.05, c=0.0, j_load=0.0025), limits=dict(omega=150.0))
        fault_load = BearingFaultLoad(base, fault_type=None, geometry=GEOMETRY, severity=0.0)
        assert fault_load.limits["omega"] == pytest.approx(150.0)

    def test_healthy_fault_type_none_has_zero_ripple(self):
        base = build_base_load()
        fault_load = BearingFaultLoad(base, fault_type=None, geometry=GEOMETRY, severity=100.0)
        assert fault_load._ripple_torque(phase=0.3) == 0.0

    def test_faulted_ripple_is_nonzero_and_bounded_by_severity(self):
        base = build_base_load()
        severity = 2.0
        fault_load = BearingFaultLoad(base, fault_type="inner_race", geometry=GEOMETRY, severity=severity)
        phases = np.linspace(0, 4 * np.pi, 200)
        ripple = np.array([fault_load._ripple_torque(p) for p in phases])
        assert np.any(ripple != 0.0)
        assert np.max(np.abs(ripple)) <= severity * sum((1.0, 0.35, 0.15, 0.07)) + 1e-9

    def test_ripple_waveform_is_zero_mean_over_one_period(self):
        base = build_base_load()
        fault_load = BearingFaultLoad(base, fault_type="ball", geometry=GEOMETRY, severity=3.0)
        phases = np.linspace(0, 2 * np.pi, 100000, endpoint=False)
        ripple = np.array([fault_load._ripple_torque(p) for p in phases])
        assert ripple.mean() == pytest.approx(0.0, abs=1e-9)

    def test_order_override_bypasses_geometry_lookup(self):
        """dashboard.py's "Custom fault types": a user-defined fault_type name (not a real
        bearing_frequencies key) plus an explicit order, injected on the electrical/MCSA path the
        same way test_fault_impulses.py's TestFaultImpulseGeneratorOrderOverride covers the
        mechanical path."""
        base = build_base_load()
        fault_load = BearingFaultLoad(base, fault_type="gear_mesh", geometry=GEOMETRY, severity=2.0, order_override=6.5)
        assert fault_load._order == pytest.approx(6.5)
        phases = np.linspace(0, 4 * np.pi, 200)
        ripple = np.array([fault_load._ripple_torque(p) for p in phases])
        assert np.any(ripple != 0.0)


class TestBearingFaultLoadExtraFaults:
    """extra_faults is how dashboard.py's "Combine with additional fault types" injects more than
    one simultaneous fault on the electrical/MCSA path -- see BearingFaultLoad's docstring."""

    def test_no_extras_keeps_original_single_phase_state_name(self):
        """The pre-existing single-fault behavior must be byte-identical when extra_faults isn't
        used -- in particular the state is still named "fault_phase", not "fault_phase_0"."""
        base = build_base_load()
        fault_load = BearingFaultLoad(base, fault_type="outer_race", geometry=GEOMETRY, severity=1.0)
        assert fault_load.state_names == list(base.state_names) + ["fault_phase"]

    def test_extras_add_one_phase_state_each(self):
        base = build_base_load()
        fault_load = BearingFaultLoad(
            base, fault_type="outer_race", geometry=GEOMETRY, severity=1.0, extra_faults=[("gear_mesh", 2.0, 6.5)]
        )
        assert fault_load.state_names == list(base.state_names) + ["fault_phase_0", "fault_phase_1"]

    def test_extra_fault_order_override_bypasses_geometry_lookup(self):
        """gear_mesh isn't a real bearing_frequencies key -- must not raise, and its order must be
        exactly the override, not a geometry-derived guess."""
        base = build_base_load()
        fault_load = BearingFaultLoad(
            base, fault_type=None, geometry=GEOMETRY, severity=0.0, extra_faults=[("gear_mesh", 4.0, 6.5)]
        )
        order, severity = fault_load._all_faults[1]
        assert order == pytest.approx(6.5)
        assert severity == pytest.approx(4.0)

    def test_combined_ripple_differs_from_either_fault_alone(self):
        """The multi-fault mechanical_ode path must actually combine both faults' contributions,
        not just use one of them -- checked indirectly through a real ODE integration since
        mechanical_ode needs a full state vector with the right number of phase entries."""
        base_single = build_base_load()
        single = BearingFaultLoad(base_single, fault_type="outer_race", geometry=GEOMETRY, severity=1.0)
        system_single = build_system(single)
        state_single = system_single.reset()
        for _ in range(2000):
            state_single = system_single.simulate(np.array([0.5]))
        torque_single = (state_single * system_single.limits)[system_single.TORQUE_IDX]

        base_combo = build_base_load()
        combo = BearingFaultLoad(base_combo, fault_type="outer_race", geometry=GEOMETRY, severity=1.0, extra_faults=[("gear_mesh", 3.0, 6.5)])
        system_combo = build_system(combo)
        state_combo = system_combo.reset()
        for _ in range(2000):
            state_combo = system_combo.simulate(np.array([0.5]))
        torque_combo = (state_combo * system_combo.limits)[system_combo.TORQUE_IDX]

        assert torque_single != pytest.approx(torque_combo, rel=1e-6)


class TestBearingFaultLoadWithSCMLSystem:
    def test_healthy_matches_base_load_trajectory(self):
        """A BearingFaultLoad with fault_type=None must reproduce the same torque/omega/current
        trajectory as using the base_load directly (the wrapper is transparent when healthy).
        Note the exported state *vector* is one entry longer for the wrapped system (fault_phase
        is a genuine extra mechanical-ODE state, per BearingFaultLoad's docstring), so the two
        systems' physically-meaningful entries (selected via each system's own OMEGA_IDX/
        TORQUE_IDX/CURRENTS_IDX) are compared, not the raw arrays."""
        system_plain = build_system(build_base_load())
        system_wrapped = build_system(BearingFaultLoad(build_base_load(), fault_type=None, geometry=GEOMETRY))

        def physical_quantities(system, state):
            idx = [system.OMEGA_IDX, system.TORQUE_IDX] + list(system.CURRENTS_IDX) + list(system.VOLTAGES_IDX)
            return (state * system.limits)[idx]

        state_plain = system_plain.reset()
        state_wrapped = system_wrapped.reset()
        np.testing.assert_allclose(physical_quantities(system_plain, state_plain), physical_quantities(system_wrapped, state_wrapped))

        for _ in range(50):
            action = np.array([0.5])
            state_plain = system_plain.simulate(action)
            state_wrapped = system_wrapped.simulate(action)
            np.testing.assert_allclose(
                physical_quantities(system_plain, state_plain),
                physical_quantities(system_wrapped, state_wrapped),
                rtol=1e-10,
            )

    def test_reset_and_simulate_do_not_crash_with_fault(self):
        system = build_system(BearingFaultLoad(build_base_load(), fault_type="outer_race", geometry=GEOMETRY, severity=5.0))
        state = system.reset()
        assert np.all(np.isfinite(state))
        for _ in range(100):
            state = system.simulate(np.array([0.5]))
            assert np.all(np.isfinite(state))

    def test_fault_injection_produces_torque_ripple_at_expected_order(self):
        """FFT of the simulated torque signal at near-constant speed should show energy near the
        fault's characteristic order (in units of the mean shaft frequency)."""
        system = build_system(BearingFaultLoad(build_base_load(), fault_type="outer_race", geometry=GEOMETRY, severity=8.0))
        limits = system.limits
        torque_idx, omega_idx = system.TORQUE_IDX, system.OMEGA_IDX

        state = system.reset()
        n_steps = 4000
        torque_trace = np.zeros(n_steps)
        omega_trace = np.zeros(n_steps)
        for k in range(n_steps):
            state = system.simulate(np.array([0.6]))
            physical_state = state * limits
            torque_trace[k] = physical_state[torque_idx]
            omega_trace[k] = physical_state[omega_idx]

        # settle: use the second half, where speed is closer to steady-state
        settled = slice(n_steps // 2, None)
        f_r_hz = np.mean(omega_trace[settled]) / (2 * np.pi)
        expected_order = bf.fault_order("outer_race", GEOMETRY)
        expected_fault_hz = expected_order * f_r_hz

        torque_ac = torque_trace[settled] - torque_trace[settled].mean()
        fs = 1.0 / TAU
        spectrum = np.abs(np.fft.rfft(torque_ac))
        freqs = np.fft.rfftfreq(len(torque_ac), d=TAU)

        peak_freq = freqs[np.argmax(spectrum)]
        assert peak_freq == pytest.approx(expected_fault_hz, rel=0.1)
