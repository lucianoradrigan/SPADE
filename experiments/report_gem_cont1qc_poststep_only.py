"""Duty 0, no load, from 150 rad/s, 0.5 s: RHS + post-step clamp (the fork) vs post-step clamp only."""
import warnings, numpy as np
warnings.simplefilter("ignore")
import gym_electric_motor as gem
from gym_electric_motor.physical_systems.converters import ContOneQuadrantConverter
from gym_electric_motor.physical_systems.mechanical_loads import PolynomialStaticLoad
for mode in ("rhs+post", "post-only"):
    load = PolynomialStaticLoad(load_parameter=dict(a=0.0, b=0.0, c=0.0, j_load=1e-3), load_initializer={"states": {"omega": 150.0}})
    env = gem.make("Cont-SC-PermExDc-v0", converter=ContOneQuadrantConverter(), load=load, visualization=(), constraints=(), tau=1e-4)
    env.reset(seed=0)
    ps = env.unwrapped.physical_system
    if mode == "post-only":
        k = ps._clamp_current_idx
        def rhs(t, y, u, _ps=ps, _k=k):
            _ps._clamp_current_idx = None
            try:
                return _ps._system_equation(t, y, u).copy()
            finally:
                _ps._clamp_current_idx = _k
        y0, t0 = ps._ode_solver.y.copy(), ps._ode_solver.t
        ps._ode_solver.set_system_equation(rhs, None)
        ps._ode_solver.set_initial_value(y0, t0)
    w = ps.state_names.index("omega")
    for _ in range(5000):
        (s, _), *_ = env.step(np.array([0.0]))
    print(mode, "omega at 0.5 s =", round(float(s[w] * ps.limits[w]), 3), "rad/s")
