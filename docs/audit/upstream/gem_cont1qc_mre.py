"""Minimal reproduction: ContOneQuadrantConverter applies full supply voltage when the armature
current is negative (gym-electric-motor 3.0.3, `pip install gym-electric-motor==3.0.3`).

A one-quadrant (buck) converter cannot conduct negative armature current: when the back-EMF exceeds
the commanded average voltage, the current decays to zero and stays there (discontinuous
conduction) and an unloaded, frictionless motor keeps its speed. In GEM, `_convert` returns duty 1
when i < 0 and holds it for the whole step tau, so the motor is driven towards the no-load speed of
the full supply voltage instead.

Run: python mre_cont1qc.py
"""
import numpy as np
import gym_electric_motor as gem
from gym_electric_motor.physical_systems.mechanical_loads import PolynomialStaticLoad
from gym_electric_motor.physical_systems.converters import ContOneQuadrantConverter

OMEGA0 = 150.0  # rad/s, initial speed
DUTY = 0.0      # transistor always off: the motor can only coast (no load, no friction)

load = PolynomialStaticLoad(load_parameter=dict(a=0.0, b=0.0, c=0.0, j_load=1e-3),
                            load_initializer={"states": {"omega": OMEGA0}})
env = gem.make("Cont-SC-PermExDc-v0", converter=ContOneQuadrantConverter(), load=load, visualization=(), constraints=())
(state, ref), _ = env.reset(seed=0)
names = env.unwrapped.physical_system.state_names
lim = env.unwrapped.physical_system.limits
i_w, i_i, i_u = names.index("omega"), names.index("i"), names.index("u")
print(f"tau = {env.unwrapped.physical_system.tau} s, u_sup = {lim[names.index('u_sup')]:.1f} V, start omega = {state[i_w] * lim[i_w]:.2f} rad/s")
min_i = 0.0
for k in range(1, 5001):
    (state, ref), *_ = env.step(np.array([DUTY]))
    min_i = min(min_i, state[i_i] * lim[i_i])
    if k in (1, 10, 100, 1000, 5000):
        print(f"t = {k * 1e-4:6.4f} s  omega = {state[i_w] * lim[i_w]:8.2f} rad/s  i = {state[i_i] * lim[i_i]:8.3f} A  u = {state[i_u] * lim[i_u]:7.2f} V")
print(f"min armature current seen: {min_i:.3f} A")
print("Expected (one-quadrant converter, duty 0, no load): i >= 0 always, u = 0 or back-EMF, omega stays ~150 rad/s.")
