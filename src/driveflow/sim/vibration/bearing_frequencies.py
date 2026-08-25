"""Bearing fault characteristic frequencies (BPFO/BPFI/BSF/FTF).

Shared between the mechanical excitation path (``sim/vibration/fault_impulses.py``, Module B)
and the electrical path (``datagen/fault_injection.py``'s ``BearingFaultLoad``, Macro-fase A.4) --
one implementation, per ``docs/addendum_vibracion_v1.md`` Sec. 3.

Formulas (addendum Sec. 3, standard rolling-element bearing kinematics)::

    BPFO = (n/2) * f_r * (1 - (d/D) * cos(phi))
    BPFI = (n/2) * f_r * (1 + (d/D) * cos(phi))
    BSF  = (D/(2*d)) * f_r * (1 - (d/D)**2 * cos(phi)**2)
    FTF  = (f_r/2) * (1 - (d/D) * cos(phi))

where ``f_r`` is the shaft rotational frequency in Hz, ``n`` the number of rolling elements,
``d``/``D`` the rolling-element/pitch diameters and ``phi`` the contact angle (0 for a purely
radial deep-groove ball bearing).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BearingGeometry:
    """Geometric parameters of a rolling-element bearing.

    Args:
        n_elements: Number of rolling elements (balls/rollers).
        element_diameter_m: Rolling-element diameter ``d``, in meters.
        pitch_diameter_m: Pitch diameter ``D`` (center-to-center of opposing elements), in meters.
        contact_angle_rad: Contact angle ``phi``, in radians. 0 for a purely radial load.
    """

    n_elements: int
    element_diameter_m: float
    pitch_diameter_m: float
    contact_angle_rad: float = 0.0

    def __post_init__(self):
        if self.n_elements <= 0:
            raise ValueError("n_elements must be positive")
        if self.element_diameter_m <= 0 or self.pitch_diameter_m <= 0:
            raise ValueError("element_diameter_m and pitch_diameter_m must be positive")
        if self.element_diameter_m >= self.pitch_diameter_m:
            raise ValueError("element_diameter_m must be smaller than pitch_diameter_m")


def shaft_frequency_hz(omega_rad_s):
    """Converts mechanical angular speed omega (rad/s, as returned by SCMLSystem) to the
    shaft rotational frequency f_r (Hz)."""
    return np.asarray(omega_rad_s) / (2.0 * np.pi)


def bpfo(f_r_hz, geometry: BearingGeometry):
    """Ball Pass Frequency Outer race."""
    ratio = geometry.element_diameter_m / geometry.pitch_diameter_m
    return (geometry.n_elements / 2.0) * f_r_hz * (1.0 - ratio * np.cos(geometry.contact_angle_rad))


def bpfi(f_r_hz, geometry: BearingGeometry):
    """Ball Pass Frequency Inner race."""
    ratio = geometry.element_diameter_m / geometry.pitch_diameter_m
    return (geometry.n_elements / 2.0) * f_r_hz * (1.0 + ratio * np.cos(geometry.contact_angle_rad))


def bsf(f_r_hz, geometry: BearingGeometry):
    """Ball Spin Frequency."""
    ratio = geometry.element_diameter_m / geometry.pitch_diameter_m
    return (geometry.pitch_diameter_m / (2.0 * geometry.element_diameter_m)) * f_r_hz * (
        1.0 - ratio**2 * np.cos(geometry.contact_angle_rad) ** 2
    )


def ftf(f_r_hz, geometry: BearingGeometry):
    """Fundamental Train Frequency (cage speed)."""
    ratio = geometry.element_diameter_m / geometry.pitch_diameter_m
    return (f_r_hz / 2.0) * (1.0 - ratio * np.cos(geometry.contact_angle_rad))


#: Maps the fault labels used across driveflow (dataset schema, BearingFaultLoad, ...) to their
#: characteristic-frequency function. "cage" (FTF) is included for completeness even though it is
#: rarely used as an injected fault type.
FAULT_FREQUENCY_FUNCS = {
    "outer_race": bpfo,
    "inner_race": bpfi,
    "ball": bsf,
    "cage": ftf,
}


def fault_order(fault_type: str, geometry: BearingGeometry) -> float:
    """The fault's characteristic frequency expressed as a multiple ("order") of the shaft
    frequency, i.e. ``fault_frequency_hz = fault_order(...) * f_r_hz``. All four formulas above
    are linear in f_r, so this is just the formula evaluated at f_r_hz=1.0. Useful for
    order-tracking under time-varying speed (see ``fault_impulses.ImpulseTrainGenerator``).
    """
    try:
        func = FAULT_FREQUENCY_FUNCS[fault_type]
    except KeyError as exc:
        raise ValueError(f"Unknown fault_type {fault_type!r}, expected one of {list(FAULT_FREQUENCY_FUNCS)}") from exc
    return float(func(1.0, geometry))


def fault_frequencies_hz(f_r_hz, geometry: BearingGeometry) -> dict:
    """Convenience: all four characteristic frequencies (Hz) for a given shaft frequency."""
    return {name: func(f_r_hz, geometry) for name, func in FAULT_FREQUENCY_FUNCS.items()}


#: Geometry of the bearing type used throughout the KAt-DataCenter (Paderborn) test rig -- a
#: type 6203 deep groove ball bearing, 0 degree contact angle. Source: the per-bearing "Profile
#: of rolling bearing damage" datasheet shipped inside every KAt-DataCenter archive (e.g.
#: K001.pdf -> Manufacturer specific information -> Geometry), not an assumed/textbook value.
KAT_DATACENTER_6203_GEOMETRY = BearingGeometry(
    n_elements=8,
    element_diameter_m=6.75e-3,
    pitch_diameter_m=29.05e-3,
    contact_angle_rad=0.0,
)
