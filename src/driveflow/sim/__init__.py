"""Physical simulation layer: motors, converters, loads, supplies, solvers, SCMLSystem.

Ported from ``gym_electric_motor.physical_systems`` (GEM 3.0.3), decoupled from
``gymnasium.Env`` -- ``SCMLSystem`` is used directly via ``reset()``/``simulate(action)``, without
going through GEM's ``ElectricMotorEnvironment``, reference generators, reward functions or the
~90-entry Gym env registry.
"""

from .converters import (
    ContB6BridgeConverter,
    ContFourQuadrantConverter,
    ContMultiConverter,
    ContOneQuadrantConverter,
    ContTwoQuadrantConverter,
    FiniteB6BridgeConverter,
    FiniteFourQuadrantConverter,
    FiniteMultiConverter,
    FiniteOneQuadrantConverter,
    FiniteTwoQuadrantConverter,
    NoConverter,
    PowerElectronicConverter,
)
from .loads import (
    ConstantSpeedLoad,
    ExternalSpeedLoad,
    MechanicalLoad,
    OrnsteinUhlenbeckLoad,
    PolynomialStaticLoad,
)
from .motors import (
    DcExternallyExcitedMotor,
    DcPermanentlyExcitedMotor,
    DcSeriesMotor,
    DcShuntMotor,
    DoublyFedInductionMotor,
    ElectricMotor,
    ExternallyExcitedSynchronousMotor,
    PermanentMagnetSynchronousMotor,
    SquirrelCageInductionMotor,
    SynchronousReluctanceMotor,
    ThreePhaseMotor,
)
from .physical_system import PhysicalSystem
from .scml_system import (
    DcMotorSystem,
    DoublyFedInductionMotorSystem,
    ExternallyExcitedSynchronousMotorSystem,
    SCMLSystem,
    SquirrelCageInductionMotorSystem,
    SynchronousMotorSystem,
    ThreePhaseMotorSystem,
)
from .solvers import EulerSolver, OdeSolver, ScipyOdeIntSolver, ScipyOdeSolver, ScipySolveIvpSolver
from .supplies import AC1PhaseSupply, AC3PhaseSupply, IdealVoltageSupply, RCVoltageSupply, VoltageSupply

__all__ = [
    "ContB6BridgeConverter",
    "ContFourQuadrantConverter",
    "ContMultiConverter",
    "ContOneQuadrantConverter",
    "ContTwoQuadrantConverter",
    "FiniteB6BridgeConverter",
    "FiniteFourQuadrantConverter",
    "FiniteMultiConverter",
    "FiniteOneQuadrantConverter",
    "FiniteTwoQuadrantConverter",
    "NoConverter",
    "PowerElectronicConverter",
    "ConstantSpeedLoad",
    "ExternalSpeedLoad",
    "MechanicalLoad",
    "OrnsteinUhlenbeckLoad",
    "PolynomialStaticLoad",
    "DcExternallyExcitedMotor",
    "DcPermanentlyExcitedMotor",
    "DcSeriesMotor",
    "DcShuntMotor",
    "DoublyFedInductionMotor",
    "ElectricMotor",
    "ExternallyExcitedSynchronousMotor",
    "PermanentMagnetSynchronousMotor",
    "SquirrelCageInductionMotor",
    "SynchronousReluctanceMotor",
    "ThreePhaseMotor",
    "PhysicalSystem",
    "DcMotorSystem",
    "DoublyFedInductionMotorSystem",
    "ExternallyExcitedSynchronousMotorSystem",
    "SCMLSystem",
    "SquirrelCageInductionMotorSystem",
    "SynchronousMotorSystem",
    "ThreePhaseMotorSystem",
    "EulerSolver",
    "OdeSolver",
    "ScipyOdeIntSolver",
    "ScipyOdeSolver",
    "ScipySolveIvpSolver",
    "AC1PhaseSupply",
    "AC3PhaseSupply",
    "IdealVoltageSupply",
    "RCVoltageSupply",
    "VoltageSupply",
]
