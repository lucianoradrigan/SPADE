"""Mechanical loads attached to the electrical motor's rotor.

Ported from ``gym_electric_motor.physical_systems.mechanical_loads`` (GEM 3.0.3), which is a
5-file subpackage there (``mechanical_load.py``, ``constant_speed_load.py``,
``external_speed_load.py``, ``ornstein_uhlenbeck_load.py``, ``polynomial_static_load.py``).
Consolidated into a single module here per the repository layout in
``docs/propuesta_consolidacion.pdf`` (Sec. 4.6). ``BearingFaultLoad`` (Addendum, Macro-fase A.4)
will be added to this module later, reusing ``sim/vibration/bearing_frequencies.py``.
"""

import warnings

import numpy as np
from scipy.stats import truncnorm

from .random_component import RandomComponent
from .utils import update_parameter_dict


class MechanicalLoad(RandomComponent):
    """The MechanicalLoad is the base class for all the mechanical systems attached
    to the electrical motors rotor.

    It contains an mechanical ode system as well as the state names, limits and
    nominal values of the mechanical quantities. The only required state is
    'omega' as the rotational speed of the motor shaft in rad/s.
    ConstantSpeedLoad can be initialized with the initializer as an
    class parameter by instantiation. ExternalSpeedLoad takes the first value
    of the SpeedProfile as initial value.

    Initialization is given by initializer(dict). Can be a constant state value
    or random value in given interval.
    dict should be like:
        { 'states'(dict): with state names and initial values
          'interval'(array like): boundaries for each state
                    (only for random init), shape(num states, 2)
          'random_init'(str): 'uniform' or 'normal'
          'random_params(tuple): mue(float), sigma(int)

    Example initializer(dict) for constant initialization:
        { 'states': {'omega': 16.0}}
    Example  initializer(dict) for random initialization:
        { 'random_init': 'normal'}
    """

    @property
    def j_total(self):
        """
        Returns:
             float: Total moment of inertia affecting the motor shaft.
        """
        return self._j_total

    @property
    def state_names(self):
        """
        Returns:
            list(str): Names of the states in the mechanical-ODE.
        """
        return self._state_names

    @property
    def limits(self):
        """
        Returns:
            dict(float): Mapping of the motor states to their limit values.
        """
        return self._limits

    @property
    def nominal_values(self):
        """
        Returns:
              dict(float): Mapping of the motor states to their nominal values

        """
        return self._nominal_values

    @property
    def initializer(self):
        """
        Returns:
            dict: The motors initial state and additional initializer parameters
        """
        return self._initializer

    OMEGA_IDX = 0

    #: Parameter indicating if the class is implementing the optional jacobian function
    HAS_JACOBIAN = False

    #: _default_initial_state(dict): Default initial motor-state values
    _default_initializer = {}

    def __init__(self, state_names=None, j_load=0.0, load_initializer=None):
        """
        Args:
            state_names(list(str)): List of the names of the states in the mechanical-ODE.
            j_load(float): Moment of inertia of the load affecting the motor shaft.
        """
        RandomComponent.__init__(self)
        self._j_total = self._j_load = j_load
        self._state_names = list(state_names or ["omega"])
        self._limits = {}
        self._nominal_values = {}
        load_initializer = load_initializer or {}
        self._initializer = self._default_initializer.copy()
        self._initializer.update(load_initializer)
        self._initial_states = self._initializer.get("states", {state: 0.0 for state in self._state_names})

    def initialize(self, state_space, state_positions, nominal_state, **__):
        """Initializes the state of the load on an episode start.

        Values can be given as a constant or sampled random out of a statistical distribution. Initial value is in
        range of the nominal values or a given interval.

        Args:
            nominal_state(list): nominal values for each state given from physical system
            state_space(Box): normalized state space boundaries
            state_positions(dict): indexes of system states
        """
        # for order and organization purposes
        interval = self._initializer["interval"]
        random_dist = self._initializer["random_init"]
        random_params = self._initializer["random_params"]
        if isinstance(nominal_state, (list, np.ndarray)):
            nominal_state = np.asarray(nominal_state, dtype=float)
        elif isinstance(self._nominal_values, dict):
            nominal_state = [nominal_state[state] for state in self._initial_states.keys()]
            nominal_state = np.asarray(nominal_state)
        # setting nominal values as interval limits
        state_idx = [state_positions[state] for state in self._initial_states.keys()]
        upper_bound = nominal_state[state_idx]
        lower_bound = upper_bound * np.asarray(state_space.low, dtype=float)[state_idx]
        # clip nominal boundaries to user defined
        if interval is not None:
            lower_bound = np.clip(lower_bound, a_min=np.asarray(interval, dtype=float).T[0], a_max=None)
            upper_bound = np.clip(upper_bound, a_min=None, a_max=np.asarray(interval, dtype=float).T[1])
        else:
            pass
        # random initialization for each load state (omega)
        if random_dist is not None:
            if random_dist == "uniform":
                initial_value = (upper_bound - lower_bound) * self.random_generator.uniform(
                    size=len(self._initial_states.keys())
                ) + lower_bound
                random_states = {state: initial_value[idx] for idx, state in enumerate(self._initial_states.keys())}
                self._initial_states.update(random_states)

            elif random_dist in ["normal", "gaussian"]:
                # specific input or middle of interval
                mue = random_params[0] or (upper_bound - lower_bound) / 2 + lower_bound
                sigma = random_params[1] or 1
                a = (lower_bound - mue) / sigma
                b = (upper_bound - mue) / sigma
                initial_value = truncnorm.rvs(
                    a,
                    b,
                    loc=mue,
                    scale=sigma,
                    size=(len(self._initial_states.keys())),
                    random_state=self.seed_sequence.pool[0],
                )
                random_states = {state: initial_value[idx] for idx, state in enumerate(self._initial_states.keys())}
                self._initial_states.update(random_states)
            else:
                raise NotImplementedError
        # constant initialization for each motor state (current, epsilon)
        elif self._initial_states is not None:
            initial_value = np.atleast_1d(list(self._initial_states.values()))
            # check init_value meets interval boundaries
            if (lower_bound <= initial_value).all() and (initial_value <= upper_bound).all():
                initial_states_ = {state: initial_value[idx] for idx, state in enumerate(self._initial_states.keys())}
                self._initial_states.update(initial_states_)
            else:
                raise Exception("Initialization Value have to be in nominal boundaries")
        else:
            raise Exception("No matching Initialization Case")

    def reset(self, state_space, state_positions, nominal_state, **__):
        """
        Reset the motors state to a new initial state. (Default 0)

        Args:
            nominal_state(list): nominal values for each state given from
                                  physical system
            state_space(Box): normalized state space boundaries
            state_positions(dict): indexes of system states
        Returns:
            numpy.ndarray(float): The initial motor states.
        """
        self.next_generator()
        if self._initializer:
            self.initialize(state_space, state_positions, nominal_state)
            return np.asarray(list(self._initial_states.values()))
        else:
            return np.zeros_like(self._state_names, dtype=float)

    def set_j_rotor(self, j_rotor):
        """
        Args:
            j_rotor(float): The moment of inertia of the rotor shaft of the motor.
        """
        self._j_total += j_rotor

    def mechanical_ode(self, t, mechanical_state, torque):
        """
        Calculation of the derivatives of the mechanical-ODE for each of the mechanical states.

        Args:
            t(float): Current time of the system.
            mechanical_state(ndarray(float)): Current state of the mechanical system.
            torque(float): Generated input torque by the electrical motor.

        Returns:
            ndarray(float): Derivatives of the mechanical state for the given input torque.
        """
        raise NotImplementedError

    def mechanical_jacobian(self, t, mechanical_state, torque):
        """
        Calculation of the jacobians of the mechanical-ODE for each of the mechanical state.

        Overriding this method is optional for each subclass. If it is overridden, the parameter HAS_JACOBIAN must also
        be set to True. Otherwise, the jacobian will not be called.

        Args:
            t(float): Current time of the system.
            mechanical_state(ndarray(float)): Current state of the mechanical system.
            torque(float): Generated input torque by the electrical motor.

        Returns:
            Tuple(ndarray, ndarray):
                [0]: Derivatives of the mechanical_state-odes over the mechanical_states shape:(states x states)
                [1]: Derivatives of the mechanical_state-odes over the torque shape:(states,)
        """
        pass

    def get_state_space(self, omega_range):
        """
        Args:
            omega_range(Tuple(int,int)): Lower and upper values the motor can generate for omega normalized to (-1, 1)

        Returns:
            Tuple(dict,dict): Lowest and highest possible values for all states normalized to (-1, 1)
        """
        return {"omega": omega_range[0]}, {"omega": omega_range[1]}


class ConstantSpeedLoad(MechanicalLoad):
    """
    Constant speed mechanical load system which will always set the speed
    to a predefined value.
    """

    HAS_JACOBIAN = True
    _default_initializer = {
        "states": {"omega": 0.0},
        "interval": None,
        "random_init": None,
        "random_params": (None, None),
    }

    @property
    def omega_fixed(self):
        """
        Returns:
            float: Constant value for omega in rad/s.
        """
        return self._omega

    def __init__(self, omega_fixed=0, load_initializer=None, **kwargs):
        """
        Args:
            omega_fixed(float)): Fix value for the speed in rad/s.
        """
        super().__init__(load_initializer=load_initializer, **kwargs)
        self._omega = omega_fixed or self._initializer["states"]["omega"]
        if omega_fixed != 0:
            self._initializer["states"]["omega"] = omega_fixed
        self._ode_result = np.array([0.0])
        self._jacobian_result = (np.array([[0.0]]), np.array([0.0]))

    def mechanical_ode(self, *_, **__):
        # Docstring of superclass
        return self._ode_result

    def mechanical_jacobian(self, t, mechanical_state, torque):
        # Docstring of superclass
        return self._jacobian_result


class ExternalSpeedLoad(MechanicalLoad):
    """
    External speed mechanical load system which will set the speed to a
    predefined speed-function/ speed-profile.
    """

    HAS_JACOBIAN = False

    @property
    def omega(self):
        """
        Returns:
            float: Function-value for omega in rad/s at time-step t.
        """
        return self._omega_initial

    def __init__(
        self,
        speed_profile,
        load_initializer=None,
        tau=1e-4,
        speed_profile_kwargs=None,
        **kwargs,
    ):
        """
        Args:
            speed_profile(float -> float): A callable(t, **speed_profile_args) -> float
                which takes a timestep t and custom further arguments and returns a speed omega
                example:
                    (lambda t, amplitude, freq: amplitude*numpy.sin(2*pi*f)))
                    with additional parameters:
                        amplitude(float), freq(float), time(float)
            tau(float): discrete time step of the system
            speed_profile_kwargs(dict): further arguments for speed_profile
            kwargs(dict): Arguments to be passed to superclass :py:class:`.MechanicalLoad`

        """
        super().__init__(**kwargs)
        speed_profile_kwargs = speed_profile_kwargs or {}
        if load_initializer is not None:
            warnings.warn(
                "Given initializer will be overwritten with starting value "
                "from speed-profile, to avoid complications at the load reset."
                " It is recommended to choose starting value of"
                " load by the defined speed-profile.",
                UserWarning,
            )

        self.speed_profile_kwargs = speed_profile_kwargs
        self._speed_profile = speed_profile
        self._tau = tau
        # setting initial load as speed-profile at time 0
        self._omega_initial = self._speed_profile(t=0, **self.speed_profile_kwargs)

    def mechanical_ode(self, t, mechanical_state, torque=None):
        # Docstring of superclass
        # calc next omega with given profile und tau
        omega_next = self._speed_profile(t=t + self._tau, **self.speed_profile_kwargs)
        # calculated T out of euler-forward, given omega_next and
        # actual omega give from system
        return np.array([(1 / self._tau) * (omega_next - mechanical_state[self.OMEGA_IDX])])

    def mechanical_jacobian(self, t, mechanical_state, torque):
        # Docstring of superclass
        # jacobian here not necessary, since omega is externally given
        return None

    def reset(self, **kwargs):
        # Docstring of superclass
        return np.asarray(self._omega_initial, dtype=float)[None]


class OrnsteinUhlenbeckLoad(MechanicalLoad):
    """The Ornstein-Uhlenbeck Load sets the speed to a torque-independent signal specified by the underlying OU-Process."""

    HAS_JACOBIAN = False

    def __init__(self, mu=0, sigma=1e-4, theta=1, tau=1e-4, omega_range=(-200.0, 200.0), **kwargs):
        """
        Args:
            mu(float): Mean value of the underlying gaussian distribution of the OU-Process.
            sigma(float): Standard deviation of the underlying gaussian distribution of the  OU-Process.
            theta(float): Drift towards the mean of the OU-Process.
            tau(float): discrete time step of the system
            omega_range(2-Tuple(float)): Minimal and maximal value for the process.
            kwargs(dict): further arguments passed to the superclass :py:class:`.MechanicalLoad`
        """
        super().__init__(**kwargs)
        self._omega = np.random.uniform(self._omega_range[0], self._omega_range[1], 1)
        self.theta = theta
        self.mu = mu
        self.tau = tau
        self.sigma = sigma
        self._omega_range = omega_range

    def mechanical_ode(self, t, mechanical_state, torque):
        omega = mechanical_state
        max_diff = (self._omega_range[1] - omega) / self.tau
        min_diff = (self._omega_range[0] - omega) / self.tau
        diff = self.theta * (self.mu - omega) * self.tau + self.sigma * np.sqrt(self.tau) * np.random.normal(size=1)
        np.clip(diff, min_diff, max_diff, out=diff)
        return diff

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._omega = np.random.uniform(self._omega_range[0], self._omega_range[1], 1)
        return self._omega


class PolynomialStaticLoad(MechanicalLoad):
    """Mechanical system that models the Mechanical-ODE based on a static polynomial load torque.

    Parameter dictionary entries:
        - :math:`a / Nm`: Constant Load Torque coefficient (for modeling static friction)
        - :math:`b / (Nm s)`: Linear Load Torque coefficient (for modeling sliding friction)
        - :math:`c / (Nm s^2)`: Quadratic Load Torque coefficient (for modeling air resistances)
        - :math:`j_load / (kg m^2)` : Moment of inertia of the mechanical system.

    Usage Example:
        >>> from driveflow.sim.loads import PolynomialStaticLoad
        >>>
        >>> my_poly_static_load = PolynomialStaticLoad(
        ...     load_parameter=dict(a=1e-3, b=1e-4, c=0.0, j_load=1e-3),
        ...     limits=dict(omega=150.0),  # rad / s
        ... )
    """

    _load_parameter = dict(a=0.0, b=0.0, c=0.0, j_load=1e-5)
    _default_initializer = {
        "states": {"omega": 0.0},
        "interval": None,
        "random_init": None,
        "random_params": (None, None),
    }

    #: Time constant to smoothen the static load functions constant term "a" around 0 velocity
    # Steps of a lead to unstable behavior of the ode-solver.
    tau_decay = 1e-3

    #: Parameter indicating if the class is implementing the optional jacobian function
    HAS_JACOBIAN = True

    @property
    def load_parameter(self):
        """
        Returns:
            dict(float): Parameter dictionary of the load.
        """
        return self._load_parameter

    def set_j_rotor(self, j_rotor):
        # Docstring of superclass
        super().set_j_rotor(j_rotor)
        self._omega_linear_factor = self._j_total / self.tau_decay
        self._omega_lim = self._a / self._j_total * self.tau_decay

    def __init__(self, load_parameter=None, limits=None, load_initializer=None):
        """
        Args:
            load_parameter(dict(float)): Parameter dictionary. Keys: ``'a', 'b', 'c', 'j_load'``
            limits(dict): dictionary to update the limits of the load-instance. Keys: ``'omega'``
            load_initializer(dict): Dictionary to parameterize the initializer.
        """
        load_parameter = load_parameter if load_parameter is not None else dict()
        self._load_parameter = update_parameter_dict(self._load_parameter, load_parameter)
        super().__init__(j_load=self._load_parameter["j_load"], load_initializer=load_initializer)
        self._limits.update(limits or {})
        self._a = self._load_parameter["a"]
        self._b = self._load_parameter["b"]
        self._c = self._load_parameter["c"]
        # Speed value at which the linear behavior switches to constant
        self._omega_lim = self._a / self._j_total * self.tau_decay
        # Slope for the linear growth of the constant load part around zero speed
        self._omega_linear_factor = self._j_total / self.tau_decay

    def _static_load(self, omega):
        """Calculation of the load torque for a given speed omega."""
        sign = 1 if omega > 0 else -1 if omega < -0 else 0
        # Limit the constant load term 'a' for velocities around zero for a more stable integration
        a = sign * self._a if abs(omega) > self._omega_lim else self._omega_linear_factor * omega
        return sign * self._c * omega**2 + self._b * omega + a

    def mechanical_ode(self, t, mechanical_state, torque):
        # Docstring of superclass
        omega = mechanical_state[self.OMEGA_IDX]
        static_torque = self._static_load(omega)
        total_torque = torque - static_torque
        return np.array([total_torque / self._j_total])

    def mechanical_jacobian(self, t, mechanical_state, torque):
        # Docstring of superclass
        omega = mechanical_state[self.OMEGA_IDX]
        sign = 1 if omega > 0 else -1 if omega < 0 else 0
        # Linear region of the constant load term 'a' ?
        a = 0 if abs(omega) > self._a * self.tau_decay / self._j_total else self._j_total / self.tau_decay
        return np.array([[(-self._b - 2 * sign * self._c * omega - a) / self._j_total]]), np.array([1 / self._j_total])
