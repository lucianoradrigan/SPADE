"""Minimal PhysicalSystem base class.

Ported from ``gym_electric_motor.core.PhysicalSystem`` (GEM 3.0.3). The original class does not
depend on ``gymnasium.Env`` itself -- it is a plain object that simply stores whatever
``action_space``/``state_space`` objects it is given. This port keeps it exactly as-is, without the
surrounding ``core.py`` module (which pulls in ``gymnasium.core.Env`` for other, unrelated classes).
"""


class PhysicalSystem:
    """The Physical System module encapsulates the physical model of the system as well as the
    simulation from one step to the next."""

    @property
    def tau(self):
        return self._tau

    @property
    def unwrapped(self):
        """Returns this instance of the physical system.

        If the system is wrapped into multiple PhysicalSystemWrappers this property returns
        directly the innermost system.
        """
        return self

    @property
    def k(self):
        """
        Returns:
             int: The current systems time step k.
        """
        return self._k

    @property
    def state_names(self):
        """
        Returns:
             ndarray(str): Array containing the names of the systems states.
        """
        return self._state_names

    @property
    def state_positions(self):
        """
        Returns:
            dict(int): Dictionary mapping the state names to its positions in the state arrays
        """
        return self._state_positions

    @property
    def action_space(self):
        """
        Returns:
            Space: The set of allowed actions on the system.
        """
        return self._action_space

    @property
    def state_space(self):
        """
        Returns:
             Space: The set of possible systems states.
        """
        return self._state_space

    @property
    def limits(self):
        """
        Returns:
             ndarray(float): An array containing the maximum allowed physical values for each state
             variable.
        """
        return NotImplementedError

    @property
    def nominal_state(self):
        """
        Returns:
             ndarray(float): An array containing the nominal values for each state variable.
        """
        return NotImplementedError

    def __init__(self, action_space, state_space, state_names, tau):
        """
        Args:
            action_space(Space): The set of allowed actions on the system.
            state_space(Space): The set of possible systems states.
            state_names(ndarray(str)): The names of the systems states
            tau(float): The systems simulation time interval.
        """
        self._action_space = action_space
        self._state_space = state_space
        self._state_names = state_names
        self._state_positions = {key: index for index, key in enumerate(self._state_names)}
        self._tau = tau
        self._k = 0

    def reset(self, initial_state=None):
        """
        Reset the physical system to an initial state before a new episode starts.

        Returns:
             element of state_space: The initial systems state
        """
        raise NotImplementedError

    def simulate(self, action):
        """
        Simulation of the Physical System for one time step with the input action.

        Args:
            action(element of action_space): The action to play on the system for the next time step.

        Returns:
            element of state_space: The systems state after the action was applied.
        """
        raise NotImplementedError

    def close(self):
        """
        Called, when the simulation is closed.
        Close the System and all of its submodules by closing files, saving logs etc.
        """
        pass
