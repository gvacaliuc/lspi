import abc

import gym
import numpy as np


class Basis(object, metaclass=abc.ABCMeta):
    """
    Class to represent our basis functions.
    """

    @abc.abstractmethod
    def __call__(self, state, action) -> np.ndarray:
        """
        Returns a numpy array of each basis function evaluated at
        the state and action specified.
        """

        pass

    @abc.abstractproperty
    def rank(self) -> int:
        """
        Returns the rank of this basis.
        """

        pass


class IndActionPolyStateBasis(Basis):
    """
    Implements a basis that provides bases polynomial in state,
    each of which is multipled by an indicator varible for each
    action.

    Example Usage:
        basis = IndActionPolyStateBasis(action_space=gym.spaces.Discrete(2), observation_space=gym.spaces.Discrete(10), order=2)
        basis(2, 1)

        basis = IndActionPolyStateBasis(action_space=gym.spaces.Discrete(2), observation_space=gym.make("CartPole-v0").observation_space, order=2)
        basis(np.ones(4) * 0.1, 1)
    """

    def __init__(self,
                 action_space: gym.spaces.Discrete,
                 observation_space: gym.spaces.Space,
                 order: int = 2):

        self._action_space = action_space
        self._observation_space = observation_space

        self._num_actions = self._action_space.n
        self._observation_dim = np.product(
            self._observation_space.shape) if len(
                self._observation_space.shape) else 1

        # rank of this basis
        self._order = order
        self._rank = self._num_actions * self._observation_dim * (
            self._order + 1)

    @property
    def rank(self):
        return self._rank

    def _eval_polynomial(self, state):
        """
        Evaluates our polynomial basis functions at each element of the given state,
        which is either a single number or an array.
        """

        return np.array(
            [state**power for power in range(self._order + 1)]).flatten()

    def __call__(self, state, action):
        """
        Computes the function ϕ(s, a) for this basis.
        """

        assert state in self._observation_space
        assert action in self._action_space

        phi = np.zeros(self.rank)
        dim = self.rank // self._num_actions

        phi[action * dim:(action + 1) * dim] = self._eval_polynomial(state)

        return phi
