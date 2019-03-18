import abc

import gym
import numpy as np

from .basis import Basis


class Policy(object):
    @abc.abstractmethod
    def __call__(self, state):
        """
        Returns an action to take given a state.
        """
        pass

    @abc.abstractmethod
    def update(self, weights):
        """
        Returns the "difference" between the current weights, and new weights.
        """
        pass


class RandomPolicy(Policy):
    def __init__(self, env):
        """
        Creates a random policy.
        """
        self._env = env

    def __call__(self, state):
        # assert state in self._env.observation_space
        return self._env.action_space.sample()

    def update(self, weights):
        return 0.0


class DiscreteActionBasisPolicy(Policy):
    """
    Implements a policy which operates over a discrete action space.

    Example Usage:
        basis = IndActionPolyStateBasis(env.action_space, env.observation_space, order=2)
        dabp = DiscreteActionBasisPolicy(env.action_space, basis, lstdq.weights_)
        dabp(1)
    """

    def __init__(self, space, basis, weights):

        self._space = space

        assert basis.rank == len(weights)
        self._basis = basis
        self._weights = weights

        self.history_ = [self._weights]

    def __call__(self, state):
        """
        Returns the optimal action at this state.
        """

        qvalues = np.zeros(self._space.n)
        for action in range(self._space.n):
            phi = self._basis(state, action)
            qvalues[action] = phi.dot(self._weights)

        return np.argmax(qvalues)

    def update(self, weights):
        """
        Updates the weights of this policy.
        """

        self.history_.append(weights)

        diff = np.linalg.norm(self._weights - weights)
        self._weights = weights
        return diff


class ExplorationPolicy(Policy):
    """
    Implements a policy which occasionally executes a random action.
    """

    def __init__(self, env, policy, rate=0.9):
        """
        policy(state) is executed rate * 100 % of the time.
        """

        self._rpolicy = RandomPolicy(env)
        self._policy = policy
        self._rate = rate

    def __call__(self, state):
        """
        Returns the optimal action at this state.
        """

        return self._policy(
            state) if np.random.rand() < self._rate else self._rpolicy(state)

    def update(self, weights):
        """
        Updates the weights of this policy.
        """

        raise NotImplemented("Cannot update an immutable ExplorationPolicy")
