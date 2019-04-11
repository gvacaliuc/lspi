import numpy as np
from numpy.linalg import LinAlgError

from .basis import Basis
from .policy import Policy


class LSTDQ(object):
    def __init__(self, basis, discount=0.9, delta=1e-3):

        self._basis = basis
        self._k = self._basis.rank
        self._discount = discount
        self._delta = delta

    def fit(self, data, policy):
        """
        Performs an iteration of LSTDQ.
        """

        B = np.eye(self._k) / self._delta
        b = np.zeros((self._k, ))

        for state, action, reward, nextstate in data:
            phi = self._basis(state, action)
            phiprime = self._basis(nextstate, policy(nextstate))

            num = B.dot(np.outer(phi, phi - self._discount * phiprime)).dot(B) 
            denom = 1 + (phi - self._discount * phiprime).T.dot(B).dot(phi)

            B -= num / denom
            b += reward * phi

        self.weights_ = B.dot(b)

        return self


    def fit_slow(self, data, policy):
        """
        Performs an iteration of LSTDQ.
        """

        A = np.eye(self._k) * self._delta
        b = np.zeros((self._k, ))

        for state, action, reward, nextstate in data:
            phi = self._basis(state, action)
            phiprime = self._basis(nextstate, policy(nextstate))

            A += np.outer(phi, phi - self._discount * phiprime)
            b += reward * phi

        try:
            self.weights_ = np.linalg.inv(A).dot(b)
        except LinAlgError as e:
            print("A is uninvertable:\n {A}".format(A=A))
            self.weights_ = np.zeros((self._k, ))

        return self


class LSPI(object):
    """
    Implements the Least-Squares Policy Iteration algorithm.

    Example Usage:
        data = generate_sample_data(1e3, env, RandomPolicy(env))
        lstdq = LSTDQ(basis, discount=0.7)
        basis = IndActionPolyStateBasis(env.action_space, env.observation_space, order = 2)
        dabp = DiscreteActionBasisPolicy(env.action_space, basis, np.zeros(basis.rank))
        lspi = LSPI(lstdq, dabp, max_iter=50, epsilon=1e-5)
        lspi.fit(data)
    """

    def __init__(self, optimizer, policy, max_iter=50, epsilon=1e-3, verbose=False):

        self._optimizer = optimizer
        self._policy = policy

        self._max_iter = max_iter
        self._epsilon = epsilon

        self._verbose = verbose

    def fit(self, data):

        for self.itr_ in range(self._max_iter):
            self._optimizer.fit(data, self._policy)
            diff = self._policy.update(self._optimizer.weights_)
            if (diff < self._epsilon):
                break

            if self._verbose:
                print("Iteration {}: Diff: {}".format(self.itr_, diff))

        return self

    @property
    def policy(self):
        return self._policy
