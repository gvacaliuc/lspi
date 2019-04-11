import abc

import gym
import numpy as np

from .util import toArray

class Basis(object):
    """
    Class to represent our basis functions.
    """

    @abc.abstractmethod
    def __call__(self, state, action):
        """
        Returns a numpy array of each basis function evaluated at
        the state and action specified.
        """

        pass

    @abc.abstractproperty
    def rank(self):
        """
        Returns the rank of this basis.
        """

        pass


# TODO: Pull out repeated code of the following bases into one class.


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

    def __init__(self, action_space, observation_space, order):

        self._action_space = action_space
        self._observation_space = observation_space

        self._num_actions = self._action_space.n
        self._observation_dim = np.product(
            self._observation_space.shape) if hasattr(self._observation_space,
                                                      "shape") else 1

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
        """

        assert self._observation_space.contains(state)
        assert self._action_space.contains(action)

        phi = np.zeros(self.rank)
        dim = self.rank // self._num_actions

        phi[action * dim:(action + 1) * dim] = self._eval_polynomial(state)

        return phi


class IndActionRBFStateBasis(Basis):
    """
    Implements a basis that provides bases which are non-linear radial-basis
    functions w.r.t state, each of which is multipled by an indicator varible
    for each action.

    Example Usage:
        basis = IndActionRBFStateBasis(list(range(env.observation_space.n)), env.action_space, env.observation_space)
        basis(0, 0)
    """

    def __init__(self,
                 anchors,
                 action_space,
                 observation_space,
                 metric="euclidean"):

        self._action_space = action_space
        self._observation_space = observation_space

        # make sure that all of the anchors are in the state space.
        self._anchors = list(anchors)
        bounds = list(
            zip(self._observation_space.low, self._observation_space.high))
        for anch in anchors:
            # assert anch in self._observation_space, "{anch} is not in {bounds}".format(anch=anch, bounds=bounds)
            pass

        self._num_actions = self._action_space.n
        self._observation_dim = len(self._anchors)
        self._rank = self._num_actions * self._observation_dim

        # allows us to change the metric later, as numpy supports quite a few.
        self._metric = metric

    @property
    def rank(self):
        return self._rank

    def _eval_rbf(self, state):
        """
        Evaluates our polynomial basis functions at each element of the given state,
        which is either a single number or an array.
        """

        # TODO: Isn't there another parameter here? RBF scale or gamma?

        return np.array([
            np.exp(-np.linalg.norm(state - anch) / 2.0)
            for anch in self._anchors
        ])

    def __call__(self, state, action):
        """
        """

        assert self._observation_space.contains(state)
        assert self._action_space.contains(action)

        phi = np.zeros(self.rank)
        dim = self.rank // self._num_actions

        phi[action * dim:(action + 1) * dim] = self._eval_rbf(state)

        return phi


class QuadraticBasis(Basis):
    def __init__(self, coefs, action_space, observation_space):

        self._coefs = coefs
        self._rank = len(self._coefs) + 1
        assert type(action_space) is gym.spaces.Discrete or type(
            action_space) is gym.spaces.Box
        self._action_space = action_space
        assert type(observation_space) is gym.spaces.Box
        self._observation_space = observation_space

        action = action_space.sample()
        action = np.array([action]) if type(action) is int else action
        state = observation_space.sample()
        z = np.hstack([action.flatten(), state.flatten()])

    @property
    def rank(self):
        return self._rank

    def __call__(self, state, action):
        action = np.array([action]) if type(action) is int else action
        z = np.hstack([action.flatten(), state.flatten()])
        return np.array([1] + [z.dot(coef).dot(z) for coef in self._coefs])
        # return np.array([z.dot(coef).dot(z) for coef in self._coefs])

class DiscreteQuadraticBasis(Basis):
    def __init__(self, coefs, actions, observation_space):

        self._coefs = coefs
        self._rank = len(self._coefs) + 1
        self._actions = actions
        self._action_space = gym.spaces.Discrete(len(self._actions))
        assert type(observation_space) is gym.spaces.Box
        self._observation_space = observation_space

        action = toArray(self._actions[0])
        state = observation_space.sample()
        z = np.hstack([action.flatten(), state.flatten()])

        for matrix in self._coefs:
            assert matrix.shape[0] == len(z) and matrix.shape[1] == len(z)

    @property
    def rank(self):
        return self._rank

    def __call__(self, state, action):
        action = toArray(self._actions[action])
        state = toArray(state)
        z = np.hstack([action.flatten(), state.flatten()])
        return np.array([1] + [z.dot(coef).dot(z) for coef in self._coefs])

class DiscreteQuadraticTupleBasis(Basis):
    def __init__(self, coef_list, actions, observation_space):
        self._bases = [
            DiscreteQuadraticBasis(coefs, actions, space)
            for coefs, space in zip(coef_list, observation_space.spaces)
        ]

    @property
    def rank(self):
        return sum([base.rank for base in self._bases])

    def __call__(self, states, action):
        return np.hstack(
            [base(state, action) for state, base in zip(states, self._bases)])



class DiscreteRBFTupleBasis(Basis):
    def __init__(self,
                 anchor_lists,
                 action_space,
                 observation_space):

        self._bases = [
            IndActionRBFStateBasis(anchors, action_space, space)
            for anchors, space in zip(anchor_lists, observation_space.spaces)
        ]

    @property
    def rank(self):
        return sum([base.rank for base in self._bases])

    def __call__(self, states, action):
        """
        """

        return np.concat(
            [base(state, action) for state, base in zip(states, self._bases)])


def genRandomMatrix(shape, mean=0.0, scale=1.0):
    return np.random.normal(loc=mean, scale=scale, size=shape)


def makeBasis(conf, actions, stateSpace):

    assert "type" in conf
    assert "size" in conf
    assert conf["type"] in ["quadratic", "rbf"]

    if conf["type"] == "quadratic":

        coefs = []
        for space in stateSpace.spaces:
            stateSize = len(toArray(actions[0])) + len(toArray(space.sample()))
            coefs.append([genRandomMatrix((stateSize, stateSize),
                                           conf.get("mean", 0.0),
                                           conf.get("scale", 1.0))
                           for _ in xrange(conf["size"])])


        return DiscreteQuadraticTupleBasis(coefs, actions, stateSpace)

    else:
        raise NotImplemented("Haven't handled rbf bases yet.")
