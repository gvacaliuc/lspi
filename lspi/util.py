# from collections.abc import Generator

from itertools import product

import gym
import numpy as np

def generate_sample_data(num_samples, env, policy):
    """
    Example Usage:
        data = generate_sample_data(1e3, env, RandomPolicy(env))
    """
    state = env.reset()

    data = []

    for _ in range(int(num_samples)):
        action = policy(state)
        nextstate, reward, done, _ = env.step(action)
        data.append((state, action, reward, nextstate))

        if done:
            state = env.reset()
        else:
            state = nextstate

    return data


class SampleGenerator(object):
    def __init__(self, env, policy):

        self._env = env
        self._policy = policy

    def sample(self, rollouts, timesteps):
        data = []

        for episode in xrange(int(rollouts)):

            state = self._env.reset()

            for step in xrange(int(timesteps)):

                action = self._policy(state)
                nextstate, reward, done, _ = self._env.step(action)
                data.append((state, action, reward, nextstate))
                state = nextstate

                if done:
                    break

        return data


def BoxAnchorGen(n, space, low=-np.inf, high=np.inf, seed=None):
    """
        Creates a generator from which to sample anchor points from.
        Also provides options to bound the space.  low and high can either
        be scalars or numpy arrays.
        """

    _space = gym.spaces.Box(
        low=space.low.clip(min=low),
        high=space.high.clip(max=high),
        # dtype=space.dtype
    )
    if seed:
        _space.seed(seed)
    _n = n
    _curr = 0

    while _curr < _n:
        _curr += 1
        yield _space.sample()


def random_discretization(space, n):
    return [space.sample() for _ in range(n)]


def ur5_discretizer(space, n, low=-np.inf, high=np.inf):
    """
    Returns a set of n valid actions in the provided space.
    """

    assert type(space) == gym.spaces.Tuple
    assert len(space.spaces) == 1
    assert type(space.spaces[0]) == gym.spaces.Box

    # for each dimension in the space we want to get a set of 
    # linearly discretized actions to take
    joints = space.spaces[0].shape[0]
    low = space.spaces[0].low.clip(min=low)
    high = space.spaces[0].high.clip(max=high)
    n = n if hasattr(n, "__iter__") else [n] * joints

    actions_per_joint = []

    for num, lower_bound, upper_bound in zip(n, low, high):
        actions_per_joint.append(np.linspace(lower_bound, upper_bound, num))

    return [(np.array(actions),) for actions in product(*actions_per_joint)]


def linear_discretization(space, n):
    if type(space) is gym.spaces.Box and space.shape == (1, ):
        return [
            np.array([action])
            for action in np.linspace(space.low[0], space.high[0], num=n)
        ]

    else:
        raise NotImplemented("Spaces other than Box")


class DiscreteEnvWrapper(gym.Env):
    def __init__(self, env, actions):
        """
        env: gym.Env
        disc_strat: Callable that accepts an action space and returns discrete actions to take.
        """

        self._env = env
        self._actions = actions
        self.action_space = gym.spaces.Discrete(len(self._actions))
        self.observation_space = self._env.observation_space

    def seed(self, seed):
        return self._env.seed()

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(self._actions[action])


def toArray(obj):
    if type(obj) is tuple:
        return np.hstack(obj)
    if type(obj) is int or type(obj) is float:
        return np.array([obj])
    if type(obj) is list:
        return np.array(obj)
    return obj
