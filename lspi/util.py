from collections.abc import Generator

import gym
import numpy as np

from .policy import Policy

def generate_sample_data(num_samples, env: gym.Env, policy: Policy):
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

class BoxAnchorGen(Generator):
    def __init__(self, n: int, space: gym.spaces.Box, low = -np.inf, high = np.inf, seed: int = None):
        """
        Creates a generator from which to sample anchor points from.
        Also provides options to bound the space.  low and high can either
        be scalars or numpy arrays.
        """

        self._space = gym.spaces.Box(
            low=space.low.clip(min=low),
            high=space.high.clip(max=high),
            dtype=space.dtype
        )
        if seed:
            self._space.seed(seed)
        self._n = n
        self._curr = 0

    def send(self, _):
        if self._curr < self._n:
            self._curr += 1
            return self._space.sample()
        raise StopIteration

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration
