import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class NChainEnv(gym.Env):
    """n-Chain environment
    
    Linear chain, you control where the rewards are.
    """

    def __init__(self, reward=np.array([0, 0, 0, 0, 1]), slip=0.2, wrap=False):
        self.reward = reward
        self.n = len(reward)
        self.slip = slip  # probability of 'slipping' an action
        self.wrap = wrap
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if self.np_random.rand() < self.slip:
            action = (1 - action)  # agent slipped, reverse action taken

        # action = 1, move forward
        if action == 1:
            self.state = (self.state + 1) % self.n if self.wrap else min(
                self.state + 1, self.n - 1)
            reward = self.reward[self.state]
        else:
            self.state = (self.state - 1) % self.n if self.wrap else max(
                self.state - 1, 0)
            reward = self.reward[self.state]

        done = False
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        return self.state
