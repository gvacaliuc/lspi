import gym
from .util import random_discretization


class RLAgent(object):
    """ Abstract base class for reinforcement learning agents. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reset(self):
        """
        Reset the agent and the contained environment.

        :return: An initial observation.
        """
        pass

    @abc.abstractmethod
    def step(self, episode_num, step_num):
        """
        Take a step in the previously initialized environment from the input state.

        :return: The previous state, action taken, reward received, and next (aka current) state, as well
        as whether the environment reported that the current episode is done.
        """
        pass


class RandomAgent(RLAgent):
    """
    Class that implements the RLAgent contract by taking random steps.
    """

    def __init__(self, env):
        """
        Stores a reference to the gym environment and sets up our state
        bookeeping.
        """

        self._env = env
        self._current_state = None

    def reset(self):
        """
        Reset the agent and the contained environment.

        :return: An initial observation.
        """

        # https://gym.openai.com/docs/#environments
        self._current_state = self._env.reset()
        return self._current_state

    def step(self, episode_num, step_num, **kwargs):
        """
        Take a step in the previously initialized environment from the input state.

        :return: The previous state, action taken, reward received, and next (aka current) state, as well
        as whether the environment reported that the current episode is done.

        :return: (state, action, reward, next (aka current) state, done)
        """

        # sample action
        action = self._env.action_space.sample()
        previous_state = self._current_state
        self._current_state, reward, done, info = self._env.step(action)

        return previous_state, action, reward, self._current_state, done


class PolicyAgent(RLAgent):
    def __init__(self, env, policy):

        self._env = env
        self._policy = policy

        self._current_state = None

    def reset(self):
        """
        Reset the agent and the contained environment.

        :return: An initial observation.
        """

        # https://gym.openai.com/docs/#environments
        self._current_state = self._env.reset()
        return self._current_state

    def step(self, episode_num, step_num, **kwargs):
        """
        Take a step in the previously initialized environment from the input state.

        :return: The previous state, action taken, reward received, and next (aka current) state, as well
        as whether the environment reported that the current episode is done.

        :return: (state, action, reward, next (aka current) state, done)
        """

        action = self._policy(self._current_state)
        previous_state = self._current_state
        self._current_state, reward, done, info = self._env.step(action)

        return previous_state, action, reward, self._current_state, done
