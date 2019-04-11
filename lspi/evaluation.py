"""
Stores classes and routines useful in evaluating policies.
"""

import argparse
import cPickle as pickle
import os
import yaml

import rosgym
import pandas as pd
import numpy as np

from lspi.util import ur5_discretizer, DiscreteEnvWrapper, SampleGenerator, toArray
from lspi.basis  import makeBasis
from lspi.optim import LSTDQ, LSPI
from lspi.policy import DiscreteActionBasisPolicy, RandomPolicy

p = argparse.ArgumentParser()
p.add_argument("-d,--directory", dest="directory", type=str, help="Directory to find the configuration file in.", required=True)
p.add_argument("-c,--config", dest="config", type=str, default="config.yml", help="Name of the configuration file to use.")

class Experiment(object):

    def __init__(self, directory, filename="config.yml"):
        """
        Runs the experiment in directory, looking for a configuration file
        under directory/filename.
        """

        self._directory = directory

        with open(os.path.join(directory, filename)) as f:
            conf = yaml.load(f)

        self._conf = conf

        assert "env" in conf
        assert "agent" in conf
        assert "experiment" in conf

    def initialize(self):
        self._ur5env = self._make_environment(self._conf["env"])
        self._make_agent(self._conf["agent"])

    def collect_data(self):
        self._data = self._collect_data(self._conf["experiment"])

    def train(self):
        self._fit()

    def evaluate(self):
        self._test(self._conf["experiment"])

    def _make_environment(self, envConf):
        """
        Creates an OpenAI Gym environment corresponding to the supplied robot
        configuration file, and configuration options.

        :return: rosgym environment
        """

        assert "robot_conf_file" in envConf
        assert "planning_group" in envConf
        assert "num_joints" in envConf


        # TODO: What are these limits actually used for?
        limits = -2.0, 2.0

        print("Making UR5 Environment with the following config:")
        print(envConf)

        # The environment provided behaves somewhat strangely... Upon creation,
        # the robot has an observation_space of Tuple(Box(6), Box(6)).  After
        # resetting, the robot has an observation_space of Tuple(Box(12)).

        # Since we reset before collecting data for training, we need to make
        # sure we're prepared for the later.
        env = rosgym.make_randomgoal_robot_env(
            "ur5_config_random_goal",
            envConf["robot_conf_file"],
            envConf["planning_group"],
            envConf["num_joints"],
            x_min=limits[0], y_min=limits[0],
            x_max=limits[1], y_max=limits[1]
        )
        env.reset()
        return env

    def _make_agent(self, agentConf):

        requiredAttributes = [
            "discount_factor",
            "tol",
            "actions_per_joint",
            "basis",
            "max_iter"
        ]

        for reqAttr in requiredAttributes:
            assert reqAttr in agentConf

        # Create the discretized actions and wrap our ur5 environemnt
        actions = ur5_discretizer(self._ur5env.action_space, agentConf["actions_per_joint"])
        self._env = DiscreteEnvWrapper(self._ur5env, actions)

        print("Original Action Space: {}".format(self._ur5env.action_space))
        print("Action Space: {}".format(self._env.action_space))
        print("Observation Space: {}".format(self._env.observation_space))

        # Create the agent's basis and lspi optimizer
        self._basis = makeBasis(agentConf["basis"], actions, self._env.observation_space)
        print("Basis Evaluated at random action and state: ")
        print(self._basis(self._env.observation_space.sample(), self._env.action_space.sample()))
        self._lstdq = LSTDQ(self._basis, discount=agentConf["discount_factor"])
        self._policy = DiscreteActionBasisPolicy(self._env.action_space, self._basis, np.zeros(self._basis.rank))
        self._lspi = LSPI(self._lstdq, self._policy, max_iter=agentConf["max_iter"], epsilon=agentConf["tol"], verbose=True)

    def _collect_data(self, experimentConf):

        requiredAttributes = [
            "num_rollouts",
            "num_timesteps"
        ]

        for reqAttr in requiredAttributes:
            assert reqAttr in experimentConf

        rp = RandomPolicy(self._env)
        data = SampleGenerator(self._env, rp).sample(experimentConf["num_rollouts"], experimentConf["num_timesteps"])

        dataFile = os.path.join(self._directory, "data.pkl")
        print("Writing collected experience data to {}".format(dataFile))
        with open(dataFile, "w") as f:
            pickle.dump(data, f)

        return data

    def _fit(self):
        # Fit the model, worry about running the tests later.
        self._lspi.fit(self._data)

    def _run_single_exp(self, policy, max_iter, **kwargs):
        state = self._env.reset()
        start = toArray(state)[:3]
        done = False
        goal = toArray(self._env.observation_space.sample())[-3:]
        numSteps = 0
        totalReward = 0
        success = False

        while not done and numSteps < max_iter:
            action = policy(state)
            state, reward, done, _ = self._env.step(action)
            totalReward += reward
            numSteps += 1

        success = np.linalg.norm(goal - toArray(state)[:3]) < 0.1

        resultDict = {
            "start": start,
            "goal": goal,
            "reward": totalReward,
            "numSteps": numSteps,
            "success": success,
        }
        resultDict.update(**kwargs)
        return resultDict



    def _test(self, experimentConf):

        requiredAttributes = [
            "num_tests_per_iteration",
            "max_test_iter"
        ]

        for reqAttr in requiredAttributes:
            assert reqAttr in experimentConf

        history = self._lspi.policy.history_
        historyFile = os.path.join(self._directory, "policyHistory.pkl")
        print("Writing policy history to {}".format(historyFile))
        with open(historyFile, "w") as f:
            pickle.dump(history, f)

        self.results_ = []
        for trainingItr, weights in enumerate(history[1:]):
            policy = DiscreteActionBasisPolicy(self._env.action_space, self._basis, weights)
            print("Testing training iteration {}".format(trainingItr))
            for testingItr in range(experimentConf["num_tests_per_iteration"]):
                self.results_.append(self._run_single_exp(policy, experimentConf["max_test_iter"], iteration=trainingItr))

        self.results_df_ = pd.DataFrame(self.results_)
        resultsFile = os.path.join(self._directory, "results.csv")
        print("Writing experiment results to {}".format(resultsFile))
        self.results_df_.to_csv(resultsFile)

if __name__ == "__main__":
    args = p.parse_args()

    exp = Experiment(args.directory, args.config)
    exp.initialize()
    exp.collect_data()
    exp.train()
    exp.evaluate()
