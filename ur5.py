from __future__ import print_function
import rosgym
import gym
import numpy as np
import os
import rospy
import imp
import random
import requests
import time
import yaml

import sys
sys.path.append(os.path.dirname(os.getcwd()) + '/python')

import basic_agents
from gv8_dummy_agent import RandomAgent

from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker


def make_environment(robot_conf_file, planning_group, num_joints, z_limit,
                     x_limits, y_limits, **kwargs):
    """
    Creates an OpenAI Gym environment corresponding to the supplied robot
    configuration file, and configuration options.

    :return: rosgym environment
    """

    env = rosgym.make_randomgoal_robot_env(
        "ur5_config_random_goal", robot_conf_file, planning_group, num_joints,
        z_limit, x_limits[0], x_limits[1], y_limits[0], y_limits[1])
    return env


def initialize_viz_marker():
    """
    Initializes a ROS Publisher that publishes the ... of the robot.

    :return: visualization_msgs.msgs.Marker
    """

    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    robot_marker = Marker()

    state = PointStamped()

    state.point.x = 1
    state.point.y = 1
    state.point.z = 1

    robot_marker = Marker()
    robot_marker.header.frame_id = "/base_link"
    robot_marker.header.stamp = rospy.get_rostime()
    robot_marker.ns = "basic_shapes"
    robot_marker.id = 1
    robot_marker.type = Marker.TEXT_VIEW_FACING
    robot_marker.action = Marker.ADD
    robot_marker.pose.position = state.point
    robot_marker.pose.orientation.x = 0
    robot_marker.pose.orientation.y = 0
    robot_marker.pose.orientation.z = 0
    robot_marker.pose.orientation.w = 1.0

    robot_marker.color.r = 1.0
    robot_marker.color.g = 1.0
    robot_marker.color.b = 1.0
    robot_marker.color.a = 1.0

    robot_marker.scale.z = 0.1
    robot_marker.text = "Not started"

    robot_marker.lifetime = rospy.Duration(1)

    def timer_callback(event):
        marker_pub.publish(robot_marker)

    timer = rospy.Timer(rospy.Duration(0.01), timer_callback)


class SimpleConfig(object):
    def __init__(self, config, defaults):
        self._config = config
        self._defaults = defaults

    def __getitem__(self, key):
        return self._config.get(key, self._defaults.get(key))


def main(config, defaults):
    """
    Main entrypoint for this script.  Simply runs our agent for a bunch of tests.
    """

    sc = SimpleConfig(config, defaults)

    # make our random agent
    print("Attempting to make the rosgym environment...")
    env = make_environment(
        sc["robot_conf_file"], sc["planning_group"], sc["num_joints"],
        sc["z_limit"], [sc["x_min"], sc["x_max"]], [sc["y_min"], sc["y_max"]])
    print("Made the rosgym environment...")

    # make our random agent
    agent = RandomAgent(env)

    print("Testing random agent: {}".format(agent))

    for test in xrange(sc["num_tests"]):

        print("Test #: {}".format(test))
        state = agent.reset()

        actions = []

        for episode in xrange(sc["num_episodes"]):

            print("Episode #: {}".format(episode))
            try:
                state, action, reward, next_state, done = agent.step(
                    test, episode, testing=True)
                actions.append(action)
            except Exception as err:
                print(err)

            # state, action, reward, next_state, done = agent.step(i, episode)
            if reward > 0:
                success = True

            if done:
                break

        print("Test actions: {}".format(actions))


if __name__ == "__main__":

    defaults = {
        "robot_conf_file":
        "/home/cannon/rl_wksp/src/rosgym/src/rosgym/ur5_config_random_goal.py",
        "planning_group":
        "manipulator",
        "num_joints":
        6,
        "z_limit":
        2.0,
        "x_min":
        -2.0,
        "x_max":
        2.0,
        "y_min":
        -2.0,
        "y_max":
        2.0,
        "num_tests":
        10,
        "num_episodes":
        100,
    }

    print("Dummy agent running with defaults: {}".format(defaults))

    main({}, defaults)
