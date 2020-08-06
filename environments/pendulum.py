#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.08.03'


import os
import time

import math
import numpy as np
from . import gymenvs
import gym
from gym import spaces
from rlglue import BaseEnvironment

#
# class PendulumEnvironment_Discrete(gymenvs.GymEnvironment):
#     ENV_NAME = "Pendulum-v0"
#     ENV = gym.make(ENV_NAME)
#     action_space = spaces.Discrete(3)
#     observation_space = spaces.Box(low=np.array([-np.pi, -2 * np.pi]), high=np.array([np.pi, 2 * np.pi]), dtype=np.float)
#     max_episode_steps = float('inf')
#
#     def env_init(self, env_info={}):
#         super(PendulumEnvironment_Discrete, self).env_init(env_info)
#         self.action_map = [-1, 0, 1]
#
#     def env_start(self):
#         ret = self.env.reset()
#         self.env.env.m = float(1. / 3.)
#         self.env.env.l = float(3. / 2.)
#
#         self.env.env.state = np.array([np.pi, 0])
#         angle = np.pi
#         return np.array([angle, ret[1]])
#
#     def env_step(self, action):
#         ret = super(PendulumEnvironment_Discrete, self).env_step([self.action_map[action]])
#
#         # beta = last_beta + betadot * self.dt
#         #
#         # # normalize angle
#         # beta = ((beta + np.pi) % (2 * np.pi)) - np.pi
#         #
#         # # reset if out of bound
#         # if betadot < self.ang_velocity_range[0] or betadot > self.ang_velocity_range[1]:
#         #     beta = -np.pi
#         #     betadot = 0.
#
#         # if self.__class__.observation_space
#         # compute reward
#         angle = math.atan2(ret[1][1], ret[1][0])
#         reward = -np.abs(angle)
#         # return (ret[0], np.array([angle, ret[1][1]]), ret[2])
#         return (reward, np.array([angle, ret[1][1]]), ret[2])
#



class PendulumEnvironment(BaseEnvironment):
    action_space = spaces.Discrete(3)
    observation_space = spaces.Box(low=np.array([-np.pi, -2 * np.pi]), high=np.array([np.pi, 2 * np.pi]), dtype=np.float)
    max_episode_steps = float('inf')
    def __init__(self):
        self.rand_generator = None
        self.ang_velocity_range = None
        self.dt = None
        self.viewer = None
        self.gravity = None
        self.mass = None
        self.length = None

        self.valid_actions = None
        self.actions = None

    def env_init(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts.

        Set parameters needed to setup the pendulum swing-up environment.
        """
        # set random seed for each run
        self.rand_generator = np.random.RandomState(env_info.get("seed"))
        self.render_flag = env_info.get("render_flag", False)
        self.sleep_time = env_info.get("sleep_time", 0.05)

        self.ang_velocity_range = [-2 * np.pi, 2 * np.pi]
        self.dt = 0.05
        self.viewer = None
        self.gravity = 9.8
        self.mass = float(1. / 3.)
        self.length = float(3. / 2.)

        self.valid_actions = (0, 1, 2)
        self.actions = [-1, 0, 1]

        self.last_action = None

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

        ### set self.reward_obs_term tuple accordingly (3~5 lines)
        # Angle starts at -pi or pi, and Angular velocity at 0.
        # reward = ?
        # observation = ?
        # is_terminal = ?

        beta = -np.pi
        betadot = 0.

        reward = 0.0
        observation = np.array([beta, betadot])
        is_terminal = False

        self.reward_obs_term = (reward, observation, is_terminal)

        # return first state observation from the environment
        return self.reward_obs_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        ### set reward, observation, and is_terminal correctly (10~12 lines)
        # Update the state according to the transition dynamics
        # Remember to normalize the angle so that it is always between -pi and pi.
        # If the angular velocity exceeds the bound, reset the state to the resting position
        # Compute reward according to the new state, and is_terminal should always be False
        #
        # reward = ?
        # observation = ?
        # is_terminal = ?

        # Check if action is valid
        assert (action in self.valid_actions)

        if self.render_flag:
            self.render()
            time.sleep(self.sleep_time)

        last_state = self.reward_obs_term[1]
        last_beta, last_betadot = last_state
        self.last_action = action

        betadot = last_betadot + 0.75 * (
                    self.actions[action] + self.mass * self.length * self.gravity * np.sin(last_beta)) / (
                              self.mass * self.length ** 2) * self.dt

        beta = last_beta + betadot * self.dt

        # normalize angle
        beta = ((beta + np.pi) % (2 * np.pi)) - np.pi

        # reset if out of bound
        if betadot < self.ang_velocity_range[0] or betadot > self.ang_velocity_range[1]:
            beta = -np.pi
            betadot = 0.

        # compute reward
        reward = -(np.abs(((beta + np.pi) % (2 * np.pi)) - np.pi))
        observation = np.array([beta, betadot])
        is_terminal = False

        self.reward_obs_term = (reward, observation, is_terminal)

        return self.reward_obs_term

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = os.path.join(os.path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        last_state = self.reward_obs_term[1]
        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(last_state[0] + np.pi / 2)
        if self.last_action is not None:
            value = self.actions[self.last_action]
            self.imgtrans.scale = (-value / 2, np.abs(value) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
