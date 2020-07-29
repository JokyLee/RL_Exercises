#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.25'

import numpy as np

from rlglue import BaseEnvironment


NO_OBSERVATION = 0


class BanditsEnvironment(BaseEnvironment):
    def __init__(self):
        self.action_num = self.arm_num = 10
        reward = None
        observation = None
        termination = None
        self.reward_obs_term = (reward, observation, termination)
        self.count = 0
        self.arms = []

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """
        self.arms = np.random.randn(self.arm_num)  # [np.random.normal(0.0, 1.0) for _ in range(10)]
        self.reward_obs_term = (0.0, NO_OBSERVATION, False)

    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        return self.reward_obs_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        reward = self.arms[action] + np.random.randn()
        self.reward_obs_term = (reward, NO_OBSERVATION, False)
        return self.reward_obs_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        pass

    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message (string): the message passed to the environment

        Returns:
            string: the response (or answer) to the message
        """
        pass
