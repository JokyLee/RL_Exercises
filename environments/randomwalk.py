#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.08.01'


import enum

import numpy as np
from gym import spaces

from rlglue import BaseEnvironment


class Actions(enum.Enum):
    LEFT = 0
    RIGHT = 1


class RandomWalkEnvironment(BaseEnvironment):
    action_space = spaces.Discrete(2)
    def env_init(self, env_info={}):
        self.rand_generator = np.random.RandomState(env_info.get("seed"))
        self.num_states = env_info["num_states"]
        self.start_state = self.num_states // 2
        self.left_terminal_state = 0
        self.right_terminal_state = self.num_states - 1
        self.num_actions = self.action_space.n

    def env_start(self):
        self.now = self.start_state
        return self.now

    def env_step(self, action):
        action = Actions(action)
        if action == Actions.LEFT:
            self.now = max(self.left_terminal_state, self.now - self.rand_generator.randint(1, 101))
        elif action == Actions.RIGHT:
            self.now = min(self.right_terminal_state, self.now + self.rand_generator.randint(1, 101))
        else:
            raise ValueError("Invalid action")

        if self.now == self.left_terminal_state:
            reward = -1
            term = True
        elif self.now == self.right_terminal_state:
            reward = 1
            term = True
        else:
            reward = 0
            term = False
        return (reward, self.now, term)
