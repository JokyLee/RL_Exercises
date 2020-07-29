#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.25'


import enum
import numpy as np

from gym import spaces
from rlglue import BaseEnvironment


class Actions(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GridWorldEnvironment(BaseEnvironment):
    action_space = spaces.Discrete(4)
    rows = 4
    cols = 4

    def env_init(self, env_info={}):
        self.rand_generator = np.random.RandomState(env_info.get("seed"))
        self.num_states = self.observation_space.n
        self.num_actions = self.action_space.n

    def env_start(self):
        self.now = self.coord2Idx([self.rows // 2, self.cols // 2])
        return self.now

    def isTerminal(self, state):
        coord = self.idx2Coord(state)
        if (coord[0] == 0 and coord[1] == 0) or (coord[0] == self.rows - 1 and coord[1] == self.cols - 1):
            return True
        return False

    def env_step(self, action):
        p = self.transitions(self.now, action)
        next_s = self.rand_generator.choice(range(self.num_states), p=p[:, 1])
        self.now = next_s
        reward = p[next_s, 0]
        return (reward, self.now, self.isTerminal(next_s))

    def transitions(self, state, action):
        action = Actions(action)
        coord = self.idx2Coord(state)
        if action == Actions.RIGHT:
            coord[1] = min(coord[1] + 1, self.cols - 1)
        elif action == Actions.LEFT:
            coord[1] = max(coord[1] - 1, 0)
        elif action == Actions.UP:
            coord[0] = max(coord[0] - 1, 0)
        elif action == Actions.DOWN:
            coord[0] = min(coord[0] + 1, self.rows - 1)
        else:
            raise NotImplementedError("Invalid action.")
        next_state = self.coord2Idx(coord)
        p = np.zeros((self.num_states, 2))
        p[:, 0] = -1
        p[next_state][1] = 1
        return p

    @property
    def observation_space(self):
        return spaces.Discrete(self.rows * self.cols)

    def coord2Idx(self, coord):
        return coord[0] * self.cols + coord[1]

    def idx2Coord(self, idx):
        return [idx // self.cols, idx % self.cols]


class GridWorldEnvironment_RandomStart(GridWorldEnvironment):
    def env_init(self, env_info={}):
        super(GridWorldEnvironment_RandomStart, self).env_init(env_info)
        self.valid_start_state = [
            i for i in range(self.num_states) if not self.isTerminal(i)
        ]

    def env_start(self):
        self.now = self.rand_generator.choice(self.valid_start_state, 1)[0]
        return self.now


class CliffWorldEnvironment(GridWorldEnvironment):
    rows = 4
    cols = 12
    def env_init(self, env_info={}):
        super(CliffWorldEnvironment, self).env_init(env_info)
        self.start = (self.rows - 1, 0)
        self.goal = (self.rows - 1, self.cols - 1)

    def env_start(self):
        self.now = self.coord2Idx(self.start)
        return self.now

    def isTerminal(self, state):
        coord = self.idx2Coord(state)
        if coord[0] == self.goal[0] and coord[1] == self.goal[1]:
            return True
        return False

    def _isInCliff(self, coord):
        if coord[0] == self.rows - 1 and 1 <= coord[1] <= self.goal[1] - 1:
            return True
        return False

    def transitions(self, state, action):
        action = Actions(action)
        coord = self.idx2Coord(state)
        if action == Actions.RIGHT:
            coord[1] = min(coord[1] + 1, self.cols - 1)
        elif action == Actions.LEFT:
            coord[1] = max(coord[1] - 1, 0)
        elif action == Actions.UP:
            coord[0] = max(coord[0] - 1, 0)
        elif action == Actions.DOWN:
            coord[0] = min(coord[0] + 1, self.rows - 1)
        else:
            raise NotImplementedError("Invalid action.")

        if self._isInCliff(coord):
            reward = -100
            coord = list(self.start)
        else:
            reward = -1
        next_state = self.coord2Idx(coord)
        p = np.zeros((self.num_states, 2))
        p[:, 0] = -1
        p[next_state][0] = reward
        p[next_state][1] = 1
        return p


class CliffWorldEnvironment_RewardGoal(CliffWorldEnvironment):
    def transitions(self, state, action):
        p = super(CliffWorldEnvironment_RewardGoal, self).transitions(state, action)
        next_state = np.where(p[:, 1] == 1)[0][0]
        if self.isTerminal(next_state):
            p[next_state][0] = 1000
        return p


class CliffWorldEnvironment_RewardGoal_RandomStart(CliffWorldEnvironment_RewardGoal):
    def env_init(self, env_info={}):
        super(CliffWorldEnvironment_RewardGoal_RandomStart, self).env_init(env_info)
        self.valid_start_state = [
            i for i in range(self.num_states) if not self.isTerminal(i) and not self._isInCliff(self.idx2Coord(i))
        ]

    def env_start(self):
        self.now = self.rand_generator.choice(self.valid_start_state, 1)[0]
        return self.now

