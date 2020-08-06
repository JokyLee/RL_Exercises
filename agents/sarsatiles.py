#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.08.03'


import numpy as np

from . import greedy
from rlglue import BaseAgent
from .tilescoding import TileCoder


class SarseTilesAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.env_observation_space = agent_info.get("observation_space")
        self.env_action_space = agent_info.get("action_space")

        self.num_actions = self.env_action_space.n
        self.num_tilings = agent_info.get("num_tilings")
        self.num_tiles = agent_info.get("num_tiles")
        self.iht_size = agent_info.get("iht_size")
        self.discount = agent_info.get("discount")
        self.epsilon = agent_info.get("epsilon")
        self.step_size = agent_info.get("step_size") / self.num_tilings
        self.initial_weights = agent_info.get("initial_weights", 0.0)
        self.weights = np.ones((self.num_actions, self.iht_size)) * self.initial_weights

        self.tile_coder = TileCoder(self.iht_size, self.num_tilings, self.num_tiles, self.env_observation_space)

    def selectAction(self, tiles):
        action_values = np.sum(self.weights[:, tiles], axis=1)
        action = greedy.epsilonGreedySelect(action_values, self.rand_generator, self.epsilon)
        return action, action_values[action]

    def agent_start(self, observation):
        self.previous_tiles = self.tile_coder.getTiles(observation).copy()
        self.last_action, _ = self.selectAction(self.previous_tiles)
        return self.last_action

    def agent_step(self, reward, observation):
        active_tiles = self.tile_coder.getTiles(observation)
        current_action, action_value = self.selectAction(active_tiles)

        prev_value = sum(self.weights[self.last_action][self.previous_tiles])
        target = reward + self.discount * action_value
        # 1
        # gradient = np.zeros_like(self.weights)
        # gradient[self.last_action][self.previous_tiles] = 1
        # self.weights += self.step_size * (target - prev_value) * gradient

        # 2
        gradient = np.ones(self.previous_tiles.shape)
        self.weights[self.last_action][self.previous_tiles] += self.step_size * (target - prev_value) * gradient

        self.last_action = current_action
        self.previous_tiles = active_tiles.copy()
        return self.last_action

    def agent_end(self, reward):
        prev_value = sum(self.weights[self.last_action][self.previous_tiles])
        target = reward

        # 1
        # gradient = np.zeros_like(self.weights)
        # gradient[self.last_action][self.previous_tiles] = 1
        # self.weights += self.step_size * (target - prev_value) * gradient

        # 2
        gradient = np.ones(self.previous_tiles.shape)
        self.weights[self.last_action][self.previous_tiles] += self.step_size * (target - prev_value) * gradient
