#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.08.04'


import numpy as np

from . import tilescoding
from rlglue import BaseAgent


def computeSoftmaxProb(actorWeights, tiles):
    state_action_preferences = [
        actorWeights[a][tiles].sum() for a in range(actorWeights.shape[0])
    ]
    max_preference = np.max(state_action_preferences)
    numerator = np.exp(state_action_preferences - max_preference)
    denominator = np.sum(numerator)
    softmax_prob = numerator / denominator
    return softmax_prob


class Critic:
    def __init__(self, tileSize, stepSize):
        self.tile_size = tileSize
        self.weights = np.zeros(tileSize)
        self.step_size = stepSize

    def getValue(self, tiles):
        return sum(self.weights[tiles])

    def updateWeights(self, tiles, delta):
        self.weights[tiles] += self.step_size * delta


class ActorCriticSoftmaxAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.env_observation_space = agent_info.get("observation_space")
        self.env_action_space = agent_info.get("action_space")

        self.num_actions = self.env_action_space.n
        self.actions = list(range(self.num_actions))
        self.num_tilings = agent_info.get("num_tilings")
        self.num_tiles = agent_info.get("num_tiles")
        self.iht_size = agent_info.get("iht_size")

        self.actor_step_size = agent_info.get("actor_step_size") / self.num_tilings
        self.critic_step_size = agent_info.get("critic_step_size") / self.num_tilings
        self.avg_reward_step_size = agent_info.get("avg_reward_step_size")

        self.avg_reward = 0
        self.actor_w = np.zeros((self.num_actions, self.iht_size))
        self.critic = Critic(self.iht_size, self.critic_step_size)

        self.tile_coder = tilescoding.TileWrapCoder(
            self.iht_size, self.num_tilings, self.num_tiles, self.env_observation_space
        )

        self.softmax_prob = None
        self.last_tiles = None
        self.last_action = None

    def policy(self, tiles):
        softmax_prob = computeSoftmaxProb(self.actor_w, tiles)
        chosen_action = self.rand_generator.choice(self.actions, p=softmax_prob)
        return chosen_action, softmax_prob

    def agent_start(self, observation):
        active_tiles = self.tile_coder.getTiles(observation)
        current_action, softmax_prob = self.policy(active_tiles)
        self.last_action = current_action
        self.last_tiles = active_tiles
        self.last_softmax_prob = softmax_prob
        return self.last_action

    def agent_step(self, reward, observation):
        active_tiles = self.tile_coder.getTiles(observation)
        delta = reward - self.avg_reward + self.critic.getValue(active_tiles) - self.critic.getValue(self.last_tiles)
        self.avg_reward += self.avg_reward_step_size * delta
        self.critic.updateWeights(self.last_tiles, delta)

        for a in self.actions:
            if a == self.last_action:
                self.actor_w[a][self.last_tiles] += self.actor_step_size * delta * (1 - self.last_softmax_prob[a])
            else:
                self.actor_w[a][self.last_tiles] += self.actor_step_size * delta * (0 - self.last_softmax_prob[a])

        current_action, softmax_prob = self.policy(active_tiles)
        self.last_action = current_action
        self.last_tiles = active_tiles
        self.last_softmax_prob = softmax_prob
        return self.last_action

    def agent_message(self, message):
        if message == "avg_reward":
            return self.avg_reward
        raise NotImplementedError
