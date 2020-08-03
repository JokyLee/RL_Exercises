#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.08.01'


import numpy as np

from rlglue import BaseAgent


class StateAggregation:
    def __init__(self, numStates, numGroups):
        self.num_states = numStates
        self.num_groups = numGroups
        if self.num_states % self.num_groups == 0:
            self.num_states_in_group = self.num_states // self.num_groups
        else:
            raise ValueError("numStates % numGroups != 0")
        self.all_state_features = [self._calFeature(i) for i in range(self.num_states)]

    def _calFeature(self, state):
        one_hot_vector = np.zeros(self.num_groups)
        one_hot_vector[state // self.num_states_in_group] = 1
        return one_hot_vector

    def getFeature(self, state):
        return self.all_state_features[state]


class GradientPredictAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.num_actions = agent_info.get('num_actions')
        self.num_states = agent_info.get('num_states')
        self.num_groups = agent_info.get('num_groups')
        self.discount = agent_info.get("discount")
        self.step_size = agent_info.get("step_size")

        self.state_aggregation = StateAggregation(self.num_states, self.num_groups)
        self.weights = np.zeros(self.num_groups)
        self.last_state = None
        self.last_action = None

    def policy(self, state):
        return self.rand_generator.randint(self.num_actions)

    def agent_start(self, observation):
        self.last_state = observation
        self.last_action = self.policy(observation)
        return self.last_action

    def agent_step(self, reward, observation):
        raise NotImplementedError

    def agent_end(self, reward):
        raise NotImplementedError

    def agent_message(self, message):
        if message == "values":
            return np.dot(self.state_aggregation.all_state_features, self.weights).ravel()
        raise NotImplementedError


class SemiGradientTDZeroAgent(GradientPredictAgent):
    def agent_step(self, reward, observation):
        cur_feature = self.state_aggregation.getFeature(observation)
        last_feature = self.state_aggregation.getFeature(self.last_state)
        target = reward + self.discount * np.dot(self.weights, cur_feature.T)
        self.weights += self.step_size * (target - np.dot(self.weights, last_feature.T)) * last_feature

        self.last_state = observation
        self.last_action = self.policy(observation)
        return self.last_action

    def agent_end(self, reward):
        last_feature = self.state_aggregation.getFeature(self.last_state)
        target = reward
        self.weights += self.step_size * (target - np.dot(self.weights, last_feature.T)) * last_feature


class GradientMonteCarloAgent(GradientPredictAgent):
    def agent_init(self, agent_info={}):
        super(GradientMonteCarloAgent, self).agent_init(agent_info)
        self.record = None

    def agent_start(self, observation):
        action = super(GradientMonteCarloAgent, self).agent_start(observation)
        self.record = []
        return action

    def agent_step(self, reward, observation):
        self.record.append((self.last_state, self.last_action, reward))
        self.last_state = observation
        self.last_action = self.policy(observation)
        return self.last_action

    def agent_end(self, reward):
        self.record.append((self.last_state, self.last_action, reward))
        G = 0
        for t in range(len(self.record) - 1, -1, -1):
            St, At, Rt_plus_1 = self.record[t]
            feature_t = self.state_aggregation.getFeature(St)
            G = self.discount * G + Rt_plus_1
            self.weights += self.step_size * (G - np.dot(self.weights, feature_t.T)) * feature_t

    def agent_message(self, message):
        if message == "values":
            return np.dot(self.state_aggregation.all_state_features, self.weights).ravel()
        raise NotImplementedError
