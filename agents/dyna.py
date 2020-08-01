#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.29'


import numpy as np

from . import greedy
from rlglue.agent import BaseAgent


TERMINAL_STATE = -1


class DynaQAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.planning_rand_generator = np.random.RandomState(agent_info.get('planning_seed'))
        self.num_actions = agent_info.get('num_actions')
        self.num_states = agent_info.get('num_states')
        self.discount = agent_info.get("discount")
        self.epsilon = agent_info.get("epsilon")
        self.step_size = agent_info.get("step_size")
        self.planning_steps = agent_info.get("planning_steps")
        self.q_values = np.zeros((self.num_states, self.num_actions))

        self.last_state = None
        self.last_action = None
        self.model = {}

    def _updateModel(self, lastState, lastAction, state, reward):
        self.model.setdefault(lastState, {})
        self.model[lastState][lastAction] = (state, reward)

    def _planning(self):
        for _ in range(self.planning_steps):
            state = self.planning_rand_generator.choice(list(self.model.keys()))
            action = self.planning_rand_generator.choice(list(self.model[state].keys()))
            s_, reward = self.model[state][action]
            if s_ == TERMINAL_STATE:
                target = reward
            else:
                target = reward + self.discount * np.max(self.q_values[s_])
            self.q_values[state, action] += self.step_size * (target - self.q_values[state, action])

    def policy(self, state):
        return greedy.epsilonGreedySelect(self.q_values[state], self.rand_generator, self.epsilon)

    def agent_start(self, observation):
        self.last_state = observation
        self.last_action = self.policy(observation)
        return self.last_action

    def agent_step(self, reward, observation):
        target = reward + self.discount * np.max(self.q_values[observation])
        self.q_values[self.last_state, self.last_action] += \
            self.step_size * (target - self.q_values[self.last_state, self.last_action])
        self._updateModel(self.last_state, self.last_action, observation, reward)
        self._planning()
        self.last_action = self.policy(observation)
        self.last_state = observation
        return self.last_action

    def agent_end(self, reward):
        target = reward
        self.q_values[self.last_state, self.last_action] += \
            self.step_size * (target - self.q_values[self.last_state, self.last_action])
        self._updateModel(self.last_state, self.last_action, TERMINAL_STATE, reward)
        self._planning()

    def agent_message(self, message):
        if message == "q_values":
            return self.q_values
        elif message == "policy":
            policy = np.zeros((self.num_states, self.num_actions))
            for s in range(self.num_states):
                policy[s] = greedy.createEpsilonGreedyPolicy(self.q_values[s], self.epsilon, self.num_actions)
            return policy
        elif message == "values":
            return np.sum(self.q_values, axis=1)
        raise NotImplementedError


class DynaQPlusAgent(DynaQAgent):
    def agent_init(self, agent_info={}):
        super(DynaQPlusAgent, self).agent_init(agent_info)
        self.kappa = agent_info.get("kappa")
        self.tau = np.zeros((self.num_states, self.num_actions))
        self._actions = list(range(self.num_actions))

    def _updateModel(self, lastState, lastAction, state, reward):
        if lastState not in self.model:
            super(DynaQPlusAgent, self)._updateModel(lastState, lastAction, state, reward)
            for action in self._actions:
                if action != lastAction:
                    super(DynaQPlusAgent, self)._updateModel(lastState, action, lastState, 0)
        else:
            super(DynaQPlusAgent, self)._updateModel(lastState, lastAction, state, reward)

    def _planning(self):
        for _ in range(self.planning_steps):
            state = self.planning_rand_generator.choice(list(self.model.keys()))
            action = self.planning_rand_generator.choice(list(self.model[state].keys()))
            s_, reward = self.model[state][action]
            reward += self.kappa * np.sqrt(self.tau[state][action])
            if s_ == TERMINAL_STATE:
                target = reward
            else:
                target = reward + self.discount * np.max(self.q_values[s_])
            self.q_values[state, action] += self.step_size * (target - self.q_values[state, action])

    def _updateTau(self, state, action):
        self.tau += 1
        self.tau[state][action] = 0

    def agent_step(self, reward, observation):
        self._updateTau(self.last_state, self.last_action)
        return super(DynaQPlusAgent, self).agent_step(reward, observation)

    def agent_end(self, reward):
        self._updateTau(self.last_state, self.last_action)
        super(DynaQPlusAgent, self).agent_end(reward)
