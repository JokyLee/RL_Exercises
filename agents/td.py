#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.26'


import numpy as np

from . import greedy
from rlglue import BaseAgent


class TDAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.policy = agent_info.get("policy")
        self.discount = agent_info.get("discount")
        self.step_size = agent_info.get("step_size")
        self.values = np.zeros(self.policy.shape[0])
        self.last_state = None
        self.last_action = None

    def agent_start(self, observation):
        self.last_state = observation
        self.last_action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[observation])
        return self.last_action

    def agent_step(self, reward, observation):
        target = reward + self.discount * self.values[observation]
        self.values[self.last_state] += self.step_size * (target - self.values[self.last_state])
        self.last_action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[observation])
        self.last_state = observation
        return self.last_action

    def agent_end(self, reward):
        target = reward
        self.values[self.last_state] += self.step_size * (target - self.values[self.last_state])

    def agent_cleanup(self):
        self.last_state = None
        self.last_action = None

    def agent_message(self, message):
        if message == "values":
            return self.values
        raise NotImplementedError


class TDControlAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.num_actions = agent_info.get('num_actions')
        self.num_states = agent_info.get('num_states')
        self.discount = agent_info.get("discount")
        self.epsilon = agent_info.get("epsilon")
        self.step_size = agent_info.get("step_size")
        self.q_values = np.zeros((self.num_states, self.num_actions))

        self.last_state = None
        self.last_action = None

    def policy(self, observation):
        return greedy.epsilonGreedySelect(self.q_values[observation], self.rand_generator, self.epsilon)

    def agent_start(self, observation):
        self.last_state = observation
        self.last_action = self.policy(observation)
        return self.last_action

    def agent_step(self, reward, observation):
        raise NotImplementedError("plaease implement agent_step")

    def agent_cleanup(self):
        self.last_state = None
        self.last_action = None

    def agent_end(self, reward):
        target = reward
        self.q_values[self.last_state][self.last_action] += \
                    self.step_size * (target - self.q_values[self.last_state][self.last_action])

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


class QLearningAgent(TDControlAgent):
    """
    Off-policy TD method, value iteration
    """
    def agent_step(self, reward, observation):
        target = reward + self.discount * max(self.q_values[observation])
        self.q_values[self.last_state][self.last_action] += \
            self.step_size * (target - self.q_values[self.last_state][self.last_action])

        self.last_action = self.policy(observation)
        self.last_state = observation
        return self.last_action


class SarsaAgent(TDControlAgent):
    """
    On-policy TD method, policy iteration
    """
    def agent_step(self, reward, observation):
        a_dot = self.policy(observation)
        target = reward + self.discount * self.q_values[observation][a_dot]
        self.q_values[self.last_state][self.last_action] += \
            self.step_size * (target - self.q_values[self.last_state][self.last_action])

        self.last_action = a_dot
        self.last_state = observation
        return self.last_action


class ExpectedSarsaAgent(TDControlAgent):
    """
    Off-policy TD method
    """
    def agent_step(self, reward, observation):
        a_dot = self.policy(observation)
        policy = greedy.createEpsilonGreedyPolicy(self.q_values[observation], self.epsilon, self.num_actions)
        target = reward + self.discount * np.sum(policy * self.q_values[observation])
        self.q_values[self.last_state][self.last_action] += \
            self.step_size * (target - self.q_values[self.last_state][self.last_action])
        self.last_action = a_dot
        self.last_state = observation
        return self.last_action

