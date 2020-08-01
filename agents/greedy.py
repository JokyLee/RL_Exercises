#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.25'


import numpy as np

from rlglue import BaseAgent


def createEpsilonGreedyPolicy(stateQValues, epsilon, numActions):
    epsilon_prob = epsilon / numActions
    state_policy = np.ones(numActions) * epsilon_prob
    argmax_a = np.where(stateQValues == np.max(stateQValues))[0]
    state_policy[argmax_a] = (1 - epsilon_prob * (numActions - len(argmax_a))) / len(argmax_a)
    return state_policy


def greedySelect(stateQValues, randomGenerator):
    max_q = np.max(stateQValues)
    return randomGenerator.choice(np.where(stateQValues == max_q)[0])


def epsilonGreedySelect(stateQValues, randomGenerator, epsilon):
    if randomGenerator.random() >= epsilon:
        return greedySelect(stateQValues, randomGenerator)
    return randomGenerator.randint(len(stateQValues))


class GreedyAgent(BaseAgent):
    def agent_init(self, agentConfig={}):
        self.num_actions = agentConfig['num_actions']
        self.q_values = np.zeros(self.num_actions)
        self.action_count = np.zeros(self.num_actions)
        self.rand_generator = np.random.RandomState(agentConfig.get("seed"))
        self.sum_award = 0
        self.last_state = None
        self.last_action = None

    def policy(self, qValues):
        return greedySelect(qValues, self.rand_generator)

    def agent_start(self, state):
        self.sum_award = 0
        self.last_state = state
        self.last_action = self.policy(self.q_values)
        return self.last_action

    def agent_step(self, reward, state):
        self.sum_award += reward
        self.action_count[self.last_action] += 1
        self.q_values[self.last_action] += (reward - self.q_values[self.last_action])\
                                           / self.action_count[self.last_action]
        self.last_state = state
        self.last_action = self.policy(self.q_values)
        return self.last_action

    def agent_end(self, reward):
        self.agent_step(reward, None)


class GreedyEpsilonAgent(GreedyAgent):
    def agent_init(self, agentConfig={}):
        super(GreedyEpsilonAgent, self).agent_init(agentConfig)
        self.epsilon = agentConfig['epsilon']

    def policy(self, qValues):
        return epsilonGreedySelect(qValues, self.rand_generator, self.epsilon)


class GreedyAgentConstantStepSize(GreedyAgent):
    def agent_init(self, agentConfig={}):
        super(GreedyAgentConstantStepSize, self).agent_init(agentConfig)
        self.step_size = agentConfig['step_size']

    def agent_step(self, reward, state):
        self.sum_award += reward
        self.q_values[self.last_action] += (reward - self.q_values[self.last_action]) * self.step_size
        self.last_state = state
        self.last_action = self.policy(self.q_values)
        return self.last_action


class GreedyEpsilonAgentConstantStepSize(GreedyEpsilonAgent, GreedyAgentConstantStepSize):
    def agent_init(self, agentConfig={}):
        GreedyEpsilonAgent.agent_init(self, agentConfig)
        GreedyAgentConstantStepSize.agent_init(self, agentConfig)

    def policy(self, qValues):
        return GreedyEpsilonAgent.policy(self, qValues)

    def agent_step(self, reward, state):
        return GreedyAgentConstantStepSize.agent_step(self, reward, state)
