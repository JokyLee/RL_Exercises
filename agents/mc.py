#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.27'

from collections import OrderedDict

import numpy as np

from rlglue import BaseAgent


class MonteCarloPredictionAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.policy = agent_info.get("policy")
        self.discount = agent_info.get("discount")
        self.step_size = agent_info.get("step_size")
        self.values = np.zeros(self.policy.shape[0])
        self.return_count = [0] * self.policy.shape[0]
        self.episode_states = None
        self.episode_rewards = None

    def agent_start(self, observation):
        self.episode_states = [observation]
        self.episode_rewards = [None]
        return self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[observation])

    def agent_step(self, reward, observation):
        self.episode_states.append(observation)
        self.episode_rewards.append(reward)
        return self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[observation])

    def agent_end(self, reward):
        self.episode_states.append(None)
        self.episode_rewards.append(reward)
        G = 0
        T = len(self.episode_rewards) - 1
        for t in range(T - 1, -1, -1):
            St, Rt_plus_1 = self.episode_states[t], self.episode_rewards[t + 1]
            G = self.discount * G + Rt_plus_1
            if St not in self.episode_states[:t]:
                self.return_count[St] += 1
                self.values[St] = self.values[St] + (G - self.values[St]) / self.return_count[St]

    def agent_message(self, message):
        if message == "values":
            return self.values
        raise NotImplementedError


class MonteCarloControlAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.num_states = agent_info.get("num_states")
        self.num_actions = agent_info.get("num_actions")
        self.discount = agent_info.get("discount")
        self.step_size = agent_info.get("step_size")

        self.policy = np.ones((self.num_states, self.num_actions)) / self.num_actions
        self.q_values = np.ones((self.num_states, self.num_actions)) * -100
        self.returns_count = {(s, a): 0 for s in range(self.num_states) for a in range(self.num_actions)}

        self.episode_state_action_pair = None
        self.episode_reward = None
        self.last_action = None
        self.last_state = None

    def agent_start(self, observation):
        self.episode_state_action_pair = []
        self.episode_reward = []
        action = self.rand_generator.choice(range(self.policy.shape[1]), p=np.ones(self.num_actions) / self.num_actions)
        self.last_action = action
        self.last_state = observation
        return action

    def agent_step(self, reward, observation):
        self.episode_state_action_pair.append((self.last_state, self.last_action))
        self.episode_reward.append(reward)
        self.last_action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[observation])
        self.last_state = observation
        return self.last_action

    def _updatePolicy(self, St, At):
        raise NotImplementedError("please implement _updatePolicy")

    def agent_end(self, reward):
        self.episode_state_action_pair.append((self.last_state, self.last_action))
        self.episode_reward.append(reward)
        G = 0
        for t in range(len(self.episode_state_action_pair) - 1, -1, -1):
            state_action_pair = self.episode_state_action_pair[t]
            Rt_plus_1 = self.episode_reward[t]
            G = self.discount * G + Rt_plus_1
            if state_action_pair not in self.episode_state_action_pair[:t]:
                self.returns_count[state_action_pair] += 1
                self.q_values[state_action_pair] += (G - self.q_values[state_action_pair]) / self.returns_count[state_action_pair]
                St, At = state_action_pair
                self._updatePolicy(St, At)

    def agent_message(self, message):
        if message == "q_values":
            return self.q_values
        elif message == "policy":
            return self.policy
        elif message == "values":
            return np.sum(self.q_values * self.policy, axis=1)
        raise NotImplementedError


class MonteCarloControlAgent_ExploringStart(MonteCarloControlAgent):
    def _updatePolicy(self, St, At):
        self.policy[St] = 0
        argmax_a = np.where(self.q_values[St] == np.max(self.q_values[St]))[0]
        for i in argmax_a:
            self.policy[St][i] = 1 / len(argmax_a)


class MonteCarloControlAgent_EpsilonSoft(MonteCarloControlAgent):
    def agent_init(self, agent_info={}):
        super(MonteCarloControlAgent_EpsilonSoft, self).agent_init(agent_info)
        self.epsilon = agent_info.get("epsilon")

    def _updatePolicy(self, St, At):
        epsilon_prob = self.epsilon / self.num_actions
        self.policy[St] = epsilon_prob
        argmax_a = np.where(self.q_values[St] == np.max(self.q_values[St]))[0]
        temp = (1 - epsilon_prob * (self.num_actions - len(argmax_a))) / len(argmax_a)
        for i in argmax_a:
            self.policy[St][i] = temp


class MonteCarloControlAgent_OffPolicy(MonteCarloControlAgent):
    def agent_init(self, agent_info={}):
        super(MonteCarloControlAgent_OffPolicy, self).agent_init(agent_info)
        self.learnt_policy = np.ones((self.num_states, self.num_actions)) / self.num_actions

    def _updatePolicy(self, St, At):
        self.learnt_policy[St] = 0
        argmax_a = np.where(self.q_values[St] == np.max(self.q_values[St]))[0]
        for i in argmax_a:
            self.learnt_policy[St][i] = 1 / len(argmax_a)

    def agent_end(self, reward):
        self.episode_state_action_pair.append((self.last_state, self.last_action))
        self.episode_reward.append(reward)
        G = 0
        W = 1
        for t in range(len(self.episode_state_action_pair) - 1, -1, -1):
            state_action_pair = self.episode_state_action_pair[t]
            Rt_plus_1 = self.episode_reward[t]
            G = self.discount * W * G + Rt_plus_1
            self.returns_count[state_action_pair] += 1
            self.q_values[state_action_pair] += (G - self.q_values[state_action_pair]) / self.returns_count[state_action_pair]
            St, At = state_action_pair
            self._updatePolicy(St, At)
            action_learnt_policy = self.rand_generator.choice(
                range(self.learnt_policy.shape[1]), p=self.learnt_policy[St]
            )
            if At != action_learnt_policy:
                return
            W *= self.learnt_policy[St][At] / self.policy[St][At]

    def agent_message(self, message):
        if message == "q_values":
            return self.q_values
        elif message == "policy":
            return self.learnt_policy
        elif message == "values":
            return np.sum(self.q_values * self.learnt_policy, axis=1)
        raise NotImplementedError
