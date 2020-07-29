#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.25'


import numpy as np


def bellmanOptimalityUpdate(env, V, state, discount):
    action_values = []
    for action in range(env.num_actions):
        transition = env.transitions(state, action)
        acc = 0
        for s_, (r, p) in enumerate(transition):
            acc += p * (r + discount * V[s_])
        action_values.append(acc)
    return max(action_values)


def valueIteration(env, discount, theta):
    V = np.zeros(env.num_states)
    while True:
        delta = 0
        for s in range(env.num_states):
            if env.isTerminal(s):
                continue
            old_v = V[s]
            V[s] = bellmanOptimalityUpdate(env, V, s, discount)
            delta = max(delta, abs(old_v - V[s]))
        if delta < theta:
            break
    policy = np.ones((env.num_states, env.num_actions)) / env.num_actions
    for s in range(env.num_states):
        greedifyPolicy(env, V, policy, s, discount)
    return V, policy


def valueIteration2(env, discount, theta):
    V = np.zeros(env.num_states)
    policy = np.ones((env.num_states, env.num_actions)) / env.num_actions
    while True:
        delta = 0
        for s in range(env.num_states):
            if env.isTerminal(s):
                continue
            old_v = V[s]
            greedifyPolicy(env, V, policy, s, discount)
            V[s] = bellmanUpdate(env, V, policy, s, discount)
            delta = max(delta, abs(old_v - V[s]))
        if delta < theta:
            break
    return V, policy


def improvePolicy(env, V, policy, discount):
    stable = True
    for s in range(env.num_states):
        old = policy[s].copy()
        greedifyPolicy(env, V, policy, s, discount)
        if not np.array_equal(old, policy[s]):
            stable = False
    return policy, stable


def policyIteration(env, policy, discount, theta):
    V = np.zeros(env.num_states)
    while True:
        V = evaluatePolicy(env, V, policy, discount, theta)
        policy, stable = improvePolicy(env, V, policy, discount)
        if stable:
            break
    return V, policy


def greedifyPolicy(env, V, policy, state, discount):
    action_values = []
    for action in range(len(policy[state])):
        transition = env.transitions(state, action)
        acc = 0.0
        for s_, (r, p) in enumerate(transition):
            acc += p * (r + discount * V[s_])
        action_values.append(acc)
    argmax_a = np.where(np.array(action_values) == max(action_values))[0]
    argmax_a_prob = 1.0 / len(argmax_a)
    policy[state][:] = 0
    policy[state][argmax_a] = argmax_a_prob


def bellmanUpdate(env, V, policy, state, discount):
    acc = 0
    for action, prob in enumerate(policy[state]):
        transition = env.transitions(state, action)
        for s_, (r, p) in enumerate(transition):
            acc += prob * p * (r + discount * V[s_])
    return acc


def evaluatePolicy(env, V, policy, discount, theta):
    while True:
        delta = 0
        for s in range(env.num_states):
            if env.isTerminal(s):
                continue
            new_v = bellmanUpdate(env, V, policy, s, discount)
            delta = max(delta, abs(new_v - V[s]))
            V[s] = new_v
        if delta < theta:
            break
    return V
