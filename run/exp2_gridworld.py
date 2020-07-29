#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.26'


import numpy as np

from tools import gridsvisualizer
from environments import gridworld


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
    idx = 0
    while True:
        if idx in (0, 1, 2, 3, 10):
            print("iteration {}:".format(idx))
            print(V.reshape(env.rows, env.cols))
        delta = 0
        old_v = V.copy()
        for s in range(env.num_states):
            if env.isTerminal(s):
                continue
            V[s] = bellmanUpdate(env, old_v, policy, s, discount)
            delta = max(delta, abs(old_v[s] - V[s]))
        if delta < theta:
            break
        idx += 1
    return V


def evaluatePolicyInPlace(env, V, policy, discount, theta):
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


def playWithPolicy(env, policy, startCoord=None):
    print("========= playing in the gridworld... ==========")
    state = env.env_start()
    if startCoord is not None:
        state = env.now = env.coord2Idx(startCoord)
    print("starting at {}".format(env.idx2Coord(state)))
    while True:
        action = np.random.choice(range(env.num_actions), p=policy[state])
        print("S={}, A={}".format(env.idx2Coord(state), gridworld.Actions(action)))
        reward, state, term = env.env_step(action)
        print("S'={}, R={}, T={}".format(env.idx2Coord(state), reward, term))
        if term:
            print("reached the end")
            break
    print("========== playing in the gridworld... done! ==========")


def main():
    env_config = {
        "seed": 0
    }
    grids = gridworld.GridWorldEnvironment()
    grids.env_init(env_config)
    policy = np.ones((grids.num_states, grids.num_actions)) * (1.0 / grids.num_actions)
    playWithPolicy(grids, policy)

    print("========== policy evaluation... ==========")
    values = np.zeros(grids.num_states)
    print("initial value function:")
    print(values.reshape(grids.rows, grids.cols))
    discount = 1
    theta = 1e-7
    values = evaluatePolicy(grids, values, policy, discount, theta)
    values_inplace = evaluatePolicyInPlace(grids, values, policy, discount, theta)
    print("evalute policy value function:")
    print(values.reshape(grids.rows, grids.cols))
    print("evalute policy in-place value function:")
    print(values_inplace.reshape(grids.rows, grids.cols))
    print("========== policy evaluation... done! ==========")

    print("policy before greedify")
    print(policy)
    for s in range(grids.num_states):
        greedifyPolicy(grids, values, policy, s, discount)
    print("policy after greedify")
    print(policy)

    playWithPolicy(grids, policy)
    playWithPolicy(grids, policy, [0, 3])
    playWithPolicy(grids, policy, [3, 0])

    print("========== policy iteration... ==========")
    vis = gridsvisualizer.GridsVisualizer("Policy Iteration", grids.rows, grids.cols)
    policy = np.ones((grids.num_states, grids.num_actions)) * (1.0 / grids.num_actions)
    V, policy = policyIteration(grids, policy, discount, theta)
    print(V.reshape(grids.rows, grids.cols))
    print(policy)
    playWithPolicy(grids, policy)
    vis.visualize(V, policy, 0)
    print("========== policy iteration... done! ==========")

    print("========== value iteration... ==========")
    vis = gridsvisualizer.GridsVisualizer("Value Iteration", grids.rows, grids.cols)
    discount = 0.9
    theta = 0.1
    V, policy = valueIteration(grids, discount, theta)
    print(V.reshape(grids.rows, grids.cols))
    print(policy)
    playWithPolicy(grids, policy)
    vis.visualize(V, policy, 0)
    print("========== value iteration... done! ==========")

    print("========== value iteration2... ==========")
    vis = gridsvisualizer.GridsVisualizer("Value Iteration2", grids.rows, grids.cols)
    discount = 0.9
    theta = 0.1
    V, policy = valueIteration2(grids, discount, theta)
    print(V.reshape(grids.rows, grids.cols))
    print(policy)
    playWithPolicy(grids, policy)
    vis.visualize(V, policy, 0)
    print("========== value iteration2... done! ==========")


if __name__ == '__main__':
    main()
