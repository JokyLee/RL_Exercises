#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.25'


import numpy as np

from agents import dp
from environments import gridworld
from tools import gridsvisualizer


def playWithPolicy(env, policy):
    print("========= playing in the cliff world... ==========")
    state = env.env_start()
    while True:
        action = np.random.choice(range(env.num_actions), p=policy[state])
        print("S={}, A={}".format(env.idx2Coord(state), gridworld.Actions(action)))
        reward, state, term = env.env_step(action)
        print("S'={}, R={}, T={}".format(env.idx2Coord(state), reward, term))
        if term:
            print("reached the end")
            break
    print("========== playing in the cliff world... done! ==========")


def main():
    env_config = {
        "seed": 0
    }
    cliff_env = gridworld.CliffWorldEnvironment()
    cliff_env.env_init(env_config)
    # policy = np.ones((cliff_env.num_states, cliff_env.num_actions)) * (1.0 / cliff_env.num_actions)
    # playWithPolicy(cliff_env, policy)
    discount = 0.9
    theta = 0.1
    vis = gridsvisualizer.GridsVisualizer("", cliff_env.rows, cliff_env.cols)

    print("========== policy iteration... ==========")
    policy = np.ones((cliff_env.num_states, cliff_env.num_actions)) * (1.0 / cliff_env.num_actions)
    V, policy = dp.policyIteration(cliff_env, policy, discount, theta)
    print(V.reshape(cliff_env.rows, cliff_env.cols))
    print(policy)
    playWithPolicy(cliff_env, policy)
    vis.visualize(V, policy, 0)
    print("========== policy iteration... done! ==========")

    print("========== value iteration... ==========")
    discount = 0.9
    theta = 0.1
    V, policy = dp.valueIteration(cliff_env, discount, theta)
    print(V.reshape(cliff_env.rows, cliff_env.cols))
    print(policy)
    playWithPolicy(cliff_env, policy)
    vis.visualize(V, policy, 0)
    print("========== value iteration... done! ==========")

    print("========== value iteration2... ==========")
    discount = 0.9
    theta = 0.1
    V, policy = dp.valueIteration2(cliff_env, discount, theta)
    print(V.reshape(cliff_env.rows, cliff_env.cols))
    print(policy)
    playWithPolicy(cliff_env, policy)
    vis.visualize(V, policy, 0)
    print("========== value iteration2... done! ==========")


if __name__ == '__main__':
    main()
