#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.08.03'


import time

import gym


def tryGymEnv(envName, sleepTime):
    env = gym.make(envName)
    action_space = env.action_space
    observation_space = env.observation_space
    print("action_space:", action_space)
    print("observation_space:", observation_space)
    print("observation_space.high:", observation_space.high)
    print("observation_space.low:", observation_space.low)


    env.reset()
    for _ in range(1000):
        env.render()
        time.sleep(sleepTime)
        env.step(env.action_space.sample())  # take a random action
    env.close()
