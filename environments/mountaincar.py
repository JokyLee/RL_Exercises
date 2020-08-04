#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.08.03'


import time

import gym

from rlglue import BaseEnvironment


class MountainCarEnvironment(BaseEnvironment):
    ENV = gym.make("MountainCar-v0")
    action_space = ENV.action_space
    observation_space = ENV.observation_space

    def env_init(self, env_info={}):
        self.env = self.__class__.ENV
        self.env._max_episode_steps = 15000
        self.env.seed(env_info.get("seed"))
        self.render = env_info.get("render", False)
        self.sleep_time = env_info.get("sleep_time", 0.05)

    def env_start(self):
        observation = self.env.reset()
        return observation

    def env_step(self, action):
        if self.render:
            self.env.render()
            time.sleep(self.sleep_time)
        observation, reward, term, _ = self.env.step(action)
        return (reward, observation, term)
