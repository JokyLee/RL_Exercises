#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.08.03'


import time

import gym

from rlglue import BaseEnvironment


def createEnvironmentType(envName, typeName, maxEpisodeSteps=None):
    env_type = type(typeName, (GymEnvironment, ), {})
    env_type.initClass(envName, maxEpisodeSteps)
    return env_type


class GymEnvironment(BaseEnvironment):
    ENV_NAME = ""
    ENV = None
    action_space = None
    observation_space = None
    max_episode_steps = None

    @classmethod
    def initClass(cls, envName, maxEpisodeSteps):
        cls.ENV_NAME = envName
        cls.ENV = gym.make(cls.ENV_NAME)
        cls.action_space = cls.ENV.action_space
        cls.observation_space = cls.ENV.observation_space
        cls.max_episode_steps = maxEpisodeSteps

    def env_init(self, env_info={}):
        self.env = self.__class__.ENV
        if self.__class__.max_episode_steps is not None:
            self.env._max_episode_steps = self.__class__.max_episode_steps
        self.env.seed(env_info.get("seed"))
        self.render_flag = env_info.get("render_flag", False)
        self.sleep_time = env_info.get("sleep_time", 0.05)

    def env_start(self):
        observation = self.env.reset()
        return observation

    def env_step(self, action):
        if self.render_flag:
            self.env.render()
            time.sleep(self.sleep_time)
        observation, reward, term, _ = self.env.step(action)
        return (reward, observation, term)
