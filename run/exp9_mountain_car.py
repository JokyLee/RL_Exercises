#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.08.02'


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from rlglue import RLGlue
from agents import sarsatiles
from environments import mountaincar


def runExp(env, agent, envConfig, agentConfig, numRuns, numEpisodes):
    all_steps = []
    for run in tqdm(range(numRuns)):
        agentConfig['seed'] = run
        envConfig['seed'] = run
        rl_glue = RLGlue(env, agent)
        rl_glue.rl_init(agentConfig, envConfig)
        steps_per_episode = []

        for episode in range(numEpisodes):
            rl_glue.rl_episode()
            steps_per_episode.append(rl_glue.num_steps)

        all_steps.append(steps_per_episode)
    plt.plot(np.mean(np.array(all_steps), axis=0))
    plt.show()
    # rendering
    rl_glue.environment.render = True
    rl_glue.rl_episode()
    rl_glue.environment.env.close()


def main():
    env_config = {
        "seed": 0,
        "num_states": 500,
    }
    env = mountaincar.MountainCarEnvironment

    agent_config = {
        "seed": 0,
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "num_tilings": 8,
        "num_tiles": 8,
        "iht_size": 4096,

        "discount": 1.0,
        "epsilon": 0.0,
        "step_size": 0.5,
        "initial_weights": 0.0,
    }
    agent = sarsatiles.SarseTilesAgent

    num_runs = 1
    num_episodes = 50
    runExp(env, agent, env_config, agent_config, num_runs, num_episodes)


if __name__ == '__main__':
    main()
