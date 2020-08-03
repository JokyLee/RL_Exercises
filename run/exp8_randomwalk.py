#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.08.01'


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from rlglue import RLGlue
from agents import gradienteva
from agents import neuralnetwork
from environments import randomwalk


def runExp(env, agent, envConfig, agentConfig, numRuns, numEpisodes):
    rl_glue = RLGlue(env, agent)
    all_values = []
    for run in tqdm(range(numRuns)):
        agentConfig['seed'] = run
        envConfig['seed'] = run
        rl_glue.rl_init(agentConfig, envConfig)
        for episode in range(numEpisodes):
            # print("episode", episode)
            rl_glue.rl_episode()
        all_values.append(rl_glue.rl_agent_message("values"))
    return all_values


def plotValues(values, envConfig):
    xs = list(range(envConfig['num_states']))[1:-1]
    plt.plot(xs, values.mean(axis=0)[1:-1])
    plt.show()


def main():
    env_config = {
        "seed": 0,
        "num_states": 500,
    }
    env = randomwalk.RandomWalkEnvironment

    agent_config = {
        "num_states": env_config['num_states'],
        "num_actions": env.action_space.n,
        "num_groups": 10,
        "discount": 1.0,
        "step_size": 0.01,
        "seed": 0,
    }
    agent = gradienteva.SemiGradientTDZeroAgent

    num_runs = 50
    num_episodes = 2000
    all_values = runExp(env, agent, env_config, agent_config, num_runs, num_episodes)
    np.save('results/semi_TD_all_values', all_values)
    all_values = np.load('results/semi_TD_all_values.npy')
    plotValues(all_values, env_config)

    agent = gradienteva.GradientMonteCarloAgent
    all_values = runExp(env, agent, env_config, agent_config, num_runs, num_episodes)
    np.save('results/gradient_MC_all_values', all_values)
    all_values = np.load('results/gradient_MC_all_values.npy')
    plotValues(all_values, env_config)

    agent_config = {
        "seed": 0,
        "num_states": env_config['num_states'],
        "num_actions": env.action_space.n,
        "num_groups": 10,
        "discount": 1.0,

        "num_hidden_units": 100,
        "beta_m": 0.9,
        "beta_v": 0.999,
        "epsilon": 0.0001,
        "step_size": 0.001,
    }
    agent = neuralnetwork.TDAgent
    num_runs = 20
    num_episodes = 5000
    all_values = runExp(env, agent, env_config, agent_config, num_runs, num_episodes)
    np.save('results/TD_NN_all_values', all_values)
    all_values = np.load('results/TD_NN_all_values.npy')
    plotValues(all_values, env_config)


if __name__ == '__main__':
    main()
