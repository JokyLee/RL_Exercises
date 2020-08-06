#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.08.03'


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from rlglue import RLGlue
from environments import gymenvs
from tools.trygym import tryGymEnv
from agents import actorcritic
from environments import pendulum


def runExp(env, agent, envConfig, agentConfig, numRuns, maxSteps):
    return_per_step = np.zeros((numRuns, maxSteps))
    rewards_per_step = np.zeros((numRuns, maxSteps))
    exp_avg_reward_per_step = np.zeros((numRuns, maxSteps))

    # using tqdm we visualize progress bars
    for run in tqdm(range(numRuns)):
        agentConfig['seed'] = run
        envConfig['seed'] = run
        rl_glue = RLGlue(env, agent)
        rl_glue.rl_init(agentConfig, envConfig)
        rl_glue.rl_start()

        total_return = 0.
        return_arr = []

        # exponential average reward without initial bias
        exp_avg_reward = 0.0
        exp_avg_reward_ss = 0.01
        exp_avg_reward_normalizer = 0

        for num_steps in range(maxSteps):
            rl_step_result = rl_glue.rl_step()

            reward = rl_step_result[0]
            total_return += reward
            return_arr.append(reward)
            avg_reward = rl_glue.rl_agent_message("avg_reward")

            exp_avg_reward_normalizer += exp_avg_reward_ss * (1 - exp_avg_reward_normalizer)
            ss = exp_avg_reward_ss / exp_avg_reward_normalizer
            exp_avg_reward += ss * (reward - exp_avg_reward)

            rewards_per_step[run][num_steps] = reward
            return_per_step[run][num_steps] = total_return
            exp_avg_reward_per_step[run][num_steps] = exp_avg_reward

    np.save('results/return_per_step', return_per_step)
    np.save('results/exp_avg_reward_per_step', exp_avg_reward_per_step)
    np.save('results/rewards_per_step', rewards_per_step)

    return_per_step = np.load('results/return_per_step.npy')
    exp_avg_reward_per_step = np.load('results/exp_avg_reward_per_step.npy')
    rewards_per_step = np.load('results/rewards_per_step.npy')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    xs = np.arange(len(return_per_step[0])) + 1
    plotFillBtw(ax1, xs, return_per_step)
    # cum_reward = np.cumsum(rewards_per_step, axis=1)
    # avg_reward = cum_reward / xs
    # plotFillBtw(ax1, xs, avg_reward)
    plotFillBtw(ax2, xs, exp_avg_reward_per_step)
    plotFillBtw(ax3, xs, rewards_per_step)
    plt.show()

    # rendering
    rl_glue.environment.render_flag = True
    rl_glue.rl_episode()
    rl_glue.environment.env.close()


def plotFillBtw(ax, xs, data_rxn, label=""):
    data_mean = np.mean(data_rxn, axis=0)
    data_std_err = np.std(data_rxn, axis=0) / np.sqrt(len(data_rxn))
    ax.fill_between(xs, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    ax.plot(xs, data_mean, linewidth=1.0, label=label)


def main():
    env_config = {
        "seed": 0,
    }
    # env = pendulum.PendulumEnvironment_Discrete
    env = pendulum.PendulumEnvironment

    agent_config = {
        "seed": 0,
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "num_tilings": 32,
        "num_tiles": 8,
        "iht_size": 4096,

        "actor_step_size": 2 ** (-2),
        "critic_step_size": 2 ** 1,
        "avg_reward_step_size": 2 ** (-6),
    }
    agent = actorcritic.ActorCriticSoftmaxAgent

    num_runs = 50
    max_steps = 20000
    runExp(env, agent, env_config, agent_config, num_runs, max_steps)


if __name__ == '__main__':
    main()
