#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.25'


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from environments import bandits
from agents import greedy


def showBanditsDistribution(env):
    data = []
    for a in range(env.arm_num):
        cur_arm_data = []
        for i in range(10000):
            reward = env.env_step(a)[0]
            cur_arm_data.append(reward)
        data.append(cur_arm_data)

    fig, ax = plt.subplots()
    ax.set_ylabel('Reward distribution')
    ax.set_xlabel('Actions')
    ax.violinplot(data, positions=range(len(data)), showmeans=True, showmedians=False, showextrema=False)
    plt.show()


def comparingGreedyAndEpsilon():
    env = bandits.BanditsEnvironment()
    env.env_init()
    # showBanditsDistribution(env)

    epsilon1 = 0.01
    epsilon2 = 0.1
    epsilon3 = 0.4
    agent_config = {
        "num_actions": env.action_num,
        "seed": None,
        "epsilon": None,
    }

    greedy_agent = greedy.GreedyAgent()
    epsilon_agent1 = greedy.GreedyEpsilonAgent()
    epsilon_agent2 = greedy.GreedyEpsilonAgent()
    epsilon_agent3 = greedy.GreedyEpsilonAgent()

    greedy_data = []
    epsilon_data1 = []
    epsilon_data2 = []
    epsilon_data3 = []

    run_num = 200
    step_num = 1000
    average_best = 0
    for run in tqdm(range(run_num)):
        env.env_init()
        average_best += np.max(env.arms)

        greedy_config = agent_config.copy()
        greedy_config['seed'] = run
        greedy_agent.agent_init(agent_config)

        epsilon_config1 = greedy_config.copy()
        epsilon_config1['epsilon'] = epsilon1
        epsilon_agent1.agent_init(epsilon_config1)

        epsilon_config2 = epsilon_config1.copy()
        epsilon_config2['epsilon'] = epsilon2
        epsilon_agent2.agent_init(epsilon_config2)

        epsilon_config3 = epsilon_config1.copy()
        epsilon_config3['epsilon'] = epsilon3
        epsilon_agent3.agent_init(epsilon_config3)

        state = env.env_start()
        action0 = greedy_agent.agent_start(state)
        action1 = epsilon_agent1.agent_start(state)
        action2 = epsilon_agent2.agent_start(state)
        action3 = epsilon_agent3.agent_start(state)

        greedy_cur = []
        epsilon1_cur = []
        epsilon2_cur = []
        epsilon3_cur = []

        for step in range(step_num):
            reward0, state0, _ = env.env_step(action0)
            reward1, state1, _ = env.env_step(action1)
            reward2, state2, _ = env.env_step(action2)
            reward3, state3, _ = env.env_step(action3)

            action0 = greedy_agent.agent_step(reward0, state0)
            action1 = epsilon_agent1.agent_step(reward1, state1)
            action2 = epsilon_agent2.agent_step(reward2, state2)
            action3 = epsilon_agent3.agent_step(reward3, state3)

            greedy_cur.append(greedy_agent.sum_award / (step + 1))
            epsilon1_cur.append(epsilon_agent1.sum_award / (step + 1))
            epsilon2_cur.append(epsilon_agent2.sum_award / (step + 1))
            epsilon3_cur.append(epsilon_agent3.sum_award / (step + 1))

        greedy_data.append(greedy_cur)
        epsilon_data1.append(epsilon1_cur)
        epsilon_data2.append(epsilon2_cur)
        epsilon_data3.append(epsilon3_cur)

    greedy_data = np.array(greedy_data).mean(axis=0)
    epsilon_data1 = np.array(epsilon_data1).mean(axis=0)
    epsilon_data2 = np.array(epsilon_data2).mean(axis=0)
    epsilon_data3 = np.array(epsilon_data3).mean(axis=0)

    fig, ax = plt.subplots()
    ax.plot(greedy_data, label="greedy")
    ax.plot(epsilon_data1, label="epsilon = {}".format(epsilon1))
    ax.plot(epsilon_data2, label="epsilon = {}".format(epsilon2))
    ax.plot(epsilon_data3, label="epsilon = {}".format(epsilon3))
    ax.plot([average_best / run_num] * len(greedy_data), label="Best posiible", linestyle="--")
    ax.legend()
    plt.show()


def comparingConstantStepSize(environmentChange):
    env = bandits.BanditsEnvironment()
    env.env_init()
    base_config = {
        "num_actions": env.action_num,
        "seed": None,
        "epsilon": 0.1,
        "step_size": None,
    }
    step_sizes = [0.01, 0.1, 0.5, 1.0, "1/N(A)"]
    agents = []
    for s in step_sizes:
        if s == "1/N(A)":
            agents.append(greedy.GreedyEpsilonAgent())
        else:
            agents.append(greedy.GreedyEpsilonAgentConstantStepSize())

    all_data = [[] for _ in range(len(agents))]
    run_num = 200
    step_num = 1000
    average_best = 0
    for run in tqdm(range(run_num)):
        env.env_init()
        average_best += np.max(env.arms)
        state = env.env_start()
        for ag, s in zip(agents, step_sizes):
            config = base_config.copy()
            config['seed'] = run
            config['step_size'] = s
            ag.agent_init(config)
            ag.agent_start(state)
        cur_data = [[] for _ in range(len(agents))]
        for step in range(step_num):
            for idx, ag in enumerate(agents):
                reward, state, _ = env.env_step(ag.last_action)
                ag.agent_step(reward, state)
                cur_data[idx].append(ag.sum_award / (step + 1))
            if environmentChange and step == step_num // 2:
                env.env_init()

        for idx in range(len(agents)):
            all_data[idx].append(cur_data[idx])

    fig, ax = plt.subplots()
    ax.plot([average_best / run_num] * step_num, label="Best posiible", linestyle="--")
    for idx in range(len(agents)):
        ax.plot(np.mean(all_data[idx], axis=0), label="step size = {}".format(step_sizes[idx]))
    ax.legend()
    plt.show()


if __name__ == '__main__':
    comparingGreedyAndEpsilon()
    comparingConstantStepSize(environmentChange=False)
    comparingConstantStepSize(environmentChange=True)
