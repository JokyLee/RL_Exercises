#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.28'

from tqdm import tqdm
import numpy as np
from scipy.stats import sem
import matplotlib
import matplotlib.pyplot as plt

import rlglue
from agents import td
from tools import gridsvisualizer
from environments import gridworld


def runExp(env, agent, agent_config, env_config, episode_num, plot_freq, name="", episodeMaxStep=None):
    rl_glue = rlglue.RLGlue(env, agent)
    # rl_glue = rlglue.RLGlue_TimeoutAsTerminal(env, agent)
    rl_glue.rl_init(agent_config, env_config)
    vis = gridsvisualizer.GridsVisualizer(name, env.rows, env.cols)
    vis.onInteractive()

    for episode in range(1, episode_num + 1):
        rl_glue.rl_episode(episodeMaxStep)
        policy = rl_glue.agent.agent_message("policy")
        if episode % plot_freq == 0:
            values = rl_glue.agent.agent_message("values")
            vis.visualize(values, policy, episode)
    vis.offInteractive()
    policy = rl_glue.agent.agent_message("policy")
    values = rl_glue.agent.agent_message("values")
    vis.visualize(values, policy, episode_num)
    return values


def compareAlgorithms():
    agents = {
        "Sarsa": td.SarsaAgent,
        "Q-learning": td.QLearningAgent,
        "Expected Sarsa": td.ExpectedSarsaAgent,
    }
    env = gridworld.CliffWorldEnvironment
    all_reward_sums = {}  # Contains sum of rewards during episode
    all_state_visits = {}  # Contains state visit counts during the last 10 episodes
    agent_info = {
        "num_states": env.rows * env.cols,
        "num_actions": env.action_space.n,
        "discount": 1,
        "epsilon": 0.1,
        "step_size": 0.5,
        "seed": 0
    }
    env_info = {
        "seed": 0
    }
    num_runs = 100  # The number of runs
    num_episodes = 500  # The number of episodes in each run

    for algorithm in ["Sarsa", "Q-learning", "Expected Sarsa"]:
        all_reward_sums[algorithm] = []
        all_state_visits[algorithm] = []
        for run in tqdm(range(num_runs)):
            agent_info["seed"] = run
            rl_glue = rlglue.RLGlue(env, agents[algorithm])
            rl_glue.rl_init(agent_info, env_info)

            reward_sums = []
            state_visits = np.zeros(48)
            #         last_episode_total_reward = 0
            for episode in range(num_episodes):
                if episode < num_episodes - 10:
                    # Runs an episode
                    rl_glue.rl_episode()
                else:
                    # Runs an episode while keeping track of visited states
                    state, action = rl_glue.rl_start()
                    state_visits[state] += 1
                    is_terminal = False
                    while not is_terminal:
                        reward, state, action, is_terminal = rl_glue.rl_step()
                        state_visits[state] += 1

                reward_sums.append(rl_glue.rl_return())
            #             last_episode_total_reward = rl_glue.rl_return()

            all_reward_sums[algorithm].append(reward_sums)
            all_state_visits[algorithm].append(state_visits)

    for algorithm in ["Sarsa", "Q-learning", "Expected Sarsa"]:
        plt.plot(np.mean(all_reward_sums[algorithm], axis=0), label=algorithm)
    plt.xlabel("Episodes")
    plt.ylabel("Sum of\n rewards\n during\n episode", rotation=0, labelpad=40)
    plt.xlim(0, 500)
    plt.ylim(-100, 0)
    plt.legend()
    plt.show()

    for algorithm, position in [("Sarsa", 311), ("Q-learning", 312), ("Expected Sarsa", 313)]:
        plt.subplot(position)
        average_state_visits = np.array(all_state_visits[algorithm]).mean(axis=0)
        grid_state_visits = average_state_visits.reshape((4, 12))[::-1]
        grid_state_visits[0, 1:-1] = np.nan
        plt.pcolormesh(grid_state_visits, edgecolors='gray', linewidth=2)
        plt.title(algorithm)
        plt.axis('off')
        cm = plt.get_cmap()
        cm.set_bad('gray')

        plt.subplots_adjust(left=0.08, bottom=0.05, right=0.65, top=0.92)
        cax = plt.axes([0.73, 0.05, 0.075, 0.92])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.set_ylabel("Visits during\n the last 10\n episodes", rotation=0, labelpad=30)
    # cbar.ax.set_ylabel("Visits during\n the last 10\n episodes")
    plt.show()


def compareStepSize():
    agents = {
        "Sarsa": td.SarsaAgent,
        "Q-learning": td.QLearningAgent,
        "Expected Sarsa": td.ExpectedSarsaAgent,
    }
    env = gridworld.CliffWorldEnvironment
    step_sizes = np.linspace(0.1, 1.0, 10)
    agent_info = {
        "num_states": env.rows * env.cols,
        "num_actions": env.action_space.n,
        "discount": 1,
        "epsilon": 0.1,
        "step_size": 0.5,
        "seed": 0
    }
    env_info = {
        "seed": 0
    }
    num_runs = 100
    num_episodes = 100
    all_reward_sums = {}

    for algorithm in ["Sarsa", "Q-learning", "Expected Sarsa"]:
        for step_size in step_sizes:
            all_reward_sums[(algorithm, step_size)] = []
            agent_info["step_size"] = step_size
            for run in tqdm(range(num_runs)):
                agent_info["seed"] = run
                rl_glue = rlglue.RLGlue(env, agents[algorithm])
                rl_glue.rl_init(agent_info, env_info)

                return_sum = 0
                for episode in range(num_episodes):
                    rl_glue.rl_episode()
                    return_sum += rl_glue.rl_return()
                all_reward_sums[(algorithm, step_size)].append(return_sum / num_episodes)

    for algorithm in ["Sarsa", "Q-learning", "Expected Sarsa"]:
        algorithm_means = np.array([np.mean(all_reward_sums[(algorithm, step_size)]) for step_size in step_sizes])
        algorithm_stds = np.array([sem(all_reward_sums[(algorithm, step_size)]) for step_size in step_sizes])
        plt.plot(step_sizes, algorithm_means, marker='o', linestyle='solid', label=algorithm)
        plt.fill_between(step_sizes, algorithm_means + algorithm_stds, algorithm_means - algorithm_stds, alpha=0.2)

    plt.legend()
    plt.xlabel("Step-size")
    plt.ylabel("Sum of\n rewards\n per episode", rotation=0, labelpad=50)
    plt.xticks(step_sizes)
    plt.show()


def main():
    env_config = {
        "seed": 0
    }
    env = gridworld.CliffWorldEnvironment

    agent_config = {
        "num_states": env.rows * env.cols,
        "num_actions": env.action_space.n,
        "discount": 1,
        "epsilon": 0.1,
        "step_size": 0.5,
        "seed": 0
    }

    # agent = td.QLearningAgent
    # print("============= {} + {} =============".format(env.__name__, agent.__name__))
    # episode_num = 500
    # plot_freq = 50
    # runExp(
    #     env, agent, agent_config, env_config, episode_num, plot_freq,
    #     name="{} + Q-Learning Behaviour Policy".format(env.__name__), episodeMaxStep=None
    # )
    # print("============= {} + {} end =============".format(env.__name__, agent.__name__))

    # agent = td.SarsaAgent
    # print("============= {} + {} =============".format(env.__name__, agent.__name__))
    # episode_num = 500
    # plot_freq = 50
    # runExp(
    #     env, agent, agent_config, env_config, episode_num, plot_freq,
    #     name="{} + {}".format(env.__name__, agent.__name__), episodeMaxStep=None
    # )
    # print("============= {} + {} end =============".format(env.__name__, agent.__name__))

    # agent = td.ExpectedSarsaAgent
    # print("============= {} + {} =============".format(env.__name__, agent.__name__))
    # episode_num = 500
    # plot_freq = 50
    # runExp(
    #     env, agent, agent_config, env_config, episode_num, plot_freq,
    #     name="{} + {}".format(env.__name__, agent.__name__), episodeMaxStep=None
    # )
    # print("============= {} + {} end =============".format(env.__name__, agent.__name__))

    # compareAlgorithms()
    compareStepSize()


if __name__ == '__main__':
    main()
