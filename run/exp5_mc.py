#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.27'


import numpy as np

from agents import mc
import rlglue
from environments import gridworld
from tools import gridsvisualizer


def runExp(env, agent, agent_config, env_config, episode_num, plot_freq, name="", episodeMaxStep=None):
    # rl_glue = rlglue.RLGlue(env, agent)
    rl_glue = rlglue.RLGlue_TimeoutAsTerminal(env, agent)
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


def main():
    env_config = {
        "seed": 0
    }
    # env = gridworld.CliffWorldEnvironment
    # env = gridworld.CliffWorldEnvironment_RewardGoal
    # env = gridworld.GridWorldEnvironment
    # env = gridworld.GridWorldEnvironment_RandomStart

    # 1
    env = gridworld.GridWorldEnvironment_RandomStart
    agent_config = {
        "num_states": env.rows * env.cols,
        "num_actions": env.action_space.n,
        "discount": 1,
        "step_size": 0.01,
        "seed": 0
    }
    agent = mc.MonteCarloControlAgent_ExploringStart
    print("============= {} + {} =============".format(env.__name__, agent.__name__))
    episode_num = 1000
    plot_freq = 100
    runExp(
        env, agent, agent_config, env_config, episode_num, plot_freq,
        name="{} + {}".format(env.__name__, agent.__name__), episodeMaxStep=100
    )
    print("============= {} + {} end =============".format(env.__name__, agent.__name__))

    # 2
    env = gridworld.GridWorldEnvironment
    agent_config = {
        "num_states": env.rows * env.cols,
        "num_actions": env.action_space.n,
        "discount": 1,
        "step_size": 0.01,
        "seed": 0
    }
    agent_config['epsilon'] = 0.1
    agent = mc.MonteCarloControlAgent_EpsilonSoft
    print("============= {} + {} =============".format(env.__name__, agent.__name__))
    episode_num = 1000
    plot_freq = 100
    runExp(
        env, agent, agent_config, env_config, episode_num, plot_freq,
        name="{} + {}".format(env.__name__, agent.__name__), episodeMaxStep=500
    )
    print("============= {} + {} end =============".format(env.__name__, agent.__name__))

    # 3
    env = gridworld.GridWorldEnvironment_RandomStart
    agent_config = {
        "num_states": env.rows * env.cols,
        "num_actions": env.action_space.n,
        "discount": 1,
        "step_size": 0.01,
        "seed": 0
    }
    agent_config['epsilon'] = 0.1
    agent = mc.MonteCarloControlAgent_EpsilonSoft
    print("============= {} + {} =============".format(env.__name__, agent.__name__))
    episode_num = 1000
    plot_freq = 100
    runExp(
        env, agent, agent_config, env_config, episode_num, plot_freq,
        name="{} + {}".format(env.__name__, agent.__name__), episodeMaxStep=500
    )
    print("============= {} + {} end =============".format(env.__name__, agent.__name__))

    # 4
    env = gridworld.GridWorldEnvironment
    agent_config = {
        "num_states": env.rows * env.cols,
        "num_actions": env.action_space.n,
        "discount": 1,
        "step_size": 0.01,
        "seed": 0
    }
    agent = mc.MonteCarloControlAgent_OffPolicy
    print("============= {} + {} =============".format(env.__name__, agent.__name__))
    episode_num = 1000
    plot_freq = 100
    runExp(
        env, agent, agent_config, env_config, episode_num, plot_freq,
        name="{} + {}".format(env.__name__, agent.__name__), episodeMaxStep=None
    )
    print("============= {} + {} end =============".format(env.__name__, agent.__name__))

    # 5
    env = gridworld.CliffWorldEnvironment_RewardGoal_RandomStart
    agent_config = {
        "num_states": env.rows * env.cols,
        "num_actions": env.action_space.n,
        "discount": 1,
        "step_size": 0.01,
        "seed": 0
    }
    agent_config['epsilon'] = 0.1
    agent = mc.MonteCarloControlAgent_EpsilonSoft
    print("============= {} + {} =============".format(env.__name__, agent.__name__))
    episode_num = 3000
    plot_freq = 100
    runExp(
        env, agent, agent_config, env_config, episode_num, plot_freq,
        name="{} + {}".format(env.__name__, agent.__name__), episodeMaxStep=500
    )
    print("============= {} + {} end =============".format(env.__name__, agent.__name__))

if __name__ == '__main__':
    main()
