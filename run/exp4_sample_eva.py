#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.27'


import numpy as np

from agents import td
from agents import mc
from agents import dp
from rlglue import RLGlue
from environments import gridworld
from tools import gridsvisualizer


def runExp(env, agent, agent_config, env_config, episode_num, plot_freq, name=""):
    rl_glue = RLGlue(env, agent)
    rl_glue.rl_init(agent_config, env_config)
    policy = agent_config.get("policy")
    vis = gridsvisualizer.GridsVisualizer(name, env.rows, env.cols)
    vis.onInteractive()

    for episode in range(1, episode_num + 1):
        rl_glue.rl_episode()
        if episode % plot_freq == 0:
            values = rl_glue.agent.agent_message("values")
            vis.visualize(values, policy, episode)
    vis.offInteractive()
    values = rl_glue.agent.agent_message("values")
    vis.visualize(values, policy, episode_num)
    return values


def main():
    env_config = {
        "seed": 0
    }
    agent_config = {
        "discount": 1,
        "step_size": 0.01,
        "seed": 0
    }
    episode_num = 5000
    plot_freq = 100
    env = gridworld.CliffWorldEnvironment
    agent = td.TDAgent
    
    # policies 
    optimal_policy = np.ones(shape=(env.rows * env.cols, env.action_space.n)) / env.action_space.n
    optimal_policy[36] = 0
    optimal_policy[36, gridworld.Actions.UP.value] = 1
    for i in range(24, 35):
        optimal_policy[i] = 0
        optimal_policy[i, gridworld.Actions.RIGHT.value] = 1
    optimal_policy[35] = 0
    optimal_policy[35, gridworld.Actions.DOWN.value] = 1
    
    safe_policy = np.ones(shape=(env.rows * env.cols, env.action_space.n)) / env.action_space.n
    safe_policy[:env.cols - 1] = 0
    safe_policy[:env.cols - 1, gridworld.Actions.RIGHT.value] = 1
    for r in range(1, env.rows):
        idx = env.cols * r
        safe_policy[idx] = 0
        safe_policy[idx, gridworld.Actions.UP.value] = 1

    for r in range(env.rows):
        idx = env.cols * (r + 1) - 1
        safe_policy[idx] = 0
        safe_policy[idx, gridworld.Actions.DOWN.value] = 1

    cliff_env = env()
    cliff_env.env_init(env_config)
    discount = 0.9
    theta = 0.1
    _, value_iterated_policy = dp.valueIteration(cliff_env, discount, theta)

    print("============ TD method for evaluating policy start ============")
    agent_config['policy'] = optimal_policy
    runExp(env, agent, agent_config, env_config, episode_num, plot_freq, name="TD method for Optimal Policy")

    agent_config['policy'] = safe_policy
    runExp(env, agent, agent_config, env_config, episode_num, plot_freq, name="TD method for Safe Policy")

    agent_config['policy'] = value_iterated_policy
    runExp(env, agent, agent_config, env_config, episode_num, plot_freq, name="TD method for Value Iterated Policy")
    print("============ TD method for evaluating policy end ============")

    print("============ MC method for evaluating policy start ============")
    agent = mc.MonteCarloPredictionAgent
    episode_num = 10
    plot_freq = 1

    agent_config['policy'] = optimal_policy
    runExp(env, agent, agent_config, env_config, episode_num, plot_freq, name="MC method for Optimal Policy")

    agent_config['policy'] = safe_policy
    runExp(env, agent, agent_config, env_config, episode_num, plot_freq, name="MC method for Safe Policy")

    agent_config['policy'] = value_iterated_policy
    runExp(env, agent, agent_config, env_config, episode_num, plot_freq, name="MC method for Value Iterated Policy")
    print("============ MC method for evaluating policy end ============")


if __name__ == '__main__':
    main()
