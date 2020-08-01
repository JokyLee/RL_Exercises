#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.07.29'


import time

from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import rlglue
from agents import dyna
from environments import gridworld


SHORTCUT_ENV_CHANGE_AT = 3000


def runExpWithDifferentPlanningSteps(env, agent, envConfig, agentConfig, numRuns, numEpisode, planningSteps):
    data = {}
    data["planning_steps"] = planningSteps
    all_averages = np.zeros((len(planningSteps), numRuns, numEpisode))
    state_visits = np.zeros((len(planningSteps), numRuns, agentConfig["num_states"]))
    for idx, planning_steps in enumerate(planningSteps):
        print('Planning steps : ', planning_steps)
        time.sleep(0.5)
        agentConfig["planning_steps"] = planning_steps

        for run in tqdm(range(numRuns)):
            agentConfig['seed'] = run
            agentConfig['planning_seed'] = run
            rl_glue = rlglue.RLGlue(env, agent)          # Creates a new RLGlue experiment with the env and agent we chose above
            rl_glue.rl_init(agentConfig, envConfig) # We pass RLGlue what it needs to initialize the agent and environment

            for j in range(numEpisode):
                state, _ = rl_glue.rl_start()
                state_visits[idx][run][state] += 1
                num_steps = 0
                while True:
                    reward, state, action, is_terminal = rl_glue.rl_step()
                    state_visits[idx][run][state] += 1
                    num_steps += 1
                    if is_terminal:
                        break
                all_averages[idx][run][j] = num_steps
    data['all_averages'] = all_averages
    data['state_visits'] = state_visits
    return data


def plotDataForDifferentPlanningSteps(planningSteps, allAverages):
    for i, planning_steps in enumerate(planningSteps):
        plt.plot(np.mean(allAverages[i], axis=0), label='Planning steps = '+str(planning_steps))
    plt.legend(loc='upper right')
    plt.xlabel('Episodes')
    plt.ylabel('Steps\nper\nepisode', rotation=0, labelpad=40)
    plt.axhline(y=16, linestyle='--', color='grey', alpha=0.4)
    plt.show()


def plotComparingTwoVisits(data, titles, shortCutEnv=False):
    positions = [211, 212]
    if shortCutEnv:
        wall_ends = [None, -1]
    else:
        wall_ends = [None, None]

    for i in range(2):
        state_visits = data[i]
        average_state_visits = np.mean(state_visits, axis=0)
        grid_state_visits = np.rot90(average_state_visits.reshape((6, 9)).T)
        grid_state_visits[2, 1:wall_ends[i]] = np.nan # walls
        plt.subplot(positions[i])
        plt.pcolormesh(grid_state_visits, edgecolors='gray', linewidth=1, cmap='viridis')
        plt.text(3+0.5, 0+0.5, 'S', horizontalalignment='center', verticalalignment='center')
        plt.text(8+0.5, 5+0.5, 'G', horizontalalignment='center', verticalalignment='center')
        plt.title(titles[i])
        plt.axis('off')
        cm = plt.get_cmap()
        cm.set_bad('gray')

    plt.subplots_adjust(left=0.08, bottom=0.05, right=0.65, top=0.92)
    cax = plt.axes([0.73, 0.05, 0.075, 0.92])
    cbar = plt.colorbar(cax=cax)
    plt.show()


def playingMazeDynaQ():
    env_config = {
        "seed": 0,
    }
    env = gridworld.MazeEnvironment

    agent_config = {
        "num_states": env.rows * env.cols,
        "num_actions": env.action_space.n,
        "discount": 0.95,
        "epsilon": 0.1,
        "step_size": 0.125,
        "planning_steps": 0,
        "seed": 0,
        "planning_seed": 0,
    }
    agent = dyna.DynaQAgent

    num_runs = 30
    num_episodes = 40
    planning_steps_all = [0, 5, 50]

    data = runExpWithDifferentPlanningSteps(
        env, agent, env_config, agent_config, num_runs, num_episodes, planning_steps_all
    )
    np.save('results/Dyna_Q_with_different_planning_steps', data)
    data = np.load('results/Dyna_Q_with_different_planning_steps.npy', allow_pickle=True).item()
    plotDataForDifferentPlanningSteps(data['planning_steps'], data['all_averages'])
    plotComparingTwoVisits(
        [data['state_visits'][0], data['state_visits'][1]],
        ["DynaQ: State visitations with planning step 0", "DynaQ: State visitations with planning step 5"],
        shortCutEnv=False
    )
    plotComparingTwoVisits(
        [data['state_visits'][1], data['state_visits'][2]],
        ["DynaQ: State visitations with planning step 5", "DynaQ: State visitations with planning step 50"],
        shortCutEnv=False
    )


def runExpWithDifferentPlanningSteps_ChangingEnv(
        env, agent, envConfig, agentConfig, numRuns, planningSteps, numMaxSteps
):
    assert numMaxSteps > 1
    data = {}
    data["planning_steps"] = planningSteps

    state_visits_before_change = np.zeros((len(planningSteps), numRuns, agentConfig["num_states"]))
    state_visits_after_change = np.zeros((len(planningSteps), numRuns, agentConfig["num_states"]))
    cum_reward_all = np.zeros((len(planningSteps), numRuns, numMaxSteps))

    for idx, planning_steps in enumerate(planningSteps):
        print('Planning steps : ', planning_steps)
        time.sleep(0.5)
        agentConfig["planning_steps"] = planning_steps

        for run in tqdm(range(numRuns)):
            agentConfig['seed'] = run
            agentConfig['planning_seed'] = run
            rl_glue = rlglue.RLGlue(env, agent)
            rl_glue.rl_init(agentConfig, envConfig)
            num_steps = 0
            cum_reward = 0

            while num_steps < numMaxSteps - 1:
                state, action = rl_glue.rl_start()
                if num_steps < envConfig["change_at_n"]:
                    state_visits_before_change[idx][run][state] += 1
                else:
                    state_visits_after_change[idx][run][state] += 1

                while num_steps < numMaxSteps - 1:
                    reward, state, action, is_terminal = rl_glue.rl_step()

                    if num_steps < envConfig["change_at_n"]:
                        state_visits_before_change[idx][run][state] += 1
                    else:
                        state_visits_after_change[idx][run][state] += 1
                    cum_reward += reward
                    num_steps += 1
                    cum_reward_all[idx][run][num_steps] = cum_reward

                    if is_terminal:
                        break

    data['cum_reward_all'] = cum_reward_all
    data['state_visits_before_change'] = state_visits_before_change
    data['state_visits_after_change'] = state_visits_after_change
    return data


def plotShortcutMazRewardWithPlanningSteps(cumRewards, planningSteps, envConfig, title):
    for i, step in enumerate(planningSteps):
        plt.plot(np.mean(cumRewards[i], axis=0), label="planning step = {}".format(step))

    plt.axvline(x=envConfig['change_at_n'], linestyle='--', color='grey', alpha=0.4)
    plt.xlabel('Timesteps')
    plt.ylabel("Cumulative reward", rotation=0, labelpad=60)
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()


def playingShortcutMaze(dynaQPlus=False):
    env_config = {
        "seed": 10,
        "change_at_n": SHORTCUT_ENV_CHANGE_AT,
    }
    env = gridworld.ShortcutMazeEnvironment

    if dynaQPlus:
        agent_config = {
            "num_states": env.rows * env.cols,
            "num_actions": env.action_space.n,
            "discount": 0.95,
            "epsilon": 0.1,
            "step_size": 0.5,
            "kappa": 0.001,
            "planning_steps": 0,
            "seed": 0,
            "planning_seed": 0,
        }
        agent = dyna.DynaQPlusAgent
        file_name = "results/Dyna_Q_Plus_Shortcut_Maze"
        alg_name = "Dyna-Q-Plus"
    else:
        agent_config = {
            "num_states": env.rows * env.cols,
            "num_actions": env.action_space.n,
            "discount": 0.95,
            "epsilon": 0.1,
            "step_size": 0.125,
            "planning_steps": 0,
            "seed": 0,
            "planning_seed": 0,
        }
        agent = dyna.DynaQAgent
        file_name = "results/Dyna_Q_Shortcut_Maze"
        alg_name = "Dyna-Q"

    num_runs = 10
    num_max_steps = 6000
    planning_steps_all = [5, 10, 50]

    data = runExpWithDifferentPlanningSteps_ChangingEnv(
        env, agent, env_config, agent_config, num_runs, planning_steps_all, num_max_steps
    )
    np.save(file_name, data)

    data = np.load('{}.npy'.format(file_name), allow_pickle=True).item()
    plotShortcutMazRewardWithPlanningSteps(
        data['cum_reward_all'], data['planning_steps'], env_config,
        "{}: Shortcut Maze environment".format(alg_name)
    )
    plotComparingTwoVisits(
        [data['state_visits_before_change'][-1], data['state_visits_after_change'][-1]],
        ["{}: State visitations before environment change".format(alg_name),
         "{}: State visitations after environment change".format(alg_name)],
        shortCutEnv=True
    )


def comparingInShortCutEnv():
    dynaq_data = np.load('results/Dyna_Q_Shortcut_Maze.npy', allow_pickle=True).item()
    dynaq_plus_data = np.load('results/Dyna_Q_Plus_Shortcut_Maze.npy', allow_pickle=True).item()
    cum_reward_q = dynaq_data['cum_reward_all'][-1]
    cum_reward_qPlus = dynaq_plus_data['cum_reward_all'][-1]

    plt.plot(np.mean(cum_reward_qPlus, axis=0), label='Dyna-Q+')
    plt.plot(np.mean(cum_reward_q, axis=0), label='Dyna-Q')

    plt.axvline(x=SHORTCUT_ENV_CHANGE_AT, linestyle='--', color='grey', alpha=0.4)
    plt.xlabel('Timesteps')
    plt.ylabel('Cumulative\nreward', rotation=0, labelpad=60)
    plt.legend(loc='upper left')
    plt.title('Average performance of Dyna-Q and Dyna-Q+ agents in the Shortcut Maze\n')
    plt.show()


def main():
    playingMazeDynaQ()
    playingShortcutMaze(dynaQPlus=False)
    playingShortcutMaze(dynaQPlus=True)
    comparingInShortCutEnv()


if __name__ == '__main__':
    main()
