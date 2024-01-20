#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

# If true, plot all best actions
PLOT_BEST_ACTIONS = True

# goal
GOAL = 100

# all states, including state 0 and state 100
STATES = np.arange(GOAL + 1)

# probability of head
HEAD_PROB = 0.4
HEAD_PROB_EX = [0.25, 0.55]


def value_iteration(head_prob):
    # state value
    state_value = np.zeros(GOAL + 1)
    state_value[GOAL] = 1.0

    sweeps_history = []

    # value iteration
    while True:
        old_state_value = state_value.copy()
        sweeps_history.append(old_state_value)

        for state in STATES[1:GOAL]:
            # get possilbe actions for current state
            actions = np.arange(min(state, GOAL - state) + 1)
            action_returns = []
            for action in actions:
                action_returns.append(
                    head_prob * state_value[state + action] + (1 - head_prob) * state_value[state - action])
            new_value = np.max(action_returns)
            state_value[state] = new_value
        delta = abs(state_value - old_state_value).max()
        if delta < 1e-9:
            sweeps_history.append(state_value)
            break

    # compute the optimal policy
    policy = np.zeros(GOAL + 1)
    optimal_actions = []
    for state in STATES[1:GOAL]:
        actions = np.arange(min(state, GOAL - state) + 1)
        action_returns = []
        for action in actions:
            action_returns.append(
                head_prob * state_value[state + action] + (1 - head_prob) * state_value[state - action])

        # round to resemble the figure in the book, see
        # https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/issues/83
        action_returns_rounded = np.round(action_returns[1:], 5)

        best_actions = actions[np.argwhere(action_returns_rounded == np.max(action_returns_rounded))]
        best_actions = np.squeeze(best_actions, 1) + 1
        for ac in best_actions:
            optimal_actions.append([state, ac])

        policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]
    optimal_actions = np.array(optimal_actions)

    return sweeps_history, policy, optimal_actions

def plot_value(sweeps_history, plot_final_value=False):
    if plot_final_value:
        state_value = sweeps_history[-1]
        plt.plot(state_value)
    else:
        for sweep, state_value in enumerate(sweeps_history):
            plt.plot(state_value, label='sweep {}'.format(sweep))
            plt.legend(loc='best')
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')

def plot_policy(policy, optimal_actions, plot_best_actions=False):
    if plot_best_actions:
        plt.scatter(optimal_actions[:, 0], optimal_actions[:, 1])
    else:
        plt.scatter(STATES, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

def figure_4_3():
    sweeps_history, policy, optimal_actions = value_iteration(HEAD_PROB)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    plot_value(sweeps_history, plot_final_value=False)

    plt.subplot(2, 1, 2)
    plot_policy(policy, optimal_actions, plot_best_actions=False)

    plt.savefig('../images/figure_4_3.png')
    plt.close()

def exercise_4_9():
    plt.figure(figsize=(20, 20))

    for i in range(2):
        sweeps_history, policy, optimal_actions = value_iteration(HEAD_PROB_EX[i])

        plt.subplot(2, 2, i + 1)
        plot_value(sweeps_history, plot_final_value=True)

        plt.subplot(2, 2, i + 3)
        plot_policy(policy, optimal_actions, plot_best_actions=1-i)

    plt.savefig('../images/ex_4_9.png')
    plt.close()

if __name__ == '__main__':
    figure_4_3()
    exercise_4_9()
