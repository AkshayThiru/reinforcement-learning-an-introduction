#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# world height
WORLD_HEIGHT = 7

# world width
WORLD_WIDTH = 10

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_UPLEFT = 4
ACTION_UPRIGHT = 5
ACTION_DOWNLEFT = 6
ACTION_DOWNRIGHT = 7
ACTION_NONE = 8

# probability for exploration
EPSILON = 0.1

# Sarsa step size
ALPHA = 0.5

# reward for each step
REWARD = -1.0

START = [3, 0]
GOAL = [3, 7]
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
ACTIONS_DIAG = [*ACTIONS, ACTION_UPLEFT, ACTION_UPRIGHT,
                ACTION_DOWNLEFT, ACTION_DOWNRIGHT]
ACTIONS_KING = [*ACTIONS_DIAG, ACTION_NONE]

def step(state, action, stochastic):
    i, j = state

    if stochastic and (WIND[j] > 0):
        wind = WIND[j] + np.random.randint(3) - 1
    else:
        wind = WIND[j]

    if action == ACTION_UP:
        return [max(i - 1 - wind, 0), j]
    elif action == ACTION_DOWN:
        return [max(min(i + 1 - wind, WORLD_HEIGHT - 1), 0), j]
    elif action == ACTION_LEFT:
        return [max(i - wind, 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        return [max(i - wind, 0), min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_UPLEFT:
        return [max(i - 1 - wind, 0), max(j - 1, 0)]
    elif action == ACTION_UPRIGHT:
        return [max(i - 1 - wind, 0), min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_DOWNLEFT:
        return [max(min(i + 1 - wind, WORLD_HEIGHT - 1), 0), max(j - 1, 0)]
    elif action == ACTION_DOWNRIGHT:
        return [max(min(i + 1 - wind, WORLD_HEIGHT - 1), 0), min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_NONE:
        return [max(i - wind, 0), j]
    else:
        assert False

# play for an episode
def episode(q_value, action_set, epsilon = EPSILON, stochastic = False):
    # track the total time steps in this episode
    time = 0

    # initialize state
    state = START

    # choose an action based on epsilon-greedy algorithm
    if np.random.binomial(1, epsilon) == 1:
        action = np.random.choice(action_set)
    else:
        values_ = q_value[state[0], state[1], :]
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # keep going until get to the goal state
    while state != GOAL:
        next_state = step(state, action, stochastic)
        if np.random.binomial(1, epsilon) == 1:
            next_action = np.random.choice(action_set)
        else:
            values_ = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

        # Sarsa update
        q_value[state[0], state[1], action] += \
            ALPHA * (REWARD + q_value[next_state[0], next_state[1], next_action] -
                     q_value[state[0], state[1], action])
        state = next_state
        action = next_action
        time += 1
    return time

def solve_TD0(ep_len, action_set, fig_path, stochastic = False):
    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(action_set)))
    episode_limit = ep_len

    steps = []
    for ep in tqdm(range(episode_limit)):
        steps.append(episode(q_value, action_set, stochastic))
        # time = episode(q_value)
        # episodes.extend([ep] * time)

    steps = np.add.accumulate(steps)

    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')

    plt.savefig(fig_path)
    plt.close()

    # display the optimal policy
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G ')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U ')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D ')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L ')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R ')
            elif bestAction == ACTION_UPLEFT:
                optimal_policy[-1].append('UL')
            elif bestAction == ACTION_UPRIGHT:
                optimal_policy[-1].append('UR')
            elif bestAction == ACTION_DOWNLEFT:
                optimal_policy[-1].append('DL')
            elif bestAction == ACTION_DOWNRIGHT:
                optimal_policy[-1].append('DR')
            elif bestAction == ACTION_NONE:
                optimal_policy[-1].append('N ')
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)
    print('Wind strength for each column:\n{}'.format([str(w) + ' ' for w in WIND]))

    print('Number of steps to goal: {}'.format(str(episode(q_value, action_set, epsilon = 0, stochastic = stochastic))))

if __name__ == '__main__':
    solve_TD0(500, ACTIONS, '../images/figure_6_3.png')
    print('')
    solve_TD0(500, ACTIONS_DIAG, '../images/ex_6_9a.png')
    print('')
    solve_TD0(500, ACTIONS_KING, '../images/ex_6_9b.png')
    print('')
    solve_TD0(500, ACTIONS_KING, '../images/ex_6_10.png', stochastic = True)
