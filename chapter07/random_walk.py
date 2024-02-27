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

# all states (except terminal states)
N_STATES = 19

# discount
GAMMA = 1

# all states but terminal states
STATES = np.arange(1, N_STATES + 1)

# start from the middle state
START_STATE = 10

# two terminal states
# an action leading to the left terminal state has reward -1
# an action leading to the right terminal state has reward 1
END_STATES = [0, N_STATES + 1]

# true state value from bellman equation
TRUE_VALUE = np.arange(-(N_STATES + 1), N_STATES + 3, 2) / (N_STATES + 1)
TRUE_VALUE[0] = TRUE_VALUE[-1] = 0

# n-steps TD method
# @value: values for each state, will be updated
# @n: # of steps
# @alpha: # step size
def temporal_difference(value, n, alpha, use_td_error = False):
    # initial starting state
    state = START_STATE

    # arrays to store states and rewards for an episode
    # space isn't a major consideration, so I didn't use the mod trick
    states = [state]
    rewards = [0] # first reward is R_1
    td_errors = []

    # track the time
    time = 0

    # the length of this episode
    T = float('inf')
    while True:
        # go to next time step
        time += 1

        if time < T:
            # choose an action randomly
            if np.random.binomial(1, 0.5) == 1:
                next_state = state + 1
            else:
                next_state = state - 1

            if next_state == 0:
                reward = -1
            elif next_state == 20:
                reward = 1
            else:
                reward = 0
            
            td_error = reward + GAMMA * value[next_state] - value[state]

            # store new state and new reward
            states.append(next_state)
            rewards.append(reward)
            td_errors.append(td_error)

            if next_state in END_STATES:
                T = time

        # get the time of the state to update
        update_time = time - n
        if update_time >= 0:
            if use_td_error:
                nstep_error = 0.0
                # calculate cumulative error
                for t in range(update_time, min(T, update_time + n)):
                    nstep_error += pow(GAMMA, t - update_time) * td_errors[t]
                state_to_update = states[update_time]
                # update the state value
                if not state_to_update in END_STATES:
                    value[state_to_update] += alpha * nstep_error
            else:
                returns = 0.0
                # calculate corresponding rewards
                for t in range(update_time + 1, min(T, update_time + n) + 1):
                    returns += pow(GAMMA, t - update_time - 1) * rewards[t]
                # add state value to the return
                if update_time + n <= T:
                    returns += pow(GAMMA, n) * value[states[(update_time + n)]]
                state_to_update = states[update_time]
                # update the state value
                if not state_to_update in END_STATES:
                    value[state_to_update] += alpha * (returns - value[state_to_update])
        if update_time == T - 1:
            break
        state = next_state

def hyperparameter_sweep(use_td_error, fig_path):
    # all possible steps
    steps = np.power(2, np.arange(0, 10))

    # all possible alphas
    alphas = np.arange(0, 1.1, 0.1)

    # each run has 10 episodes
    episodes = 10

    # perform 100 independent runs
    runs = 100

    # track the errors for each (step, alpha) combination
    errors = np.zeros((len(steps), len(alphas)))
    for run in tqdm(range(0, runs)):
        for step_ind, step in enumerate(steps):
            for alpha_ind, alpha in enumerate(alphas):
                # print('run:', run, 'step:', step, 'alpha:', alpha)
                value = np.zeros(N_STATES + 2)
                for ep in range(0, episodes):
                    temporal_difference(value, step, alpha, use_td_error = use_td_error)
                    # calculate the RMS error
                    errors[step_ind, alpha_ind] += np.sqrt(np.sum(np.power(value - TRUE_VALUE, 2)) / N_STATES)
    # take average
    errors /= episodes * runs

    for i in range(0, len(steps)):
        plt.plot(alphas, errors[i, :], label='n = %d' % (steps[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.ylim([0.25, 0.55])
    plt.legend()

    plt.savefig(fig_path)
    plt.close()

# Figure 7.2, it will take quite a while
def figure7_2():
    hyperparameter_sweep(False, '../images/figure_7_2.png')

def exercise7_2a():
    hyperparameter_sweep(True, '../images/ex_7_2a.png')

def exercise7_2b():
    # the best step size and alpha are selected from hyperparameter sweep
    nstep_error_step = 4
    td_error_step = 4

    nstep_error_alpha = 0.4
    td_error_alpha = 0.4

    episodes = 100

    runs = 100

    # track the errors for each epsiode
    nstep_error_errors = np.zeros(episodes)
    td_error_errors = np.zeros(episodes)
    for run in tqdm(range(0, runs)):
        nstep_error_value = np.zeros(N_STATES + 2)
        td_error_value = np.zeros(N_STATES + 2)
        for ep in range(0, episodes):
            temporal_difference(nstep_error_value, nstep_error_step, nstep_error_alpha, use_td_error = False)
            temporal_difference(td_error_value, td_error_step, td_error_alpha, use_td_error = True)
            nstep_error_errors[ep] += np.sqrt(np.sum(np.power(nstep_error_value - TRUE_VALUE, 2)) / N_STATES)
            td_error_errors[ep] += np.sqrt(np.sum(np.power(td_error_value - TRUE_VALUE, 2)) / N_STATES)
    # take average
    nstep_error_errors /= runs
    td_error_errors /= runs

    plt.plot(np.arange(1, episodes+1), nstep_error_errors, label='eq. (7.2)')
    plt.plot(np.arange(1, episodes+1), td_error_errors, label='td error')
    plt.xlabel('episodes')
    plt.ylabel('RMS error')
    plt.ylim([0.0, 0.55])
    plt.legend()

    plt.savefig('../images/ex_7_2b.png')
    plt.close()

if __name__ == '__main__':
    figure7_2()
    exercise7_2a()
    exercise7_2b()


