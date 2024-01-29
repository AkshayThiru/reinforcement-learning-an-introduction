import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from tqdm import tqdm
from itertools import product
import os


## Action space and velocity profiles.
MAX_ACCELERATION = 1
acc_ = np.arange(-MAX_ACCELERATION, MAX_ACCELERATION + 1, dtype=int)
ACTIONS = np.dstack(np.meshgrid(acc_, acc_)).reshape(-1, 2)
def inverse_actions(ac):
    return (ac[1] + MAX_ACCELERATION) * (2 * MAX_ACCELERATION + 1) + \
        ac[0] + MAX_ACCELERATION

MAX_VELOCITY = 4
VELOCITY_PROFILE = np.empty((MAX_VELOCITY + 1, MAX_VELOCITY + 1), dtype=object)

## Hyper-parameters.
GAMMA = 1
EPSILON_B = 5e-2 # Epsilon for the soft behaviour policy.
EPSILON_T = 2e-3 # Epsilon for the soft target policy.
EPSILON_T_DEPLOY = 0
EPSILON_B = min(EPSILON_B, 1 / ACTIONS.shape[0])
EPSILON_T = min(EPSILON_T, 1 / ACTIONS.shape[0])
NOISE_PROB = 0.1
EPSIODES = [int(1e5), int(1e5)]
REWARD = -1

## Racetracks.
GRID_EMPTY = 1
GRID_OCCUPIED = 0
GRID_START = 2
GRID_END = 3
GRID_CAR = 4

# Racetrack coordinates start from the bottom-left.
# y-axis is flipped when displaying.
RACETRACK1 = np.ones((17, 32), dtype=int) * GRID_EMPTY
RACETRACK1[0:2, -3:] = GRID_OCCUPIED #
RACETRACK1[0, -4] = GRID_OCCUPIED
RACETRACK1[2, -1] = GRID_OCCUPIED
RACETRACK1[0, 0:18] = GRID_OCCUPIED #
RACETRACK1[1, 0:10] = GRID_OCCUPIED
RACETRACK1[2, 0:3] = GRID_OCCUPIED
RACETRACK1[10:, -7] = GRID_OCCUPIED
RACETRACK1[9:, -8::-1] = GRID_OCCUPIED
START_POS1 = np.zeros((6, 2), dtype=int)
START_POS1[:, 0] = np.arange(3, 9, dtype=int)
RACETRACK1[START_POS1[:, 0], START_POS1[:, 1]] = GRID_START
RACETRACK1[-1, -6:] = GRID_END

RACETRACK2 = np.ones((32, 30), dtype=int) * GRID_EMPTY
occupied_ = np.array([p for p in product(range(32), range(30)) \
            if (p[1] >= p[0] + 3) and (p[0] <= 13) and (p[1] <= -p[0] + 33)])
RACETRACK2[occupied_[:, 0], occupied_[:, 1]] = GRID_OCCUPIED
RACETRACK2[0:11, -7:] = GRID_OCCUPIED
RACETRACK2[11, -3:] = GRID_OCCUPIED
RACETRACK2[12, -2:] = GRID_OCCUPIED
RACETRACK2[13:16, -1] = GRID_OCCUPIED
RACETRACK2[-9:, 0:17] = GRID_OCCUPIED
RACETRACK2[-8:, 17] = GRID_OCCUPIED
RACETRACK2[-6:, 18] = GRID_OCCUPIED
RACETRACK2[-5:, 19] = GRID_OCCUPIED
RACETRACK2[-2:, 20] = GRID_OCCUPIED
START_POS2 = np.zeros((23, 2), dtype=int)
START_POS2[:, 0] = np.arange(23, dtype=int)
RACETRACK2[START_POS2[:, 0], START_POS2[:, 1]] = GRID_START
RACETRACK2[-1, -9:] = GRID_END

RACETRACKS = [RACETRACK1, RACETRACK2]
START_POS = [START_POS1, START_POS2]
##

ASSERTS = False

DISPLAY_VELOCITY = True
FPS = 2 if DISPLAY_VELOCITY else 10
COLOR_MAP = np.array([[192, 192, 192], # 0: gray (outside)
                      [255, 255, 255], # 1: white (inside)
                      [255, 100, 100], # 2: red (start line)
                      [32, 224, 32],   # 3: green (finish line)
                      [0, 0, 0]])      # 4: black (car position)


def pos_step(pos, velocity):
    new_pos = pos + velocity
    if VELOCITY_PROFILE[velocity[0], velocity[1]] is None:
        traj_ = np.rint(velocity * \
                                np.linspace(0, 1, MAX_VELOCITY + 1)[:, np.newaxis]).astype(int)
        _, idx_ = np.unique(traj_, axis=0, return_index=True)
        VELOCITY_PROFILE[velocity[0], velocity[1]] = traj_[np.sort(idx_)]
    return new_pos, pos + VELOCITY_PROFILE[velocity[0], velocity[1]]

def step(track_id, state, ac, noise_prob):
    pos = state[0:2]
    vel = state[2:]

    # Assert validity of state, ac.
    if ASSERTS:
        grid_value = RACETRACKS[track_id][pos[0], pos[1]]
        invalid_pos = (grid_value == GRID_OCCUPIED) or (grid_value == GRID_END)
        assert not invalid_pos, 'invalid position (occupied or in end position)'
        if (vel[0] + ac[0] < 0) or (vel[1] + ac[1] < 0) or \
            (vel[0] + ac[0] > MAX_VELOCITY) or (vel[1] + ac[1] > MAX_VELOCITY):
            raise ValueError('vel+ac is out of bounds')
        if (pos.tolist() not in START_POS[track_id].tolist()) and \
            (vel[0] + ac[0] == 0) and (vel[1] + ac[1] == 0):
            raise ValueError('zero velocity when not in starting position')
    
    # Compute new state.
    if np.random.uniform() < noise_prob:
        new_vel = vel
    else:
        new_vel = vel + ac
    new_pos, traj = pos_step(pos, new_vel)
    finished = False
    new_traj = []
    for p_ in traj.tolist():
        if p_[0] >= RACETRACKS[track_id].shape[0] or p_[1] >= RACETRACKS[track_id].shape[1]:
            grid_value = GRID_OCCUPIED
        else:
            grid_value = RACETRACKS[track_id][p_[0], p_[1]]
        if grid_value == GRID_OCCUPIED:
            rand_idx = np.random.choice(START_POS[track_id].shape[0])
            new_pos = START_POS[track_id][rand_idx]
            new_vel = np.zeros((2,), dtype=int)
            new_traj.append(new_pos.tolist())
            break
        elif grid_value == GRID_END:
            new_pos = np.array(p_, dtype=int)
            finished = True
            new_traj.append(p_)
            break
        new_traj.append(p_)
    new_state = np.concatenate((new_pos, new_vel))
    return new_state, finished, np.array(new_traj, dtype=int)

def play(track_id, player_policy, noise_prob=NOISE_PROB):
    rand_idx = np.random.choice(START_POS[track_id].shape[0])
    pos = START_POS[track_id][rand_idx]
    vel = np.zeros((2,), dtype=int)
    state = np.concatenate((pos, vel))
    states = [state]
    actions = []
    ac_probs = []
    finished = False

    while not finished:
        ac, ac_prob, _ = player_policy(state)
        state, finished, _ = step(track_id, state, ac, noise_prob)
        states.append(state)
        actions.append(ac)
        ac_probs.append(ac_prob)
    return states, actions, ac_probs

def off_policy_mc(track_id, episodes):
    track = RACETRACKS[track_id]
    state_action_values = np.zeros(
        (track.shape[0], track.shape[1], MAX_VELOCITY + 1, MAX_VELOCITY + 1, \
         (2 * MAX_ACCELERATION + 1) ** 2), \
        dtype=float
    )
    # Initialize Q function.
    for i in range(track.shape[0]):
        for j in range(track.shape[1]):
            if track[i, j] == GRID_OCCUPIED:
                # Occupied grid cells are invalid.
                state_action_values[i, j, :, :, :] = -np.Inf
            elif track[i, j] == GRID_EMPTY:
                # Zero velocity in empty grid cells is invalid.
                state_action_values[i, j, 0, 0, :] = -np.Inf
            for vi in range(MAX_VELOCITY + 1):
                for vj in range(MAX_VELOCITY + 1):
                    for a in range((2 * MAX_ACCELERATION + 1) ** 2):
                        ac = ACTIONS[a]
                        if (vi + ac[0] < 0) or (vi + ac[0] > MAX_VELOCITY) or \
                            (vj + ac[1] < 0) or (vj + ac[1] > MAX_VELOCITY):
                            # Out-of-bound actions are invalid.
                            state_action_values[i, j, vi, vj, a] = -np.Inf
                        if (track[i, j] == GRID_EMPTY) and \
                            (vi + ac[0] == 0) and (vj + ac[1] == 0):
                            # Zero-velocity in empty grid cells is invalid.
                            state_action_values[i, j, vi, vj, a] = -np.Inf
    # Cumulative sum array.
    state_action_cum_sum = np.zeros_like(state_action_values)
    
    # eps-greedy policy.
    def eps_greedy_policy(state, eps):
        values_ = state_action_values[state[0], state[1], state[2], state[3], :]
        max_v_ = np.max(values_)
        max_ac_idx_ = [idx_ for idx_, value_ in enumerate(values_) if value_ == max_v_]
        ac_probs_ = np.ones_like(values_) * eps
        ac_probs_[values_ == -np.Inf] = 0
        ac_probs_[max_ac_idx_] += (1 - np.sum(ac_probs_)) / len(max_ac_idx_)
        idx_ = np.random.choice(len(values_), p=ac_probs_)
        return ACTIONS[idx_], ac_probs_[idx_], ac_probs_
    # Target policy:
    target_policy = lambda state: eps_greedy_policy(state, EPSILON_T)
    # Behaviour policy.
    behaviour_policy = lambda state: eps_greedy_policy(state, EPSILON_B)

    # Run MC control for some episodes.
    for _ in tqdm(range(episodes)):
        states, actions, ac_probs_b = play(track_id, behaviour_policy)
        G = 0
        W = 1
        T = len(actions)
        for t in reversed(range(T)):
            if W == 0:
                break
            G = GAMMA * G + REWARD
            st = states[t]
            ac = actions[t]
            ac_idx = inverse_actions(ac)
            ac_prob_b = ac_probs_b[t]
            state_action_cum_sum[st[0], st[1], st[2], st[3], ac_idx] += W
            state_action_values[st[0], st[1], st[2], st[3], ac_idx] += \
                W / state_action_cum_sum[st[0], st[1], st[2], st[3], ac_idx] * \
                (G - state_action_values[st[0], st[1], st[2], st[3], ac_idx])
            _, _, ac_probs_t = target_policy(st)
            W = W * ac_probs_t[ac_idx] / ac_prob_b
    return state_action_values

def show_racetrack(track, ax=None, grids='all'):
    img = COLOR_MAP[track.T]
    ax = ax or plt.gca()
    ax.imshow(img, origin='lower')
    if grids == 'all':
        ax.set_xticks(np.arange(-0.5, track.shape[0], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, track.shape[1], 1), minor=True)
        ax.grid(which='minor', c='black', lw=0.5)
        ax.tick_params(which='minor', bottom=False, left=False)
    elif grids == 'non-zero':
        def rect(ax, pos):
            r = plt.Rectangle(pos - 0.5, 1, 1, fc='none', ec='black', lw=0.5)
            ax.add_patch(r)
        xidx, yidx = np.nonzero(track)
        for xpos, ypos in zip(xidx, yidx):
            rect(ax, np.array([xpos, ypos]))
    plt.show()

def plot_velocity_profiles():
    _, axs = plt.subplots(MAX_VELOCITY + 1, MAX_VELOCITY + 1, figsize=(20,20))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(MAX_VELOCITY + 1):
        for j in range(MAX_VELOCITY + 1):
            velocity = np.array([i, j], dtype=int)
            _, traj = pos_step(np.array([0, 0], dtype=int), velocity)
            track = np.zeros((MAX_VELOCITY + 1, MAX_VELOCITY + 1), dtype=int)
            track[traj[:, 0], traj[:, 1]] = 1
            show_racetrack(track, axs[i, j], grids='non-zero')
    plt.show()

def animate_trajectory(track_id, state_action_values):
    # Target policy.
    def target_policy(state):
        values_ = state_action_values[state[0], state[1], state[2], state[3], :]
        max_v_ = np.max(values_)
        max_ac_idx_ = [idx_ for idx_, value_ in enumerate(values_) if value_ == max_v_]
        ac_probs_ = np.ones_like(values_) * EPSILON_T_DEPLOY
        ac_probs_[values_ == -np.Inf] = 0
        ac_probs_[max_ac_idx_] += (1 - np.sum(ac_probs_)) / len(max_ac_idx_)
        idx_ = np.random.choice(len(values_), p=ac_probs_)
        return ACTIONS[idx_]

    # Initialize state and compute trajectory.
    rand_idx = np.random.choice(START_POS[track_id].shape[0])
    pos = START_POS[track_id][rand_idx]
    vel = np.zeros((2,), dtype=int)
    state = np.concatenate((pos, vel))
    if DISPLAY_VELOCITY:
        poses = [[pos.tolist()]]
    else:
        poses = [pos.tolist()]
    finished = False
    iter = 0
    while (not finished) and iter < 100:
        ac = target_policy(state)
        state, finished, traj = step(track_id, state, ac, noise_prob=0)
        if DISPLAY_VELOCITY:
            poses.append(traj[1:, 0:2].reshape(-1, 2).tolist())
        else:
            for st in traj[1:, :].tolist():
                poses.append(st[0:2])
        iter += 1
    
    # Initialize animation background.
    background_track = RACETRACKS[track_id]
    if DISPLAY_VELOCITY:
        p_ = poses[0][0]
    else:
        p_ = poses[0]
    track = background_track.copy()
    track[p_[0], p_[1]] = GRID_CAR
    img = COLOR_MAP[track.T]
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    im = ax.imshow(img, interpolation='none', origin='lower')
    def rect(ax, pos):
        r = plt.Rectangle(pos - 0.5, 1, 1, fc='none', ec='black', lw=0.5)
        ax.add_patch(r)
    xidx, yidx = np.nonzero(background_track)
    for xpos, ypos in zip(xidx, yidx):
        rect(ax, np.array([xpos, ypos]))
    def animate_func(i):
        track = background_track.copy()
        if DISPLAY_VELOCITY:
            for p_ in poses[i]:
                track[p_[0], p_[1]] = GRID_CAR
        else:
            p_ = poses[i]
            track[p_[0], p_[1]] = GRID_CAR
        img = COLOR_MAP[track.T]
        im.set_array(img)
        return [im]
    anim = animation.FuncAnimation(fig, animate_func, frames = len(poses), interval = 1000 / FPS)
    plt.show()
    

if __name__ == '__main__':
    # plot_velocity_profiles()
    # show_racetrack(RACETRACK1, grids='non-zero')
    # show_racetrack(RACETRACK2, grids='non-zero')
    
    track_id = int(0)
    file_str = './racetrack_data/q_func_track_' + str(track_id) + '_eplen_' + str(EPSIODES[track_id]) + \
        '_epsb_' + str(EPSILON_B) + '_epst_' + str(EPSILON_T) + '.npy'
    if not os.path.isfile(file_str):
        state_action_values = off_policy_mc(track_id, EPSIODES[track_id])
        np.save(file_str, state_action_values)
    else:
        state_action_values = np.load(file_str)
    animate_trajectory(track_id, state_action_values)
