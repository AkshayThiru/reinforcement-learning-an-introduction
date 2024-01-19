from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from ex_2_5 import RandomWalkBandit

SAVE_FIG = True
dir = str(Path(__file__).parent.absolute().as_posix())


def simulate(runs, time, bandits):
    file = dir + "/ex_2_10_data/rewards.csv"
    if not Path(file).is_file():
        mean_rewards = np.zeros((len(bandits), 2))
        np.savetxt(file, mean_rewards, delimiter=',')
    with open(file, 'rb') as f:
        mean_rewards = np.loadtxt(f, delimiter=',')
    
    for i, bandit in enumerate(bandits):
        if mean_rewards[i, 0] == 1:
            continue
        print(f"Solving bandit case {i}")
        cur_mean_reward = np.zeros((time,))
        for r in trange(runs):
            cur_reward = np.zeros((time,))
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                cur_reward[t] = bandit.step(action)
            cur_mean_reward += 1/(r+1) * (cur_reward - cur_mean_reward)
        mean_rewards[i, 1] = np.mean(cur_mean_reward[-int(time/2):])
        mean_rewards[i, 0] = 1
        with open(file, 'wb') as f:
            np.savetxt(file, mean_rewards, delimiter=',')
    return mean_rewards[:, 1]


def solution(runs=2000, time=int(2e5)):
    labels = [
        "$\epsilon$-greedy-sample-average",
        "epsilon-greedy-constant-step",
        "gradient bandit",
        "UCB",
        "optimistic initialization",
    ]
    generators = [
        lambda epsilon: RandomWalkBandit(epsilon=epsilon, sample_averages=True),
        lambda epsilon: RandomWalkBandit(epsilon=epsilon, step_size=0.1),
        lambda alpha: RandomWalkBandit(
            gradient=True, step_size=alpha, gradient_baseline=True
        ),
        lambda coef: RandomWalkBandit(epsilon=0, UCB_param=coef, sample_averages=True),
        lambda initial: RandomWalkBandit(epsilon=0, initial=initial, step_size=0.1),
    ]
    parameters = [
        np.arange(-7, -1, dtype=np.float64),
        np.arange(-7, -1, dtype=np.float64),
        np.arange(-5, 2, dtype=np.float64),
        np.arange(-4, 3, dtype=np.float64),
        np.arange(-2, 3, dtype=np.float64),
    ]

    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))

    rewards = simulate(runs, time, bandits)

    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i : i + l], label=label)
        i += l
    plt.xlabel("Parameter($2^x$)")
    plt.ylabel("Average reward")
    plt.legend()

    if SAVE_FIG:
        plt.savefig(dir + "/../images/ex_2_10.png")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    solution()
