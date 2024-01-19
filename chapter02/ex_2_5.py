from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ten_armed_testbed import Bandit, simulate

SAVE_FIG = False
dir = str(Path(__file__).parent.absolute().as_posix())


class RandomWalkBandit(Bandit):
    # @walk_sd: standard deviation of random walk
    def __init__(self, walk_sd=0.01, **kwargs):
        super(RandomWalkBandit, self).__init__(**kwargs)
        self.walk_sd = walk_sd

    def reset(self):
        super(RandomWalkBandit, self).reset()
        self.q_true = np.zeros(self.k) + self.true_reward
        self.best_action = np.argmax(self.q_true)

    def step(self, action):
        self.q_true = self.q_true + self.walk_sd * np.random.randn(self.k)
        self.best_action = np.argmax(self.q_true)
        return super(RandomWalkBandit, self).step(action)


def solution(runs=2000, time=int(1e4)):
    step_size = 0.1
    eps = 0.1
    rw_bandits = [
        RandomWalkBandit(epsilon=eps, sample_averages=True),
        RandomWalkBandit(epsilon=eps, step_size=step_size),
    ]
    best_action_counts, mean_rewards = simulate(runs, time, rw_bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    plt.plot(mean_rewards[0], label="$sample\_averages$")
    plt.plot(mean_rewards[1], label="$step\_size = %.02f$" % (step_size))
    plt.xlabel("steps")
    plt.ylabel("average reward")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(best_action_counts[0], label="$sample\_averages$")
    plt.plot(best_action_counts[1], label="$step\_size = %.02f$" % (step_size))
    plt.xlabel("steps")
    plt.ylabel("% optimal action")
    plt.legend()

    if SAVE_FIG:
        plt.savefig(dir + "/../images/ex_2_5.png")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    solution()
