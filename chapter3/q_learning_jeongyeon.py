# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import random
import numpy as np


def Q_learning_prac():
    alpha = 1.0
    gamma = 0.8
    epsilon = 0.2
    num_episodes = 200

    R = np.array(
        [
            [-1, 0, -1, -1, -1, 0],
            [-50, -1, 0, -1, 0, -1],
            [-1, 0, -1, -1, -1, -1],
            [-1, -1, -1 - 1, -1, -1],
            [-1, 0, -1, 100, -1, 0],
            [-50, -1, -1, -1, 0, -1],
        ]
    )

    Q = np.zeros((6, 6))

    for _ in range(num_episodes):
        s = np.random.choice(5)
        while s != 3:
            actions = [a for a in range(6) if R[s][a] != -1]
            if np.random.binomial(1, epsilon) == 1:
                a = random.choice(actions)
            else:
                a = actions[np.argmax(Q[s][actions])]
            next_state = a
            Q[s][a] += alpha * (R[s][a] + gamma * np.max(Q[next_state]) - Q[s][a])
            s = next_state
    return Q


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    Q = Q_learning_prac()
    print(Q)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
