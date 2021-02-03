import random, numpy


def Q_learning_demo():
    # Learning rate
    alpha = 1.0

    # Discount factor
    gamma = 0.8

    epsilon = 0.2
    num_episodes = 500

    R = numpy.array(
        [
            [-1, 0, -1, -1, -1, 0],
            [-50, -1, 0, -1, 0, -1],
            [-1, 0, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, 0, -1, 100, -1, 0],
            [-50, -1, -1, -1, 0, -1],
        ]
    )

    # Initialize Q
    Q = numpy.zeros((6, 6))

    # Run for each episode
    for _ in range(num_episodes):

        # Randomly choose an initial state
        s = 2
        while s != 3:
            # Get all the possible actions
            actions = [a for a in range(6) if R[s][a] != -1]

            # Epsilon-greedy
            if numpy.random.binomial(1, epsilon) == 1:
                a = random.choice(actions)
            else:
                a = actions[numpy.argmax(Q[s][actions])]

            next_state = a

            # Update Q (s, a)
            Q[s][a] += alpha * (R[s][a] + gamma * numpy.max(Q[next_state]) - Q[s][a])

            # Go to the next state
            s = next_state

    return Q


if __name__ == "__main__":
    Q = Q_learning_demo()
    print(Q)