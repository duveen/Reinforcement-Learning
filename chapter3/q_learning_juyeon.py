import random, numpy


def Q_learning_demo():
    alpha = 1.0
    gamma = 0.8
    epsilon = 0.2
    num_episodes = 200
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

    Q = numpy.zeros((6, 6))

    for _ in range(num_episodes):
        s = numpy.random.choice(5)
        while s != 3:
            actions = [a for a in range(6) if R[s][a] != -1]
            if numpy.random.binomial(1, epsilon) == 1:
                a = random.choice(actions)
            else:
                a = actions[numpy.argmax(Q[s][actions])]
            next_state = a
            Q[s][a] += alpha * (R[s][a] + gamma * numpy.max(Q[next_state]) - Q[s][a])
            s = next_state
    return Q


result = Q_learning_demo()

print(result)
