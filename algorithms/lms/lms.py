"""
LMS
y = w1 * x1 + w2 * x2
"""

import numpy as np

alpha = .1  # learning rate
X = np.array([[1., 1.],
              [1., 0.],
              [0., 1.],
              [0., 0.]])  # input
y = np.array([[1.],
              [1.],
              [0.],
              [0.]])  # expect output
theta = np.array([0.,
                  0.])  # weight
epsilon = .005
max_itr = 20


def get_grad(j):
    randint = np.random.randint(0, len(X))
    h_theta = X[randint][0] * theta[0] + X[randint][1] * theta[1]
    return (y[randint] - h_theta) * X[randint][j]


for itr in range(100):
    tmp = theta[0]
    theta[0] = theta[0] + alpha * (get_grad(0))

    theta[1] = theta[1] + alpha * (get_grad(1))
    print(theta)

# 可以改成分别迭代，加入epsilon
