import numpy as np
import matplotlib.pyplot as plt
from os import path

# 1. load data
current_dir = path.dirname(__file__)
data = np.load(path.join(current_dir, 'data/data.npz'))
X = data['X']
d = data['d']

# 2. pre-process, check if there is nan
print(np.nan in X)
print(np.nan in d)

# 3. scatter
print("X Shape: {}; d Shape{}".format(np.shape(X), np.shape(d)))
plt.subplot(2, 1, 1)  # row , column, number
plt.scatter(X[:, 0], d[:, 0])


# 4. build a model
# f为非线性函数
# y = ax^2 + bx + c
# loss = (y -real)^2


def model(x, w):
    a, b, c = w
    return a * x ** 2 + b * x + c


def grad(x, d, w):
    a, b, c = w
    y = model(x, w)
    grad_loss_a = 2 * (y - d) * x ** 2
    grad_loss_b = 2 * (y - d) * x
    grad_loss_c = 2 * (y - d)
    return grad_loss_a, grad_loss_b, grad_loss_c


# 5. train
w = np.random.normal(0, 1, [3])  # initial value
eta = 0.01  # learn speed
batch_size = 100  # 32-128

loss = []
selected_points = np.random.randint(0, len(X), [batch_size])
for itr in range(1000):
    sum_grad_a, sum_gard_b, sum_grad_c = 0, 0, 0
    for _ in range(batch_size):
        idx = np.random.randint(0, len(X))  # random index
        inx = X[idx]
        ind = d[idx]
        grad_a, grad_b, grad_c = grad(inx, ind, w)
        sum_grad_a += grad_a
        sum_gard_b += grad_b
        sum_grad_c += grad_c
    w[0] -= eta * sum_grad_a / batch_size
    w[1] -= eta * sum_gard_b / batch_size
    w[2] -= eta * sum_grad_c / batch_size
    loss.append(np.mean((model(X[selected_points], w) - d[selected_points]) ** 2))

# 6. 绘图
x = np.linspace(-4, 6, 1000)
y = model(x, w)
print(w)

plt.plot(x, y, lw=3, color='#ff0000')

# 7. 分析收敛过程
plt.subplot(2, 1, 2)
plt.plot(loss)

plt.show()
