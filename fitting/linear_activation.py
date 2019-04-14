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
plt.scatter(X[:, 0], d[:, 0])


# 4. build a model
# f为非线性函数
# y = f(ax+b)
# loss = (y -real)^2
# 矩阵中所有小于0的数值都转化为0
def activation(x):
    ret = np.array(x)
    ret[x < 0] = 0
    return ret


def d_activation(x):
    ret = np.zeros_like(x)
    ret[x > 0] = 1
    return ret


def model(x, w):
    a, b = w
    return activation(a * x + b)


def grad(x, real, w):
    a, b = w
    y = model(x, w)
    dy = d_activation(a * x + b)
    grad_loss_a = 2 * (y - real) * dy * x
    grad_loss_b = 2 * (y - real) * dy
    return grad_loss_a, grad_loss_b


# 5. train
w = np.random.normal(0, 1, 2)  # initial value
eta = 0.01  # learn speed
batch_size = 100

for itr in range(1000):
    sum_grad_a, sum_gard_b = 0, 0
    for _ in range(batch_size):
        idx = np.random.randint(0, len(X))  # random index
        inx = X[idx]
        ind = d[idx]
        grad_a, grad_b = grad(inx, ind, w)
        sum_grad_a += grad_a
        sum_gard_b += grad_b
    w[0] -= eta * sum_grad_a / batch_size
    w[1] -= eta * sum_gard_b / batch_size

# 6. 绘图选取最有解
x = np.linspace(-4, 4, 1000)
y = model(x, w)
print(w)
plt.plot(x, y, color='red')
plt.show()
