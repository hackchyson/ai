from os import path
import numpy as np
import matplotlib.pyplot as plt

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


# plt.show()
# there is no exceptional data in the scatter


# 4. build a model
# y = ax + b
# loss = (y -real)^2
def model(x, a, b):
    return a * x + b


# 5. train
def grad(x, real, a, b):
    y = model(x, a, b)
    grad_loss_a = 2 * (y - real) * x
    grad_loss_b = 2 * (y - real)
    return grad_loss_a, grad_loss_b


a, b = np.random.normal(0, 1, [2])  # initial value
eta = 0.01  # learn speed
batch_size = 100

# 递归求极值
# 梯度方向
# 理想的情况是，每次都把所有的数据都进行计算，得到梯度的平均值，但那样太浪费资源了
# 所以，有两种简化方式，
# 1. 每次递归随机选取一个点进行递归运算
# 2. 每次递归随机选取一部分点进行递归运算
for itr in range(1000):
    sum_grad_a, sum_gard_b = 0, 0
    for _ in range(batch_size):
        idx = np.random.randint(0, len(X))  # random index
        inx = X[idx]
        ind = d[idx]
        grad_a, grad_b = grad(inx, ind, a, b)
        sum_grad_a += grad_a
        sum_gard_b += grad_b
    a -= eta * sum_grad_a / batch_size
    b -= eta * sum_gard_b / batch_size

# 6. 绘图
x = np.linspace(-4, 4, 1000)
y = model(x, a, b)
print(a, b)
plt.plot(x, y, color='red')
plt.show()
