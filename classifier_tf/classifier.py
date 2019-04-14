import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# tensorflow graph
g = tf.Graph()

# load data
data = np.load('data/data.npz')
data_x = data['X']  # (2000, 2)
data_label = data['d']  # (2000,)
# print(data_x[:10])
# print(data_label[:10])

# onehot
label_onehot = []
for label in data_label:
    if label == 1:
        label_onehot.append([1, 0])
    else:
        label_onehot.append([0, 1])
label_onehot = np.array(label_onehot)

# placeholder to hold the data
x = tf.placeholder(tf.float32, [None, 2])
label = tf.placeholder(tf.float32, [None, 2])

# layer
# output = activation(input * kernel + bias)
net = tf.layers.dense(x, 4, activation=tf.nn.relu)
for i in range(3):
    net = tf.layers.dense(net, 4, activation=tf.nn.relu)

# predict label
y = tf.layers.dense(net, 2)  # y is the predict value

# loss
# 将网络的输出转换为概率
p = tf.nn.softmax(y, axis=1)
# 计算交叉熵
qlogp = tf.reduce_sum(- label * tf.log(p), axis=1)
loss = tf.reduce_mean(qlogp)

# train step
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# accuracy
correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(label, axis=1))  # 1,0
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# init
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# graph writer
train_writer = tf.summary.FileWriter('classifier-logdir', sess.graph)

# iteration
xy = []
start_time = time.time()
for itr in range(1000):
    idx = np.random.randint(0, 2000, 200)
    inx = data_x[idx]
    ind = label_onehot[idx]
    sess.run(train_step, feed_dict={x: inx, label: ind})

    if itr % 100 == 0:
        acc = sess.run(accuracy, feed_dict={x: data_x, label: label_onehot})
        duration = time.time() - start_time
        xy.append([itr, duration, acc])
        print('step: {:>6} , accuracy: {}'.format(itr, acc))

# plot
xy = np.array(xy)
plt.subplot(2, 1, 1)
for i in xy:
    plt.text(i[0], i[2], float('%.3f' % i[2]), horizontalalignment='center')
plt.plot(xy[:, 0], xy[:, 2], marker='o')
plt.xlabel('steps')
plt.ylabel('accuracy(a.u.)')

plt.subplot(2, 1, 2)
for i in xy:
    plt.text(i[0], i[1], float('%.2f' % i[1]))
plt.plot(xy[:, 0], xy[:, 1], marker='o')
plt.xlabel('step')
plt.ylabel('total time(s)')

plt.show()
