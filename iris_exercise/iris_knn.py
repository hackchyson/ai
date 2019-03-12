from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 1) load data
iris = load_iris()

# 2） observe data
print(type(iris))  # <class 'sklearn.utils.Bunch'> Bunch类似dict，对dict进行了封装。
print(iris.keys())  # dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(type(iris.data))  # <class 'numpy.ndarray'>
print(type(iris.target))  # <class 'numpy.ndarray'>
data = iris.data
target = iris.target
print(data.shape)  # (150, 4) 150行4列
print(target.shape)  # (150,) 150行1列
print(data[:3, :])
print(target[:3])

# 3) feature engineering
# 数据已经可以进行模型训练，可直接进行模型训练
# 根据预测结果选择需不需要进行特征工程

data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=4)  # 数据随机切分成训练集和测试集

# 4) model train
knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)  # 查看api或源码，清楚参数含义
knn.fit(data_train, target_train)

# 5) train score
print('knn train score', knn.score(data_train, target_train))  # knn score 0.9732142857142857

# 6) test score
print('knn test score', knn.score(data_test, target_test))  # knn test score 0.9736842105263158

# 7）optimization
k_range = range(1, 20)
k_score = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(data_train, target_train)
    score = knn.score(data_test, target_test)
    print('knn test score {score} with k = {k} '.format(**locals()))
    k_score.append(score)

# 8) optional. 绘图帮助分析。
plt.plot(k_range, k_score, marker='o')
plt.xlabel('k')
plt.ylabel('score')
plt.show()
