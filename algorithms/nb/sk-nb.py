from sklearn.naive_bayes import GaussianNB
import numpy as np

np.random.seed(1)
X = np.random.randint(0, 10, [3, 3])
print(X)

y = np.random.randint(0, 3, [3])
print(y)

gnb = GaussianNB()
gnb.fit(X, y)
predict = gnb.predict([[1, 2, 3]])
print(predict)
