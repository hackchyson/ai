from sklearn.neighbors import KNeighborsClassifier
import numpy as np

np.random.seed(1)
X = np.random.randint(0, 10, [10, 3])
y = np.random.randint(0, 4, [10])
print(X, y, sep='\n')

knc = KNeighborsClassifier()
knc.fit(X, y)

test = np.array([[1, 2, 3]])
predict = knc.predict(test)
print(predict)
