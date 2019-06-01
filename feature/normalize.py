from sklearn.preprocessing import Normalizer

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
print('data: ', data)
normalizer = Normalizer()
normalizer.fit(data)
print(normalizer.transform(data))
print(normalizer.transform([[2, 2]]))


