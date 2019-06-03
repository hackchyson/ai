import numpy as np
from pandas import DataFrame

# np.random.seed(1)
# X = np.random.randint(0, 2, [4, 4])
# print(X)
# y = np.random.randint(0, 4, [4])
# print(y)


# 颜色  风格  款式
X = [['蓝色', '小巧', '简约'],
     ['白色', '大气', '简约'],
     ['蓝色', '大气', '精致'],
     ['蓝色', '小巧', '精致'],
     ['蓝色', '大气', '简约'],
     ['蓝色', '大气', '精致']
     ]
y = [['不喜欢', '不喜欢', '喜欢', '不喜欢', '不喜欢', '喜欢']]

df = DataFrame(X, columns=['颜色', '风格', '款式'])
print(df.values)

print(df[df['颜色'] == '蓝色'])
