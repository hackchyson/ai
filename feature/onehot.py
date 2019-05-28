import pandas as pd

df = pd.DataFrame(['a', 'b', 'a', 'c'], columns=['char'])
print(df.values)

onehot = pd.get_dummies(df['char'], prefix='char')
print(onehot.values)
