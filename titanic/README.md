train.csv 为 titanic 的训练数据，有如下字段：

'PassengerId',
'Survived',
'Pclass', #  passenger class
'Name',
'Sex',
'Age',
'SibSp', # siblings, spouse
'Parch', # parent, children
'Ticket',
'Fare', 
'Cabin', 
'Embarked' 

surivived 是标签，表示是否存活。


使用逻辑回归，随机森林和神经网络的准确度都在80%左右。
可以考虑调参，更换模型，做特征工程等。
