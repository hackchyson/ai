from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
import tensorflow as tf

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


current_dir = path.dirname(__file__)
data_path = path.join(current_dir, 'data/train.csv')

# 1. load data
data = pd.read_csv(data_path)
pd.set_option('display.max_columns', None)  # to show all the columns


# 2. observe the data
# print(data.shape)
# print(data.info())
# print(data.describe())
# print(data.head(3))


# 3. basic feature pretreatment
# 3.1 null age
def set_missing_ages(df):
    # age and features used to predict absent age
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # split known age and unknown age, i.e. train and test
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    # age column
    y = known_age[:, 0]
    # feature columns
    X = known_age[:, 1:]

    # model
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    predict_ages = rfr.predict(unknown_age[:, 1:])

    # .loc[] is primarily label based, but may also be used with a boolean array.
    #  Single tuple for the index with a single label for the column
    df.loc[(df.Age.isnull()), 'Age'] = predict_ages
    return df


# 3.2 null cabin
def set_cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'
    return df


# 3.3 object with one hot encode
def set_one_hot(df, obj_list):
    for obj in obj_list:
        dummies_tmp = pd.get_dummies(df[obj], prefix=obj)
        df = pd.concat([df, dummies_tmp], axis=1)
        df.drop([obj], axis=1, inplace=True)
    return df


# 3.4 filter
def set_filter(df):
    df = df.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    return df


# combine all the feature engineering
def feature_engineering(df):
    df = set_missing_ages(df)
    df = set_cabin_type(df)
    df = set_one_hot(df, ['Cabin', 'Embarked', 'Sex', 'Pclass'])
    df = df.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    df = df.values
    return df


data = feature_engineering(data)
y = data[:, 0]
X = data[:, 1:]
label_onehot = []
for label in y:
    if label == 1:
        label_onehot.append([1, 0])
    else:
        label_onehot.append([0, 1])
y = np.array(label_onehot)
print(X.shape, y.shape)

# 4. placeholder
input_x = tf.placeholder(tf.float32, [None, 14], name='input_x')
input_y = tf.placeholder(tf.int32, [None, 2], name='input_y')

# 5. model
net = tf.layers.dense(input_x, 6)
# for i in range(5):
#     net = tf.layers.dense(net, 4)
logits = tf.layers.dense(net, 2)
y_predict = tf.argmax(tf.nn.softmax(logits), 1)

# 6. loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=input_y)
loss = tf.reduce_mean(cross_entropy)

# 7. optimizer
optimize = tf.train.AdamOptimizer().minimize(loss)

# 8. accuracy
correct_pred = tf.equal(tf.argmax(input_y, axis=1), y_predict)
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 8. init
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 9. train
for i in range(10000):
    idx = np.random.randint(0, len(X), 300)
    inx = X[idx]
    iny = y[idx]
    sess.run(optimize, feed_dict={input_x: inx, input_y: iny})
    if i % 1000 == 0:
        accuracy = sess.run(acc, feed_dict={input_x: inx, input_y: iny})
        print({'iteration: {:>6} ;  accuracy: {:>6}'.format(i, accuracy)})

# {'iteration:      0 ;  accuracy: 0.33666667342185974'}
# {'iteration:   1000 ;  accuracy: 0.8266666531562805'}
# {'iteration:   2000 ;  accuracy: 0.8066666722297668'}
# {'iteration:   3000 ;  accuracy: 0.8333333134651184'}
# {'iteration:   4000 ;  accuracy: 0.8100000023841858'}
# {'iteration:   5000 ;  accuracy: 0.8500000238418579'}
# {'iteration:   6000 ;  accuracy: 0.8199999928474426'}
# {'iteration:   7000 ;  accuracy: 0.7933333516120911'}
# {'iteration:   8000 ;  accuracy: 0.7833333611488342'}
# {'iteration:   9000 ;  accuracy: 0.8033333420753479'}
