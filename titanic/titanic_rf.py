from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier

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

# 4. model


model = RandomForestClassifier(n_estimators=100, n_jobs=-1)


# 5. plot
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5, n_jobs=1, train_sizes=np.linspace(.05, 1., 20),
                        verbose=0, plot=True):
    """
    :param estimator:
    :param title:
    :param X: features
    :param y: label
    :param ylim: y limit
    :param cv: cross validation
    :param n_jobs: The number of jobs to run in parallel
    :param train_sizes:
    :param verbose:
    :param plot: flag to control plot or not
    :return:
    """
    # learning_curve:     Determines cross-validated training and test scores for different training set sizes.
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel('training sample num')
        plt.ylabel('scores')
        # gca: get the current axes
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color='b')  # blue
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                         color='r')  # red
        plt.plot(train_sizes, train_scores_mean, 'o-', color='b', label='score on train data set')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='r', label='score on test data set')
        plt.legend(loc='best')
        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


midpoint, diff = plot_learning_curve(model, 'learning curve', X, y)
print('训练和测试中间值为： ', midpoint)
print('训练与测试的最大差值为： ', diff)
