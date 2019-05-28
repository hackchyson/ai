import numpy as np
import operator
import collections


def create_data_set():
    # 四组二维特征(接吻，打斗）
    group = np.array([[5, 115], [7, 106], [56, 11], [66, 9]])
    # 四组对应标签
    labels = ('动作片', '动作片', '爱情片', '爱情片')
    return group, labels


def classify(int_x, data_set, labels, k):
    """
    knn
    :param int_x:
    :param data_set:
    :param labels:
    :param k:
    :return:
    """

    data_size = data_set.shape[0]

    # 将intX在横向重复data_size次，纵向重复1次
    # 例如int_x=([1,2])--->([[1,2],[1,2],[1,2],[1,2]])便于后面计算
    # tile: 瓦片
    diff_mat = np.tile(int_x, (data_size, 1)) - data_set
    # 二维特征相减后乘方
    square_diff_mat = diff_mat ** 2
    # 计算距离
    seq_distances = square_diff_mat.sum(axis=1)
    distances = seq_distances ** 0.5
    print("distances:", distances)
    # 返回distance中元素从小到大排序后的索引
    sort_distance = distances.argsort()
    print("sort distance:", sort_distance)

    class_count = collections.defaultdict(int)
    # key: label
    # value: vote count
    for i in range(k):
        # 取出前k个元素的类别
        vote_label = labels[sort_distance[i]]
        print("第{}个voteLabel={}".format(i, vote_label))
        class_count[vote_label] = class_count[vote_label] + 1

    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    print("sorted_class_count:", sorted_class_count)
    return sorted_class_count[0][0]


if __name__ == '__main__':
    group, labels = create_data_set()
    test = [200, 10]
    test_class = classify(test, group, labels, 3)
    print("result: ", test_class)
