import numpy as np

def createDataSet():
    groups = np.array([[1.0, 1.2], [1.2, 1.2], [0.1, 0.2], [0.1, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return groups, labels


# 获取数据集
groups, labels = createDataSet()


def classifier(inX, dataSet, labels, k):
    '''
    kNN分类器
    :param inX:输入集合 
    :param dataSet: 已知集合
    :param labels: 已知集合对应labels
    :param k: 取前k个距离最近
    :return: kNN分类结果
    '''
    dataSetSize = dataSet.shape[0]
    # 距离计算采用欧式距离
    dataSetSubs = np.tile(inX, (dataSetSize, 1)) - dataSet
    dataSetSubrs = np.square(dataSetSubs).sum(axis=1)
    dataSetSqurs = np.sqrt(dataSetSubrs)
    # 返回排序索引值
    sortIdx = dataSetSqurs.argsort()
    # 结果记录的dic
    classCount = {}
    for i in range(k):
        lablename = labels[sortIdx[i]]
        classCount[lablename] = classCount.get(lablename, 0) + 1
    rs = max(classCount, key=classCount.get(1))
    return rs


rs = classifier([0, 0], groups, labels, 3)
print(rs)
