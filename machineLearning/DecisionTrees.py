import numpy as np
def createDataSet():
    '''
    模拟数据集，分类
    :return:
    '''
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels
def calShannonEnt(dataSet):
    '''
    计算香农信息熵
    :param dataSet:数据集
    :return:
    '''
    numEntries = len(dataSet) ##计算数据集总数
    lableCounts={} ##保存
    for featVec in dataSet:
        currentLable = featVec[-1]
        lableCounts[currentLable] = lableCounts.get(currentLable,0)+1
    shannonEnt = 0.
    for key in lableCounts:
        prob = float(lableCounts[key])/numEntries
        shannonEnt -= prob * np.log2(prob) #核心公式
    return shannonEnt
myDat,labels = createDataSet()
ShannonEnt = calShannonEnt(myDat) ##熵越高，混合数据越多，即度量集合的无序程度
print(ShannonEnt)