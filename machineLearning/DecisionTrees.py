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
        prob = float(lableCounts[key])/numEntries ##计算Pi概率
        shannonEnt -= prob * np.log2(prob) #核心公式
    return shannonEnt
# myDat,labels = createDataSet()
# ShannonEnt = calShannonEnt(myDat) ##熵越高，混合数据越多，即度量集合的无序程度
# print(ShannonEnt)

#按照特征值划分数据集
def splitDataSet(dataSet,axis,value):
    """

    :param dataSet:数据集
    :param axis: 划分数据集的列
    :param value: 返回的
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if(featVec[axis] == value):
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的划分方式
def chooseBestFeatureToSplit(dataSet):
    '''

    :param dataSet:传入的list，每一项最后一列为label，前面均为特征向量
    :return:
    '''
    numFeatures = len(dataSet[0])-1 ##计算特征数量
    baseEntropy = calShannonEnt(dataSet) ##初始香农熵
    bestInfoGain = 0.
    bestFeature = -1 ##当不满足时不再划分
    for i in range(numFeatures):
        featList = [] #特征集合
        for e in dataSet:
            for j in range(numFeatures):
                featList.append(e[j]) ##加入特征值集合
        # print(featList)
        uniqueVals = set(featList)
        print(uniqueVals)
        newEntropy = 0.
        for value in uniqueVals:
            subDataset = splitDataSet(dataSet,i,value) ##分别划分数据集
            prob = len(subDataset)/float(len(dataSet)) ##计算概率
            newEntropy += prob * calShannonEnt(subDataset)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain): ##擂台法
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
myDat,labels = createDataSet()
a = chooseBestFeatureToSplit(myDat)
print(a)