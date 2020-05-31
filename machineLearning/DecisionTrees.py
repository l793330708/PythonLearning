import numpy as np
import operator
def createDataSet():
    '''
    模拟数据集，分类
    :return:
    '''
    # dataSet = [[1,1,'yes'],
    #            [1,1,'yes'],
    #            [1,0,'no'],
    #            [0,1,'no'],
    #            [0,1,'no']]
    # labels = ['no surfacing','flippers']
    # 来个西瓜
    dataSet = [['青绿','蜷缩','浊响','是'],
               ['乌黑','蜷缩','浊响','是'],
               ['青绿','硬挺','清脆','否'],
               ['乌黑','稍蜷','沉闷','否']]
    labels = ['色泽','根蒂','敲声']
    return dataSet,labels
def calShannonEnt(dataSet):
    '''
    计算香农信息熵
    :param dataSet:数据集，要求最后一列为对应分类
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

#按照特征值划分数据集
def splitDataSet(dataSet,axis,value):
    """
    划分数据集，返回列表
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
    信息增益计算 infoGain判定BestFeature
    :param dataSet:传入的list，每一项最后一列为label，前面均为特征向量
    :return:
    '''
    numFeatures = len(dataSet[0])-1 ##计算特征数量
    baseEntropy = calShannonEnt(dataSet) ##初始香农熵
    bestInfoGain = 0.
    bestFeature = -1 ##当不满足时不再划分,信息增益为0即划分的是两个互斥事件
    for i in range(numFeatures):
        featList = [] #特征集合
        for e in dataSet:
            featList.append(e[i]) ##加入特征值集合
        uniqueVals = set(featList)
        newEntropy = 0.
        for value in uniqueVals:
            subDataset = splitDataSet(dataSet,i,value) ##分别划分数据集
            prob = len(subDataset)/float(len(dataSet)) ##计算概率
            newEntropy += prob * calShannonEnt(subDataset) ##信息增益计算
        infoGain = baseEntropy - newEntropy ##infoGain的值>=0
        if(infoGain > bestInfoGain): ##擂台法
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
# myDat,labels = createDataSet()
# a = chooseBestFeatureToSplit(myDat)
# print(a)
def majorityCnt(classList):
    '''多数表决决定类属'''
    classCount ={}
    for vote in classList:
        classCount[vote] = classCount.get(vote,0)+1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet,labels):
    classList = []
    for example in dataSet:
        classList.append(example[-1]) ##插入TraningSet的分类标签
    if classList.count(classList[0]) ==len(classList): ##种类相同返回
        return classList[0]
    if len(dataSet[0]) == 1: ##使用完所有的标签类别任不统一，返回多数
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [] ##统计bestFeatLabel内的Value
    for example in dataSet:
        featValues.append(example[bestFeat])
        uniqueVals = set(featValues)
        for value in uniqueVals: ##划分数据集
            sublabels = labels[:] ##保存当前labels
            myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),sublabels)
    return myTree
myDat,labels = createDataSet()
myTree = createTree(myDat,labels)
print(myTree)
import  matplotlib.pyplot as plt
# Draw Trees
decisionNode = dict(boxstyle='sawtooth',fc='0.8')
leafNode = dict(boxstyle='round4',fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,\
                 textcoords='axes fraction',va='center',ha='center',bbox=nodeType,\
            arrowprops=arrow_args)
def createPlot():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False)
    plotNode(U'Decisive Node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode(U'leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
# createPlot()
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())
    secondDict = myTree[firstStr[0]]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ =='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs +=1
    return numLeafs
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())
    secondDict = myTree[firstStr[0]]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ =='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:thisDepth = 1
        if thisDepth>maxDepth:maxDepth = thisDepth
    return maxDepth
# myDat,labels = createDataSet()
# myTree =createTree(myDat,labels)
# print(myTree)
def retrieveTree(i):
    # listOfTrees=[{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
    #              {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}
    #              ]
    listOfTrees = [{'根蒂': {'蜷缩': '是', '硬挺': '否', '稍蜷': '否'}}]
    return listOfTrees[i]

# myTree = retrieveTree(0)
# a= getNumLeafs(myTree)
# b = getTreeDepth(myTree)
# print(a,b)
def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 +cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString,va='center',ha='center',rotation=30)
def plotTree(myTree,parentPt,nodeTxt):
    numleafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff+(1.0+float(numleafs))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]

    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in list(secondDict.keys()):
        if type(secondDict[key]).__name__ =='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
        plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
def createPlot(inTree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()
# myDat,labels = createDataSet()
# myTree =createTree(myDat,labels)
# createPlot(myTree)
                     