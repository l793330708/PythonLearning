import numpy as np
'''
 base DecisionTree
'''

def getSourceMat(filename):
    fr = open(filename)
    flines = fr.readlines()
    dataLen = len(flines)
    labels = ['buying', 'maintenance', 'doors', 'persons' , 'lug_boot' , 'safety']
    dataMat = []
    for i in range(dataLen):
         lineArr = flines[i].strip().split(',')
         dataMat.append(lineArr)
    np.array(dataMat)
    return dataMat, labels


dataMat, labels = getSourceMat("E:\\car.data")

def splitData2Mat(dataSet, axis, value):
    resultArr = []
    for example in dataSet:
        if example[axis] == value:
            returnVec = example[:axis]
            returnVec.extend(example[axis+1:])
            resultArr.append(returnVec)
    return  resultArr


# rsArr = splitData2Mat(dataMat, 1 , 'med')
# print(rsArr)
def calShannon(dataSet):
    '''
    Shannon = - Sum(prob *log(prob))
    :param dataSet:
    :return:
    '''
    dataSetLen = np.shape(dataSet)[0]
    countClass = {}
    for example in dataSet:
        className = example[-1]
        countClass[className] = countClass.get(className, 0) + 1
    shannon = 0.
    for key in countClass:
        prob = countClass[key] / dataSetLen
        shannon -= prob * np.log2(prob)
        # print("the prob is %f,and the name is %s" % (prob, list(uniqueClass)[i]))
    return shannon


def chooseBestFeature(dataSet):
    totalLen = len(dataSet)
    baseShannon = calShannon(dataSet)
    bestinfoGain = 0.
    bestFeature = -1
    featureNum = np.shape(dataSet)[1] - 1
    for i in range(featureNum):
        featureList = [example[i] for example in dataSet]
        uniqueFeatVal = set(featureList)
        newShannon = 0.
        for feat in uniqueFeatVal:
             returnVec = splitData2Mat(dataSet, i , feat)
             subprob = len(returnVec) / totalLen
             newShannon += subprob *calShannon(returnVec)
        infoGain = baseShannon - newShannon
        if(infoGain > bestinfoGain):
            bestFeature = i
            bestinfoGain = infoGain
    return bestFeature

def majorityVote(classList):
    classCount = {}
    for className in classList:
        classCount[className] = classCount.get(className,0) +1
    return max(classCount,key= classCount.get())


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    classSet = set(classList)
    if len(classSet) == 1:
        return classList[0]
    if len(labels) == 0:
        return majorityVote(classList)
    bestFeat = chooseBestFeature(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitData2Mat(dataSet, bestFeat, value), subLabels)
    return myTree


myTree = createTree(dataMat , labels)
from machineLearning import  treePloter

treePloter.createPlot(myTree)