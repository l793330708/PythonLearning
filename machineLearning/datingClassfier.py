import numpy as np

def getSourceMat(filename):
    fr = open(filename)
    lines = fr.readlines()
    lens = len(lines)
    dataMat = np.zeros((lens,3))
    labelMat = []
    for i in range(lens):
        linearr = lines[i].strip()
        linearr = linearr.split()
        dataMat[i, :]= linearr[:3]
        labelMat.append(int(linearr[-1]))
    fr.close()
    return dataMat,labelMat

dataMat,labelMat = getSourceMat("E:\\MLiA_SourceCode\\machinelearninginaction\\Ch02\\datingTestSet2.txt")
##获取数据集

def auto2Norm(dataMat):
    lens = np.shape(dataMat)[0]
    minA = dataMat.min(0)
    maxA = dataMat.max(0)
    rangeA = maxA - minA
    for i in range(lens):
        dataMat[i] = (dataMat[i] - minA) / rangeA
    return dataMat
# auto2Norm(dataMat)

import matplotlib.pyplot as plt
def drawScatter():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,1],dataMat[:,2],15.*np.array(labelMat),15.*np.array(labelMat))
    plt.show()
# drawScatter()
def kNNclassifier(dataMat,labelMat):
    for K in range(3,8):
        testRitio = 0.5
        dataLen = np.shape(dataMat)[0]
        dataMat = auto2Norm(dataMat)
        index = int(dataLen*testRitio)
        trainingSet = dataMat[:index]
        testSet = dataMat[index:]
        ##compare with TrainSet
        count = index
        ec = 0.
        for example in testSet:
            exampleTile = np.tile(example, (trainingSet.shape[0], 1))
            rsMat = np.sum(np.power((exampleTile - trainingSet), 2), axis=1)
            rsMat = np.sqrt(rsMat)
            sortLabelMat = list(np.argsort(rsMat))
            classCount = {}

            for i in range(K):
                labelname = labelMat[sortLabelMat.index(i)]
                classCount[labelname] = classCount.get(labelname,0)+1
            rs = max(classCount)
            # print("classfied as %d,the real is %d" %(rs ,labelMat[count]))

            if(labelMat[count] != rs ):
                ec += 1.0
            count += 1
        errorRate = ec / float(dataLen - index)
        print("The error rate is %f,and K is %d" % (errorRate, K))


kNNclassifier(dataMat,labelMat)




getSourceMat("E:\\MLiA_SourceCode\\machinelearninginaction\\Ch02\\datingTestSet2.txt")