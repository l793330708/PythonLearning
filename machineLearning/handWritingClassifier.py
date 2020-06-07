import numpy as np
import os


def getDataSource(DirPath):
    dirs = os.listdir(DirPath)
    labelMat = []
    lens = len(dirs)
    dataMat = np.zeros((lens,1024))
    count = 0
    for dir in dirs:
        fr = open(DirPath+"\\"+dir)
        labelMat.append(int(dir.split('_')[0]))
        for i in range(32):
            line = fr.readline()
            for j in range(32):
                dataMat[count, 32*i+j] = int(line[j])
        count += 1
    return dataMat, labelMat


##获取训练集的matrix and label
trainingDataMat, traininglabelMat = getDataSource("E:\\MLiA_SourceCode\\machinelearninginaction\\Ch02\\trainingDigits")
testDataMat, testLabelMat = getDataSource("E:\\MLiA_SourceCode\\machinelearninginaction\\Ch02\\testDigits")

def testKnn():
    for K in range(3,7):
        lens = np.shape(trainingDataMat)[0]
        lens_1 = len(testLabelMat)
        ec =0.
        count = 0
        for example in testDataMat:
            exampleMat = np.tile(example, (lens, 1))
            distantMat = np.sum(np.power((exampleMat - trainingDataMat),2),axis=1)
            distantMat = np.sqrt(distantMat)
            sortMat = list(np.argsort(distantMat))
            classCount = {}
            for k in range(K):
            #提取相应分类
                labelname = traininglabelMat[sortMat.index(k)]
                classCount[labelname] = classCount.get(labelname,0)+1
            rsLabel = max(classCount)
            # print("the classifier result is %d,the real is %d" % (rsLabel,testLabelMat[count]))
            if(rsLabel != testLabelMat[count]):
                ec += 1.0
            count += 1
        # print(count,testLabelMat[count])
        print("the error rate is %f,K is %d" % (ec/float(lens_1),K

                                                ))
# testKnn()