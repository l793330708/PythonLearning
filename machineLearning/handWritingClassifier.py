import numpy as np
import os


def getDataSource(DirPath):
    dirs = os.listdir(DirPath)
    labelMat = [0]* len(dirs)
    DataMat = np.array((len(dirs),32*32))
    for dir in dirs:
        labelMat.append(dir.split('_')[0])
    print(labelMat,len(labelMat))
    print(dirs)
getDataSource("E:\\MLiA_SourceCode\\machinelearninginaction\\Ch02\\testDigits")
