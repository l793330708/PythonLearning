import numpy as np
import operator
## 模拟数据集
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0.,0.],[0.,0.1]])
    labels = ['A','A','B','B']
    return group,labels
## 获取数据集
group,labels = createDataSet()
##分类器
def classifier(inX,dataSet,labels,k):
    '''
    :param inX: 输入的点
    :param dataSet: kNN已知参数集合
    :param labels: kNN已知参数集合对应lables
    :param k: 取前k个最近距离统计
    :return: 辨识类别
    '''
    dataSetSize = dataSet.shape[0] #获取数据集内点数
    # print(dataSetSize)
    ##距离计算
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = np.square(diffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    distances = np.sqrt(sqDistances)
    print(distances)
    sortedDistindicies = np.argsort(distances) ##argsort()
    print(sortedDistindicies)
    classCount = {}
    ## 选取距离最小的k个点
    for i in range(k):
        votellabel = labels[sortedDistindicies[i]]
        classCount[votellabel] = classCount.get(votellabel,0)+1
    rs = max(classCount,key=classCount.get) ##返回分类最可能结果
    # sortedClassCount = sorted(classCount,key=operator.itemgetter(1),reverse=True)
    return rs
rs =  classifier([2,3],group,labels,3)
print(rs)