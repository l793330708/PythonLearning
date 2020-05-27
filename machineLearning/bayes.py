import  numpy as np
def loadDataSet():
    '''

    :return:vocabulary list,labels
    '''
    postingList =[['my','dog','has','flea','problems','help','please'],
                  ['maybe','not','take','him','to','dog','park','stupid'],
                  ['my','dalmation','is','so','cute','i','love','him'],
                  ['stop','posting','stupid','worhless','garbage'],
                  ['mr','licks','ate','my','steak','how','to','stop','him'],
                  ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1] # lables 0 contain insult words
    return postingList,classVec
def createVocalList(dataSet):
    '''
    create vocabulary list
    :param dataSet:
    :return:
    '''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) #obtain merge Set
    return list(vocabSet)
def setOfWords2Vec(vocabList,inputSet):
    '''
    Sentence to word Vec
    :param vocabList:
    :param inputSet:
    :return:
    '''
    returnVec =[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1 ##find absolute index and update
        else:
            print("the word: %s is not in my vocabulary" % word)
    return returnVec

def trainNBO(trainMatrix,trainCategory):
    '''
    利用转换的词组向量与
    :param trainMatrix:词组向量
    :param trainCategory:标注的类别
    :return:
    '''
    numTrainDocs = len(trainMatrix) ## word vector
    numWords = len(trainMatrix[0]) ##
    pAbusive = sum(trainCategory) / float(numTrainDocs) ##offensive prob
    #intialize prob
    # p0Num = np.zeros(numWords);p1Num = np.zeros(numWords) ##word counter vector
    p0Num = np.ones(numWords);p1Num = np.ones(numWords)
    p0Denom = 2.;p1Denom = 2.
    # p0Denom = 0.;p1Denom = 0. ## total words num
    for i in range(numTrainDocs):
        if(trainCategory[i]==1):
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect = p1Num / p1Denom
    # p0Vect = p0Num / p0Denom
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pclass1):
    '''
    分类函数
    :param vec2Classify:vec for classifying
    :param p0Vec: vacabulary in class 0 prob Vec
    :param p1Vec: vacabulary in class 1 prob Vec
    :param pclass1: prob of offensive
    :return:
    '''
    p1 = sum(vec2Classify * p1Vec) + np.log(pclass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1. - pclass1) ##二分类
    if p1> p0:
        return 1
    else:
        return 0
def testingNB():
    '''
    Test Function
    :return:
    '''
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocalList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    print(trainMat)
    p0V, p1V, pAb = trainNBO(trainMat, listClasses)
    testEntry = ['love','my','stupid']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,"classified as",classifyNB(thisDoc,p0V,p1V,pAb))
