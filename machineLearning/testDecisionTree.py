from machineLearning import DecisionTrees
myDat,labels = DecisionTrees.createDataSet()
myTree = DecisionTrees.createTree(myDat,labels)
print(myTree)