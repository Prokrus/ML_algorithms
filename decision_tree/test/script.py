import trees

dataSet, labels = trees.createDataSet()
shann = trees.calcShannonEnt(dataSet)
split_result = trees.splitDataSet(dataSet, 1, 0)
trees.chooseBestFeatureToSplit(dataSet)
print dataSet
