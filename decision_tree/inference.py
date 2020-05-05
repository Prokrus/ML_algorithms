import algorithm as alg
import numpy as np
import pickle

def classify(dTree, dataVec):
    key = list(dTree.keys())[0]
    subnodes = dTree[key]
    feature = dataVec[key-1]
    nextTree = subnodes[feature]
    if(type(nextTree).__name__=='dict'):
        label = classify(nextTree, dataVec)
    else:
        label = nextTree
    return label

datavec = np.array([1,1])
tree_file = 'test_tree.txt'
tree = alg.loadTree(tree_file)
label = classify(tree, datavec)
print(label)
