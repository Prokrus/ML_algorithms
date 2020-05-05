import numpy as np
import pickle

def calcshannon(dataset):
    data_num = dataset.shape[0]
    labelcount = {}
    for data in dataset:
        label = data[0]
        labelcount[label]=labelcount.get(label, 0) +1
    shannon_value = 0
    for key in labelcount:
        p = labelcount[key]/data_num
        shannon = -np.log(p)/np.log(2)
        shannon_value = shannon_value + shannon

    return shannon_value

def splitdataset(dataset, feature_index):
    if(feature_index ==0):
        print("data index 0 is labels, threfore feature index must greater than 0")
    feature = dataset[:,feature_index]
    feature_values = list(set(feature))
    sets = {}
    for i in feature_values:
        sets[i] = dataset[dataset[:,feature_index]==i,:]
    return sets

def determine_feature(dataset):
    shannon_v = calcshannon(dataset)
    num = dataset.shape[0]
    feature_num =dataset.shape[1]-1
    shannon_reduce = []
    for i in range(feature_num):
        temp_subsets = splitdataset(dataset,i+1)
        sum_shannon = 0
        for subset_key in temp_subsets.keys():
            temp_shannon = calcshannon(temp_subsets[subset_key])
            prob = temp_subsets[subset_key].shape[0]/float(num)
            sum_shannon = sum_shannon +prob*temp_shannon
        shannon_reduce.append(shannon_v - sum_shannon)
    shannon_array = np.array(shannon_reduce)
    feature_index = np.argmax(shannon_array)
    return feature_index +1

def majorityC(labellist):
    label_array = np.array(labellist)
    minv = np.min(label_array)
    temp_array = label_array - minv
    label = np.argmax(np.bincount(temp_array))+minv
    return label

def createTree(dataset):
    classlist = [temp[0] for temp in dataset]
    if(len(set(classlist)) == len(classlist)):
        return classlist[0]
    feature_array = dataset[:,1:]
    if((feature_array == feature_array[0]).all()):
        return majorityC(classlist)
    best_feature = determine_feature(dataset)
    data_sets = splitdataset(dataset,best_feature)
    dTree = {best_feature:{}}
    for key in data_sets.keys():
        dTree[best_feature][key] = createTree(data_sets[key])
    return dTree

def storeTree(dTree, filename):
    fw = open(filename, 'wb')
    pickle.dump(dTree,fw)
    fw.close()

def loadTree(filename):
    fr = open(filename,'rb')
    return pickle.load(fr) 

dataset = np.array([[1,1,1],[1,1,1],[0,1,0],[0,0,1],[0,0,1]])
#print(dataset)
tree = createTree(dataset)
#print(tree)
filename = 'test_tree.txt'
storeTree(tree, filename)



