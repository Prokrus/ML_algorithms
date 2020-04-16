import numpy as np

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
    sets = {}
    for i in range(len(feature)):
        sets[feature[i]] = sets.get(feature[i],[]) + [i]
    sub_nums = len(sets)
    subsets = []
    for key in sets.keys():
        #temp = dataset[sets[key],:]
        #print(temp)
        subsets.append(dataset[sets[key],:])

    #print(sets)
    print(subsets)

#def createtree(dataset):
    
dataset = np.array([[1,23,3,1],[3,2,5,2],[3,2,6,2]])
splitdataset(dataset, 1)
#shannon_v = calcshannon(dataset)
#print(shannon_v)
 

