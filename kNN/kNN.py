import numpy as np

def classifykNN(example, trainset, labels, k):
    data_shape = trainset.shape
    example_mat = np.tile(example,(data_shape[0],1))
    example_mat = (example_mat - trainset)**2
    squar_sum = example_mat.sum(axis = 1)
    results = squar_sum**0.5
    indexes = np.argsort(results)
    k_set = indexes[:k]
    labels_count = {}
    for label_index in k_set:
        label = labels[label_index]
        labels_count[label] = labels_count.get(label,0) +1
    #sort_dict = sorted(labels_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    sort_dict = sorted(labels_count.items())
    return sort_dict[0][0]

def create_data():
    trainset = np.array([[1,2,3,4,5],[4,2,4,2,4],[6,34,56,2,2]])
    labels = ['a','b','a']
    return trainset, labels

def datanorm(train_data):
    data_num  = train_data.shape[0]
    min_features = np.amin(train_data,axis=0)
    max_features = np.amax(train_data, axis=0)

    ranges = max_features-min_features
    norm_data = train_data - np.tile(min_features,(data_num, 1))
    norm_data = norm_data/np.tile(ranges, (data_num,1))
    return norm_data, ranges, min_features

trainset, labels = create_data()

#print(trainset)
#print(labels)
norm_data,ranges, min_features = datanorm(trainset)
example = np.array([1,3,2,2,1])
print(example)
norm_example = (example - min_features)/ranges
#print(norm_example)
k = 3
label = classifykNN(norm_example,trainset, labels, k)
print("class of this example is: " + str(label))




