__author__ = 'devashishthakur'

import matplotlib
from matplotlib import pyplot as plt
import pickle

__author__ = 'devashishthakur'

import matplotlib
from matplotlib import pyplot as plt

f = open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/K-means/DecisionTree/results.txt','r');
mapVal = {}
for line in f:
    arr = line.rstrip().split(",");
    cluster_id = arr[0]
    ht = arr[1]
    test_error = arr[2]
    train_error = arr[3]
    if cluster_id not in mapVal:
        mapVal[cluster_id] = {}

    if ht not in mapVal[cluster_id]:
        mapVal[cluster_id][ht]={}

    mapVal[cluster_id][ht]['train'] = train_error
    mapVal[cluster_id][ht]['test'] = test_error



for el in mapVal:
    p = mapVal[el]
    xAxis = []
    trainArr = []
    testArr = []
    for el2 in p:        
        xAxis.append(el2)
    	trainArr.append(p[el2]['train'])
    	testArr.append(p[el2]['test'])

    # print(testArr)
    # print(trainArr)
    plt.plot(xAxis,trainArr,'+',color='red',label='Training Error for cluster = {}'.format(el))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	           ncol=2, mode="expand", borderaxespad=0.)

    plt.plot(xAxis,testArr,'+',color='blue',label='CV error')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
		           ncol=2, mode="expand", borderaxespad=0.)

    plt.show()

