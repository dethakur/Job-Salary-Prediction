# filePath = 'data-sets/temporary.csv'

from sklearn.feature_extraction.text import *
import csv
import scipy.sparse as sp

import pickle

vectorizerMap = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/vectorizer-{}.pkl'.format(vecture_features),'rb'))
vecture_features = 2000

# Path for training data csv file
filePath = '/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/data-sets/Test_rev1.csv'

train_data = []
test_data_index = []

f_file = open(filePath)
l_val = list(csv.reader(f_file,delimiter=","))



import numpy as np


map_el = {}
test_map_el = {}
X = {}
X_test = {}
Y = []
Y_test = []

feature_arr = []
for line in l_val[0:1]:
    for i in range(0,len(line)):
        feature_arr.append(line[i])


count =1
for line in l_val[1:]:
    for i in range(0,len(line)):
        key = feature_arr[i]
        if key not in map_el:
            map_el[key] = []

        map_el[key].append(line[i])

    count +=1

del map_el['Id']
del map_el['SalaryRaw']
del map_el['ContractType']
del map_el['SalaryNormalized']

array_indices = []

for key in map_el.keys():
    value = map_el[key]
    X[key] = vectorizerMap[key].fit_transform(value)



output_arr = []
for key in X.keys():
    output_arr.append(X[key])

X = sp.hstack(output_arr, format='csr')
print(X.shape)

pickle.dump(X,open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/X-true-test.pkl','wb'))